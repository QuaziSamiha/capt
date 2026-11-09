import os
import sys

sys.path.append(os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import random
from tqdm import tqdm
import argparse
from distutils.util import strtobool as _bool
import time
import re
import string

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from tensorboardX import SummaryWriter
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from transformers import logging
from transformers import get_linear_schedule_with_warmup
from transformers import Adafactor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

from transformers.models.t5 import *
from src.dataset import SuperGlueData
from config.model_config import Config

logging.set_verbosity_error()

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train(args):
    torch.cuda.empty_cache()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config()
    
    try:
        model_name_or_path, tokenizer_name_or_path = cfg.models[args.model_name_or_path], cfg.models[args.tokenizer_name_or_path]
    except KeyError:
        model_name_or_path, tokenizer_name_or_path = args.model_name_or_path, args.tokenizer_name_or_path

    lr = args.lr
    batch_size = args.batch_size
    epoch = args.epoch

    step_batch_size = args.step_batch_size if args.step_batch_size else batch_size
    accum_steps = 1 if step_batch_size is None else batch_size // step_batch_size

    if args.method == "prefix-tuning" or args.method == "full":
        max_length = args.max_length
    else:
        max_length = args.max_length - args.num_virtual_tokens

    log_step = 10
    log_dir = f"log/{model_name_or_path}/{args.dataset_name}/{args.method}/"
    os.makedirs(log_dir, exist_ok=True)
    
    if args.method == "gen-prompt":
        args_string = f"{lr=} {batch_size=} {epoch=} {args.num_virtual_tokens=} prompt={args.generator} {args.random_seed=}"
    else:
        args_string = f"{lr=} {batch_size=} {epoch=} {args.num_virtual_tokens=} {args.random_seed=}"
    args_string = args_string.replace('args.', "")
    tb_writer = SummaryWriter(os.path.join(log_dir, args_string))
    print(args_string)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_length)
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        
    text_to_text = True
    train_dataloader, val_dataloader, test_dataloader, num_labels, metric = prepare_dataset(args.dataset_name, tokenizer, step_batch_size, max_length=max_length, text_to_text=text_to_text, seed=args.random_seed)
    
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.prompt_len = args.num_virtual_tokens
    config.prompt_layers = args.prompt_layers
    config.generator = args.generator
    if config.pad_token_id == None:
        config.pad_token = tokenizer.pad_token
        config.pad_token_id = tokenizer.pad_token_id
    config.num_labels = num_labels

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    tuning_params = [p for n, p in list(model.named_parameters()) if p.requires_grad]
    
    total_steps = len(train_dataloader) // accum_steps * epoch
    warmup_steps = int(total_steps * args.warmup_ratio)

    for param in model.parameters():
        param.requires_grad = False
    
    if args.method == 'gen-prompt':
        gen_prompt_params = []
        
        for block in model.encoder.block:
            if hasattr(block, 'gen_prompt'):
                for name, param in block.gen_prompt.named_parameters():
                    param.requires_grad = True
                    gen_prompt_params.append(param)
        
        tuning_params = [
            {'params': gen_prompt_params, 'lr': args.lr}, 
        ]
        
        optimizer = Adafactor(tuning_params, lr=args.lr, relative_step=False, 
                            weight_decay=args.weight_decay, scale_parameter=False)
    else:
        optimizer = Adafactor(tuning_params, lr=args.lr, relative_step=False, weight_decay=args.weight_decay, scale_parameter=False)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    total_params = 0
    trainable_params = 0
        
    for name, param in model.named_parameters(): 
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.5f}%")

    scaler = GradScaler()
    step = 0
    log_loss = []
    
    max_patience = args.max_patience

    patience_loss = max_patience
    min_val_loss = 100
    max_test_result_loss = 0

    patience_score = max_patience
    max_val_score = 0
    max_test_result_score = 0

    train_time_total = 0

    start_time = time.time()

    log_path = f"{log_dir}/{args_string}/load_counts_log.txt"
    l = open(log_path, 'wt')

    print(f"---model dtype: {model.dtype}")
    
    for e in range(epoch):
        l.write(f"Epoch {e+1}\n")
        model.train()
        train_loss, step, log_loss, train_time = train_epoch(model, optimizer, scheduler, scaler, train_dataloader, accum_steps, tb_writer, step, log_step, log_loss, device, metric, tokenizer)
        train_time_total += train_time
        if args.method == 'prompt-routing':
            l.write(f"Train: {model.prompt_encoder.load_counts}\n")
            model.prompt_encoder.print_and_reset_load_counts() 
        
        model.eval()
        with torch.no_grad():
            val_loss, preds, answers = evaluate_epoch(model, scaler, val_dataloader, device, tokenizer, text_to_text, test=False)
            if args.method == 'prompt-routing':
                l.write(f"Validation: {model.prompt_encoder.load_counts}\n")
                model.prompt_encoder.print_and_reset_load_counts() 
            
        val_score = score(metric, preds, answers)
        print("Epoch {} | Train loss: {:.5f} | Validation loss: {:.5f} | Validation acc: {:.1f}".format(e+1, train_loss, val_loss, val_score*100))
        is_better_loss = (val_loss < min_val_loss)
        is_better_score = (val_score > max_val_score)

        tb_writer.add_scalar('validation/loss', val_loss, e+1)
        tb_writer.add_scalar('validation/performance', val_score, e+1)

        os.makedirs(f"models/{args.model_name_or_path}/{args.method}/{args.dataset_name}/", exist_ok=True)

        if is_better_score and patience_score > 0:
            print("Updating result due to higher validation score")
            max_val_score = val_score
            
            with torch.no_grad():
                _, preds, answers = evaluate_epoch(model, scaler, test_dataloader, device, tokenizer, text_to_text, test=True)
            test_score = score(metric, preds, answers)                
            print("Epoch {} test result: {:.2f}".format(e+1, test_score * 100))
            
            max_test_result_score = test_score
            patience_score = max_patience
            
            if args.method != "full":
                peft_model_id = f"models/{args.model_name_or_path}/{args.method}/{args.dataset_name}/{args_string}_early_stop_score"
                
            if is_better_loss:
                min_val_loss = val_loss
            
            full_model_save_path = f"t5_models/{args.model_name_or_path}/{args.generator}/{args.dataset_name}"
            os.makedirs(full_model_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(full_model_save_path, "model_state_dict.pt"))
            if hasattr(model, 'config'):
                model.config.save_pretrained(full_model_save_path)
            print("saved the model")
        
        elif is_better_loss and patience_score > 0:
            print("Updating result due to lower validation loss")
            min_val_loss = val_loss
            
            with torch.no_grad():
                _, preds, answers = evaluate_epoch(model, scaler, test_dataloader, device, tokenizer, text_to_text, test=True)
            test_score = score(metric, preds, answers)                
            print("Epoch {} test result: {:.2f}".format(e+1, test_score * 100))
            
            max_test_result_score = test_score
            patience_score = max_patience
            
            if args.method != "full":
                peft_model_id = f"models/{args.model_name_or_path}/{args.method}/{args.dataset_name}/{args_string}_early_stop_score"
            
            full_model_save_path = f"t5_models/{args.model_name_or_path}/{args.generator}/{args.dataset_name}"
            os.makedirs(full_model_save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(full_model_save_path, "model_state_dict.pt"))
            if hasattr(model, 'config'):
                model.config.save_pretrained(full_model_save_path)
            print("saved the model")
        else:
            patience_score -= 1    
            if patience_score == 0:
                with torch.no_grad():
                    _, preds, answers = evaluate_epoch(model, scaler, test_dataloader, device, tokenizer, text_to_text, test=True)
                    final_test_score = score(metric, preds, answers)                
                print("Final epoch test result: {:.2f}".format(final_test_score * 100))
                print("Patience of score-based early stopping has reached 0. No more updates.")
        
        if e == epoch - 1:
            with torch.no_grad():
                _, preds, answers = evaluate_epoch(model, scaler, test_dataloader, device, tokenizer, text_to_text, test=True)
                final_test_score = score(metric, preds, answers)                
                print("Final epoch test result: {:.2f}".format(final_test_score * 100))
        if args.method == 'prompt-routing':
            l.write(f"Test: {model.prompt_encoder.load_counts}\n")
            model.prompt_encoder.print_and_reset_load_counts() 
            print()
        print(f"patience_score: {patience_score}")
        if patience_score <= 0:
            break

    running_time = time.time() - start_time
    avg_time = train_time_total / (e+1)
    
    print("##############################################################################")
    print(args_string)
    print("Test result | Score : {:.2f}".format(max_test_result_score * 100))
    print("Took {:.2f} minutes for training".format(running_time / 60))
    print("Took {:.2f} seconds per epoch".format(avg_time))
    
    export_dir = f"results/{model_name_or_path}/{args.dataset_name}"
    os.makedirs(export_dir, exist_ok=True)
    with open(os.path.join(export_dir, f"{args.method}.txt"), 'a') as g:
        output_args = f"{args.method=} {args.dataset_name=} {args_string}".replace("args.", "")            
        g.write(output_args + '\n')
        g.write("Took {:.2f} minutes for training / {:.2f} seconds per epoch".format(running_time / 60, avg_time) + '\n')
        g.write("Test result | Score : {:.2f}".format(max_test_result_score * 100) + '\n')

    l.close()

def train_epoch(model, optimizer, scheduler, scaler, train_dataloader, accum_steps, tb_writer, step, log_step, log_loss, device, metric, tokenizer):
    st = time.time()
    model.train()
    epoch_loss = []
    a = 0
    min_len = 512
    
    for train_sample in tqdm(train_dataloader, ncols=100, desc=f"Train", leave=False):

        ids = train_sample["ids"]
        input_ids = train_sample["input_ids"].to(device)
        
        if input_ids.shape[1] < min_len:
            min_len = input_ids.shape[1]
            
        padding_id = tokenizer.pad_token_id
        
        att_mask = torch.ones_like(input_ids) * (input_ids != padding_id).long()
        labels = train_sample["label"].to(device)
        labels = labels.float() if metric == "spearman" else labels.long()

        outputs = model.forward(input_ids=input_ids, attention_mask=att_mask, labels=labels)
        try:
            model.prompt_encoder.save_load_information(ids)
        except:
            pass
        with autocast():
            lm_loss = outputs['loss']

        if args.method == "prompt-routing" and args.stochastic and args.consistency_alpha > 0:
            additional_outputs = model.forward(input_ids=input_ids, attention_mask=att_mask, labels=labels)
            lm_loss_2 = additional_outputs['loss']
            logit1, logit2 = outputs['logits'], additional_outputs['logits']
            if lm_loss == lm_loss_2:
                loss = lm_loss
            else:
                consistency_loss = symmetric_KL_loss(logit1, logit2)
                loss = lm_loss + lm_loss_2 + args.consistency_alpha * consistency_loss
        else:
            loss = lm_loss
        scaler.scale(loss).backward()

        if torch.isfinite(loss):
            loss_val = loss.item()
            epoch_loss.append(loss_val)
            log_loss.append(loss_val)
        a += 1
        if a % accum_steps == 0 or a == len(train_dataloader):
            scaler.step(optimizer)
            scheduler.step()       
            scaler.update()
            step += 1
            a = 0
        else:
            continue
        
        optimizer.zero_grad()

        if step % log_step == 0:
            log_loss = sum(log_loss) / len(log_loss)
            tb_writer.add_scalar('train/train_loss', log_loss/log_step, step//10)
            tb_writer.add_scalar('train/train_lr', optimizer.param_groups[0]['lr'], step//10)
            log_loss = []
            try:
                tb_writer.add_scalar('train/balance_loss', balance_loss, step//10)
                tb_writer.add_scalar('train/lm_loss', lm_loss, step//10)
            except:
                pass

    train_time = time.time() - st
    print("{:.2f} seconds".format(time.time()-st))
    return sum(epoch_loss) / len(epoch_loss), step, log_loss, train_time

def evaluate_epoch(model, scaler, val_dataloader, device, tokenizer, text_to_text, test):
    val_loss = []
    preds, answers = [], []
    with torch.no_grad():
        model.eval()
        for val_sample in tqdm(val_dataloader, ncols=100, desc="Test" if test else "Validation", leave=False):
            ids = val_sample["ids"]
            input_ids = val_sample["input_ids"].to(device)
            labels = val_sample["label"].to(device)
            att_mask = torch.ones_like(input_ids) * (input_ids != tokenizer.pad_token_id).long()
            outputs = model(input_ids=input_ids, attention_mask=att_mask, labels=labels, output_attentions=True)

            pred = model.generate(input_ids=input_ids, attention_mask=att_mask, max_new_tokens=10)
            
            if not test:
                try:
                    model.prompt_encoder.save_load_information(ids)
                except:
                    pass

            loss = outputs.loss
            val_loss.append(loss.item())
            
            preds.extend(tokenizer.batch_decode(pred, skip_special_tokens=True))
            labels[labels==-100] = tokenizer.pad_token_id
            answers.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))
            
    preds = [normalize_answer(p) for p in preds]
    answers = [normalize_answer(a) for a in answers]
    
    return sum(val_loss)/ len(val_loss), preds, answers


def symmetric_KL_loss(input, target, reduction='batchmean'):
    input = input.float()
    target = target.float()
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32),
                    F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
           F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32),
                    F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
    return 0.5 * loss.sum()


def score(metric, preds, answers,):
    if metric == 'spearman':
        result = spearmanr(preds, answers)[0]
        
    elif metric == 'matthews':
        result = matthews_corrcoef(answers, preds)

    elif metric == 'acc':
        correct = sum(a == b for a, b in zip(preds, answers))
        result = correct / len(preds)

    elif metric == 'f1':       
        result = f1_score(answers, preds, average='macro')

    elif metric == 'f1a':
        result = f1_score(answers, preds, average="macro")
    
    elif metric == 'acc_f1_avg':
        correct = sum(a == b for a, b in zip(preds, answers))
        acc_result = correct / len(preds)
        
        f1_result = f1_score(answers, preds, average='macro')
        
        result = (acc_result + f1_result) / 2

    return result


def prepare_dataset(dataset_name, tokenizer, batch_size, max_length, text_to_text, seed):
    superglue_dataset_names = ['boolq', 'cb', 'copa', 'multirc', 'rte_superglue', 'wic', 'semeval']
    commonsense_datasets = ['commonsense_qa']
    
    all_supported_datasets = superglue_dataset_names + commonsense_datasets
    
    if dataset_name not in all_supported_datasets:
        raise ValueError(f"Wrong dataset name '{dataset_name}'. Supported datasets: {all_supported_datasets}")
    
    dataset_metrics = {
        'multirc': 'f1a',
        'cb': 'acc_f1_avg',
        'commonsense_qa': 'acc'
    }
    
    try:
        metric = dataset_metrics[dataset_name]
    except KeyError:
        metric = 'acc'
    
    if dataset_name in all_supported_datasets:
        train_dataset = SuperGlueData(dataset_name, 'train', tokenizer, max_length=max_length, text_to_text=text_to_text)
        val_dataset = SuperGlueData(dataset_name, 'dev', tokenizer, max_length=max_length, text_to_text=text_to_text)
        test_dataset = SuperGlueData(dataset_name, 'test', tokenizer, max_length=max_length, text_to_text=text_to_text)
    else:
        raise ValueError(f"Wrong dataset name '{dataset_name}'. Try again.")
    
    train_dataset.create_dataset()
    val_dataset.create_dataset()
    test_dataset.create_dataset()
    num_labels = train_dataset.num_labels

    def collate_fn(examples):
        ids = [example["ids"] for example in examples]
        input_ids = [example["input_ids"] for example in examples]
        segment_ids = [example["segment_ids"] for example in examples]
        labels = [example["label"] for example in examples]
        padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if input_ids[0].ndim == 1:
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
            padded_segment_ids = torch.nn.utils.rnn.pad_sequence(segment_ids, batch_first=True)
        else:
            input_ids = [d.transpose(0, 1) for d in input_ids]
            segment_ids = [d.transpose(0, 1) for d in segment_ids]   
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
            padded_segment_ids = torch.nn.utils.rnn.pad_sequence(segment_ids, batch_first=True)
            padded_input_ids = padded_input_ids.transpose(1, 2)
            padded_segment_ids = padded_segment_ids.transpose(1, 2)

        if type(labels[0]) == int:
            padded_labels = torch.LongTensor(labels)
        else:
            labels = [torch.LongTensor(d) for d in labels]
            padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {"ids": torch.LongTensor(ids).contiguous(),
                "input_ids": padded_input_ids.contiguous(),
                "segment_ids": padded_segment_ids.contiguous(),
                "label": padded_labels.contiguous(),
                }
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=(0 if os.name == 'nt' else 4), worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=(0 if os.name == 'nt' else 4), worker_init_fn=seed_worker, generator=g)   
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=(0 if os.name == 'nt' else 4), worker_init_fn=seed_worker, generator=g)   
    return train_dataloader, val_dataloader, test_dataloader, num_labels, metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.02)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--step_batch_size', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--prompt_layers', type=str, default="all")
    parser.add_argument('--generator', type=str, default="capt")
    parser.add_argument('--max_patience', type=int, default=20)

    parser.add_argument('--weight_decay', type=float, default=1e-6) 
    parser.add_argument('--dataset_name', type=str, default='cola')
    parser.add_argument('--model_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='roberta-base')
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--method', type=str, choices=['gen-prompt']) 
    parser.add_argument('--num_virtual_tokens', type=int, default=1)

    parser.add_argument('--num_virtual_tokens_full', type=int, default=1)
    parser.add_argument('--routing_level', type=str, default='prompt')

    parser.add_argument('--consistency_alpha', type=float, default=0.1)
    parser.add_argument('--init_copy', type=_bool, default=True)

    parser.add_argument('--perturb_router', type=_bool, default=False)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--stochastic', type=_bool, default=False)
    parser.add_argument('--gumbel', type=_bool, default=False)
    args = parser.parse_args()

    train(args)