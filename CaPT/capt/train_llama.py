import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.append(os.getcwd())
import random
from tqdm import tqdm
import argparse
from distutils.util import strtobool as _bool
import time
import re
import string

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from tensorboardX import SummaryWriter
from scipy.stats import spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score

from transformers import get_linear_schedule_with_warmup
from transformers import Adafactor, DataCollatorWithPadding
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import LlamaPreTrainedModel, LlamaModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from src.superglue_loader_decoder import get_superglue

TASK_CONFIGS = {
    'rte_superglue': {
        'format': "Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: Does the premise entail the hypothesis?\nAnswer:",
        'label_map': {
            "not_entailment": 0,
            "entailment": 1
        },
        'orig_labels': ["not_entailment", "entailment"]
    },
    'cb': {
        'format': "Premise: {premise}\nHypothesis: {hypothesis}\nQuestion: What is the relationship between the premise and hypothesis?\nAnswer:",
        'label_map': {
            "entailment": 0,
            "contradiction": 1,
            "neutral": 2
        },
        'orig_labels': ["entailment", "contradiction", "neutral"]
    },
    'boolq': {
        'format': "Passage: {passage}\nQuestion: {question}\nAnswer:",
        'label_map': {
            "False": 0,
            "True": 1,
            "No": 0,
            "Yes": 1
        },
        'orig_labels': ["No", "Yes"]
    },
    'wic': {
        'format': "Sentence 1: {sentence1}\nSentence 2: {sentence2}\nQuestion: Does the word '{word}' have the same meaning in both sentences?\nAnswer:",
        'label_map': {
            "False": 0,
            "True": 1,
            "No": 0,
            "Yes": 1
        },
        'orig_labels': ["No", "Yes"]
    },
    'copa': {
        'format': "Premise: {premise}\nQuestion: {question}\nChoice 1: {choice1}\nChoice 2: {choice2}\nAnswer:",
        'label_map': {
            "Choice 1": 0,
            "Choice 2": 1,
            "choice1": 0,
            "choice2": 1
        },
        'orig_labels': ["Choice 1", "Choice 2"]
    },
    'multirc': {
        'format': "Paragraph: {paragraph}\nQuestion: {question}\nAnswer: {answer}\nIs this answer correct?\nAnswer:",
        'label_map': {
            "False": 0,
            "True": 1,
            "No": 0,
            "Yes": 1
        },
        'orig_labels': ["No", "Yes"]
    },
    'commonsense_qa': {
        'format': "Question: {question}\nChoices: {choices}\nAnswer:",
        'label_map': {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4
        },
        'orig_labels': ["A", "B", "C", "D", "E"]
    }
}

class LlamaClassificationDataset(Dataset):
    def __init__(self, dataset_name, split, tokenizer, max_length=512):
        self.dataset = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.metric = 'acc'
        
        if dataset_name not in TASK_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} not supported. Available datasets: {list(TASK_CONFIGS.keys())}")
        
        self.template = TASK_CONFIGS[dataset_name]['format']
        self.label_map = TASK_CONFIGS[dataset_name]['label_map']
        self.orig_labels = TASK_CONFIGS[dataset_name]['orig_labels']
        
        if dataset_name == 'multirc':
            self.metric = 'f1a'
        elif dataset_name == 'cb':
            self.metric = 'acc_f1_avg'
        
        self.data_list, self.num_labels = get_superglue(dataset_name, split, text_to_text=True)
        self.create_dataset()
    
    def create_dataset(self):
        for i, data in tqdm(enumerate(self.data_list), total=len(self.data_list), 
                            desc=f"Formatting dataset", ncols=100):
            if len(data) == 2:
                input_seq, label = data
            elif len(data) == 3:
                input_seq, label, _ = data
            else:
                raise ValueError(f"Unexpected data format: {data}")
            
            formatted_text = self.format_input(input_seq)
            
            if isinstance(label, int):
                label_index = label
            elif isinstance(label, str):
                if label in self.label_map:
                    label_index = self.label_map[label]
                else:
                    raise ValueError(f"Unknown label: {label}. Available labels: {self.label_map.keys()}")
            else:
                raise TypeError(f"Unexpected label type: {type(label)}")
            
            input_encoding = self.tokenizer(
                formatted_text,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt"
            )
            
            self.dataset.append({
                "ids": i,
                "input_ids": input_encoding.input_ids.squeeze(),
                "attention_mask": input_encoding.attention_mask.squeeze(),
                "label": label_index
            })
    
    def format_input(self, input_seq):
        if isinstance(input_seq, tuple) and len(input_seq) == 2:
            text_a, text_b = input_seq
            
            if "Premise" in self.template and "Hypothesis" in self.template:
                return self.template.format(premise=text_a, hypothesis=text_b)
            
            elif "Passage" in self.template and "Question" in self.template:
                return self.template.format(passage=text_b, question=text_a)
            
            elif "Premise" in self.template and "Choice" in self.template:
                if isinstance(text_a, str) and ("because" in text_a or "so" in text_a):
                    premise = text_a.split(' because')[0].split(' so')[0]
                    question = "What was the cause?" if "because" in text_a else "What happens next?"
                    return self.template.format(
                        premise=premise,
                        question=question,
                        choice1=text_b,
                        choice2=input_seq[1][1] if isinstance(input_seq[1], tuple) else "N/A"
                    )
                else:
                    try:
                        ((premise_joiner, choice1), (_, choice2)) = input_seq
                        premise = premise_joiner.split(' so')[0].split(' because')[0]
                        question = "What happens next?" if " so" in premise_joiner else "What was the cause?"
                        return self.template.format(
                            premise=premise,
                            question=question,
                            choice1=choice1,
                            choice2=choice2
                        )
                    except:
                        return str(input_seq)
            
            elif "Paragraph" in self.template and "Question" in self.template:
                paragraph = text_a
                qa_text = text_b
                
                if isinstance(qa_text, str) and "?" in qa_text:
                    parts = qa_text.split("?", 1)
                    question = parts[0] + "?"
                    answer = parts[1].strip() if len(parts) > 1 else ""
                    return self.template.format(
                        paragraph=paragraph,
                        question=question,
                        answer=answer
                    )
                else:
                    return self.template.format(
                        paragraph=paragraph,
                        question=qa_text,
                        answer=""
                    )
            
            elif "Sentence" in self.template and "word" in self.template:
                full_text = text_a
                
                if "Does" in full_text:
                    parts = full_text.split('Does')
                    sentences = parts[0].strip()
                    word_part = 'Does' + parts[1].strip()
                    
                    sentence_parts = sentences.split(' ')
                    mid_point = len(sentence_parts) // 2
                    sentence1 = ' '.join(sentence_parts[:mid_point])
                    sentence2 = ' '.join(sentence_parts[mid_point:])
                    
                    word_match = re.search(r"Does (\w+) have", word_part)
                    word = word_match.group(1) if word_match else "word"
                    
                    return self.template.format(
                        sentence1=sentence1, 
                        sentence2=sentence2, 
                        word=word
                    )
                else:
                    return self.template.format(
                        sentence1=text_a,
                        sentence2=text_b,
                        word="[WORD]"
                    )
            else:
                return f"Input A: {text_a}\nInput B: {text_b}"
        else:
            return str(input_seq)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "label": item["label"],
            "ids": item["ids"]
        }
    
    def get_label_mapping(self):
        return {
            "label_map": self.label_map,
            "index_to_label": {v: k for k, v in self.label_map.items()},
            "num_labels": self.num_labels,
            "original_labels": self.orig_labels
        }

def pad_sequence_collate(batch, tokenizer):
    max_length = max(item["input_ids"].size(0) for item in batch)
    
    input_ids = []
    attention_masks = []
    labels = []
    ids = []
    
    for item in batch:
        curr_len = item["input_ids"].size(0)
        padding_len = max_length - curr_len
        
        if padding_len > 0:
            padded_input_ids = torch.cat([
                item["input_ids"],
                torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            
            padded_attention_mask = torch.cat([
                item["attention_mask"],
                torch.zeros(padding_len, dtype=torch.long)
            ])
            
            input_ids.append(padded_input_ids)
            attention_masks.append(padded_attention_mask)
        else:
            input_ids.append(item["input_ids"])
            attention_masks.append(item["attention_mask"])
            
        labels.append(item["label"])
        ids.append(item["ids"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "label": torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels),
        "ids": torch.tensor(ids)
    }

def prepare_classification_datasets(dataset_name, tokenizer, batch_size, max_length, seed):
    train_dataset = LlamaClassificationDataset(dataset_name, 'train', tokenizer, max_length)
    val_dataset = LlamaClassificationDataset(dataset_name, 'dev', tokenizer, max_length)
    test_dataset = LlamaClassificationDataset(dataset_name, 'test', tokenizer, max_length)
    
    label_info = train_dataset.get_label_mapping()
    
    print(f"Dataset: {dataset_name}")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Validation examples: {len(val_dataset)}")
    print(f"  Test examples: {len(test_dataset)}")
    print(f"  Number of labels: {label_info['num_labels']}")
    print(f"  Label mapping: {label_info['label_map']}")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g,
        collate_fn=lambda batch: pad_sequence_collate(batch, tokenizer)
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=lambda batch: pad_sequence_collate(batch, tokenizer)
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=lambda batch: pad_sequence_collate(batch, tokenizer)
    )
    
    return train_dataloader, val_dataloader, test_dataloader, label_info['num_labels'], train_dataset.metric, label_info

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

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

def train_epoch(model, optimizer, scheduler, scaler, train_dataloader, accum_steps, 
                tb_writer, step, log_step, log_loss, device, metric):
    st = time.time()
    model.train()
    epoch_loss = []
    a = 0
    
    for batch in tqdm(train_dataloader, ncols=100, desc=f"Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        labels = labels.float() if metric == "spearman" else labels.long()
        
        with autocast():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            loss = outputs.loss
            
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

        if step % log_step == 0 and log_loss:
            avg_log_loss = sum(log_loss) / len(log_loss)
            tb_writer.add_scalar('train/train_loss', avg_log_loss, step//10)
            tb_writer.add_scalar('train/train_lr', optimizer.param_groups[0]['lr'], step//10)
            log_loss = []
            
    train_time = time.time() - st
    print(f"Training took {train_time:.2f} seconds")
    return sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0, step, log_loss, train_time

def evaluate_epoch(model, val_dataloader, device, test=False):
    val_loss = []
    preds, answers = [], []
    
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_dataloader, ncols=100, desc="Test" if test else "Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            loss = outputs.loss
            val_loss.append(loss.item())
            
            logits = outputs.logits
            if logits.shape[1] > 1:
                pred = torch.argmax(logits, dim=-1)
            else:
                pred = (logits > 0).long()
                
            preds.extend(pred.cpu().tolist())
            answers.extend(labels.cpu().tolist())
            
    return sum(val_loss) / len(val_loss) if val_loss else 0, preds, answers

def score(metric, preds, answers):
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

def train(args):
    seed_everything(args.random_seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_path = args.model_name_or_path
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epoch
    max_length = args.max_length

    step_batch_size = args.step_batch_size if args.step_batch_size else batch_size
    accum_steps = 1 if step_batch_size is None else batch_size // step_batch_size

    log_step = 10
    log_dir = f"log/{os.path.basename(model_path)}/{args.dataset_name}/{args.generator}/"
    os.makedirs(log_dir, exist_ok=True)
    
    args_string = f"{lr=} {batch_size=} {epochs=} {args.num_virtual_tokens=} {args.random_seed=}"
    args_string = args_string.replace('args.', "")
    tb_writer = SummaryWriter(os.path.join(log_dir, args_string))
    print(f"Training with settings: {args_string}")

    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Preparing {args.dataset_name} dataset")
    train_dataloader, val_dataloader, test_dataloader, num_labels, metric, label_info = (
        prepare_classification_datasets(
            args.dataset_name, tokenizer, step_batch_size, 
            max_length=max_length, seed=args.random_seed
        )
    )
    
    print(f"Loading model configuration from {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    
    config.num_labels = num_labels
    config.pad_token_id = tokenizer.pad_token_id
    config.prompt_len = args.num_virtual_tokens
    config.gen_prompt = args.generator
    
    from transformers.models.llama import LlamaForSequenceClassification
    print("Initializing LlamaForSequenceClassification model")
    model = LlamaForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )
    
    for param in model.parameters():
        param.requires_grad = False
    
    if hasattr(model.model, 'gen_prompt'):
        for prompt_tuning in model.model.gen_prompt:
            for param in prompt_tuning.parameters():
                param.requires_grad = True
    
    model.score.weight.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params:.4%})")
    print(f"Total params: {total_params:,}")
    
    model = model.to(device)
    
    total_steps = len(train_dataloader) // accum_steps * epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        relative_step=False,
        weight_decay=args.weight_decay,
        scale_parameter=False
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    step = 0
    log_loss = []
    
    max_patience = args.max_patience
    patience_score = max_patience
    max_val_score = 0
    max_test_result_score = 0
    
    train_time_total = 0
    start_time = time.time()
    
    save_dir = f"llama_models/{os.path.basename(model_path)}/{args.generator}/{args.dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Beginning training for {epochs} epochs")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        train_loss, step, log_loss, train_time = train_epoch(
            model, optimizer, scheduler, scaler, train_dataloader, 
            accum_steps, tb_writer, step, log_step, log_loss, device, metric
        )
        train_time_total += train_time
        
        model.eval()
        val_loss, preds, answers = evaluate_epoch(model, val_dataloader, device, test=False)
        val_score = score(metric, preds, answers)
        
        print(f"Epoch {epoch+1} | Train loss: {train_loss:.5f} | Validation loss: {val_loss:.5f} | " +
              f"Validation {metric}: {val_score*100:.2f}%")
        
        is_best = val_score >= max_val_score
        
        tb_writer.add_scalar('validation/loss', val_loss, epoch+1)
        tb_writer.add_scalar('validation/performance', val_score, epoch+1)
        
        if is_best:
            print("New best validation score! Saving model and updating patience.")
            max_val_score = val_score
            patience_score = max_patience
            
            _, test_preds, test_answers = evaluate_epoch(model, test_dataloader, device, test=True)
            test_score = score(metric, test_preds, test_answers)
            max_test_result_score = test_score
            
            print(f"Test {metric}: {test_score*100:.2f}%")
            
            if args.save_model:
                model_save_path = os.path.join(save_dir, f"{args_string}_best")
                model.save_pretrained(model_save_path)
                print(f"Model saved to {model_save_path}")
        else:
            patience_score -= 1
            print(f"Validation score did not improve. Patience: {patience_score}/{max_patience}")
            
            if patience_score <= 0:
                print("Patience exhausted. Early stopping.")
                break
    
    total_time = time.time() - start_time
    avg_epoch_time = train_time_total / (epoch + 1)
    
    print(f"Training complete!")
    print(f"Best validation {metric}: {max_val_score*100:.2f}%")
    print(f"Best test {metric}: {max_test_result_score*100:.2f}%")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    
    results_dir = f"results/{os.path.basename(model_path)}/{args.dataset_name}/"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f"{args.generator}.txt"), 'a') as f:
        f.write(f"{args.generator=} {args.dataset_name=} {args_string}\n")
        f.write(f"Training time: {total_time/60:.2f} minutes / {avg_epoch_time:.2f} seconds per epoch\n")
        f.write(f"Best validation {metric}: {max_val_score*100:.2f}%\n")
        f.write(f"Best test {metric}: {max_test_result_score*100:.2f}%\n")
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.09)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--step_batch_size', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_patience', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_model', action='store_true')
    
    parser.add_argument('--dataset_name', type=str, default='copa')
    parser.add_argument('--model_name_or_path', type=str, default='Llama-3.2-1B')
    parser.add_argument('--random_seed', type=int, default=42)
    
    parser.add_argument('--generator', type=str, default='capt')
    parser.add_argument('--num_virtual_tokens', type=int, default=1)
    
    args = parser.parse_args()
    train(args)