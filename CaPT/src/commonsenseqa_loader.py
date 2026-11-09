import os
import json
from collections import defaultdict
from datasets import load_dataset


def construct_dev_from_train_t2t_csqa(train_list, num_labels):
    """Split training data to create validation set for CommonsenseQA"""
    # Use 90% for training, 10% for validation
    border = len(train_list) * 9 // 10
    actual_train_list = train_list[:border]
    val_list = train_list[border:]
    return actual_train_list, val_list


def commonsenseqa(text_to_text):
    """Load and format CommonsenseQA dataset"""
    try:
        dataset = load_dataset("tau/commonsense_qa")
        train = dataset['train']
        val = dataset['validation']
        
        train_data = []
        val_data = []
        
        for item in train:
            train_data.append({
                'id': item['id'],
                'question': item['question'],
                'choices': {
                    'label': item['choices']['label'],
                    'text': item['choices']['text']
                },
                'answerKey': item['answerKey']
            })
            
        for item in val:
            val_data.append({
                'id': item['id'], 
                'question': item['question'],
                'choices': {
                    'label': item['choices']['label'],
                    'text': item['choices']['text']
                },
                'answerKey': item['answerKey']
            })
            
    except Exception as e:
        print(f"Error loading from HuggingFace datasets: {e}")
        print("Trying to load from local files...")
        
        path = "data/commonsenseqa"
        try:
            with open(os.path.join(path, 'train_rand_split.jsonl'), 'r') as f:
                train_data = [json.loads(line) for line in f]
            with open(os.path.join(path, 'dev_rand_split.jsonl'), 'r') as f:
                val_data = [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(
                "CommonsenseQA data not found. Please either:\n"
                "1. Install datasets: pip install datasets\n"
                "2. Or download data to data/commonsenseqa/ directory"
            )

    if text_to_text:
        text_format = "commonsenseqa question: **question** choices: **choices**"
        
        train_list = []
        val_list = []
        
        for d in train_data:
            choices_text = ", ".join([f"{label}: {text}" for label, text in zip(d['choices']['label'], d['choices']['text'])])
            input_text = text_format.replace("**question**", d['question']).replace("**choices**", choices_text)
            
            answer = d['answerKey']
            train_list.append((input_text, answer))
        
        for d in val_data:
            choices_text = ", ".join([f"{label}: {text}" for label, text in zip(d['choices']['label'], d['choices']['text'])])
            input_text = text_format.replace("**question**", d['question']).replace("**choices**", choices_text)
            answer = d['answerKey']
            val_list.append((input_text, answer))
        
        num_labels = 5
        
    else:
        train_list = []
        val_list = []
        
        answer_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        
        for d in train_data:
            choices_text = " ".join([f"({label}) {text}" for label, text in zip(d['choices']['label'], d['choices']['text'])])
            input_text = f"{d['question']} {choices_text}"
            label = answer_to_idx[d['answerKey']]
            train_list.append((input_text, label))
        
        for d in val_data:
            choices_text = " ".join([f"({label}) {text}" for label, text in zip(d['choices']['label'], d['choices']['text'])])
            input_text = f"{d['question']} {choices_text}"
            label = answer_to_idx[d['answerKey']]
            val_list.append((input_text, label))
        
        num_labels = 5

    return train_list, val_list, num_labels


def get_commonsenseqa(split, text_to_text=False):
    train_list, val_list, num_labels = commonsenseqa(text_to_text)
    
    if text_to_text:
        print("csqa t2t")
        # For text-to-text, create our own validation split from training data
        # since we want to use the original validation set as test set
        test_list = val_list
        train_list, val_list = construct_dev_from_train_t2t_csqa(train_list, num_labels)

    if split == "train":
        return train_list, num_labels
    elif split == "dev":
        return val_list, num_labels
    elif split == "test":
        return test_list, num_labels
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'dev', or 'test'")