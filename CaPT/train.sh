#!/bin/bash

# T5 Example
CUDA_VISIBLE_DEVICES=0 python capt/train_t5.py --lr 1 --generator "capt" --prompt_layers "all" --max_patience 20 --batch_size 32  --epoch 50 --max_length 512  --model_name_or_path t5-base --tokenizer_name_or_path t5-base --warmup_ratio 0.01 --method gen-prompt --dataset_name copa  --num_virtual_tokens 1 --random_seed 88


# Llama Example
CUDA_VISIBLE_DEVICES=0 python capt/train_llama.py --generator capt  --lr 7e-4  --dataset_name cb --random_seed 88 --num_virtual_tokens 1 --epoch 50 --batch_size 32 --warmup_ratio 0.03 --weight_decay 0.005 --model_name_or_path meta-llama/Llama-3.2-1B


# Qwen Example
CUDA_VISIBLE_DEVICES=0 python capt/train_qwen.py --lr 1e-1  --dataset_name cb  --random_seed 88 --num_virtual_tokens 1 --epoch 50 --batch_size 16 --warmup_ratio 0.01 --weight_decay 0.005 --max_patience 20 --generator capt --model_name_or_path Qwen/Qwen2.5-1.5B