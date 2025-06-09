#!/bin/bash

data_type="conversational"
dataset_name="mmlu"
is_preprocessed=False
upload_user="HuggingFaceTB"
model_type="SmolLM2-360M-Instruct"
moe_type="moesturized"
initialize=True
num_experts=8
num_experts_per_tok=3
left_padding=True
max_new_tokens=512
eval_batch_size=128
workers_ratio=8
use_all_workers=False

python main.py mode=test \
    data_type=$data_type \
    dataset_name=$dataset_name \
    is_preprocessed=$is_preprocessed \
    upload_user=$upload_user \
    model_type=$model_type \
    moe_type=$moe_type \
    initialize=$initialize \
    num_experts=$num_experts \
    num_experts_per_tok=$num_experts_per_tok \
    left_padding=$left_padding \
    max_new_tokens=$max_new_tokens \
    eval_batch_size=$eval_batch_size \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers
