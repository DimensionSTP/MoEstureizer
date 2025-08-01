#!/bin/bash

data_type="conversational"
dataset_name="mmlu"
is_preprocessed=False
moe_type="moesturized"
initialize=False
base_path="/data/MoEstureizer"
model_detail="SmolLM2-360M-Instruct-${moe_type}-experts_8-tok_3"
pretrained_model_name="${base_path}/${model_detail}"
revision="main"
left_padding=True
max_length=2048
max_new_tokens=256
do_sample=True
temperature=0.6
top_p=0.95
top_k=20
eval_batch_size=16
workers_ratio=8
use_all_workers=False
num_gpus=$(nvidia-smi -L | wc -l)

torchrun --nproc_per_node=$num_gpus main.py mode=test \
    data_type=$data_type \
    dataset_name=$dataset_name \
    is_preprocessed=$is_preprocessed \
    moe_type=$moe_type \
    initialize=$initialize \
    model_detail=$model_detail \
    pretrained_model_name=$pretrained_model_name \
    revision=$revision \
    left_padding=$left_padding \
    max_length=$max_length \
    max_new_tokens=$max_new_tokens \
    do_sample=$do_sample \
    generation_config.temperature=$temperature \
    generation_config.top_p=$top_p \
    generation_config.top_k=$top_k \
    eval_batch_size=$eval_batch_size \
    workers_ratio=$workers_ratio \
    use_all_workers=$use_all_workers
