#!/bin/bash

data_type="conversational"

dataset_names=(
    "aime24"
    "arc_challenge"
    "arc_easy"
    "gpqa_diamond"
    "gpqa_extended"
    "gpqa_main"
    "gsm8k"
    "humaneval"
    "math500"
    "mbpp"
    "mmlu"
    "mmlu_pro"
    "open_rewrite_eval"
)

is_preprocessed=False
moe_type="moesturized"
initialize=False
base_path="/data/MoEstureizer"

model_details=(
    "SmolLM2-135M-Instruct-${moe_type}-experts_8-tok_3"
    "SmolLM2-135M-Instruct-DUS-36layers-${moe_type}-experts_8-tok_3"
    "SmolLM2-360M-Instruct-${moe_type}-experts_8-tok_3"
    "SmolLM2-360M-Instruct-DUS-40layers-${moe_type}-experts_8-tok_3"
    "SmolLM2-1.7B-Instruct-${moe_type}-experts_8-tok_3"
    "SmolLM2-1.7B-Instruct-DUS-36layers-${moe_type}-experts_8-tok_3"
)

left_padding=True
max_length=2048
max_new_tokens=256
eval_batch_size=16
workers_ratio=8
use_all_workers=False
num_gpus=$(nvidia-smi -L | wc -l)

for model_detail in "${model_details[@]}"
do
    pretrained_model_name="${base_path}/${model_detail}"

    for dataset_name in "${dataset_names[@]}"
    do
        echo "Running evaluation on dataset: $dataset_name"

        torchrun --nproc_per_node=$num_gpus main.py mode=test \
            data_type=$data_type \
            dataset_name=$dataset_name \
            is_preprocessed=$is_preprocessed \
            moe_type=$moe_type \
            initialize=$initialize \
            model_detail=$model_detail \
            pretrained_model_name=$pretrained_model_name \
            left_padding=$left_padding \
            max_length=$max_length \
            max_new_tokens=$max_new_tokens \
            eval_batch_size=$eval_batch_size \
            workers_ratio=$workers_ratio \
            use_all_workers=$use_all_workers
    done
done
