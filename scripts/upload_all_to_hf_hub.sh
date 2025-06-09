#!/bin/bash

path="src/postprocessing"
dataset_name="tulu"
strategy="deepspeed"
model_type="SmolLM2-360M-Instruct"
moe_type="moesturized"
num_experts=8
num_experts_per_tok=3
model_detail="${model_type}-${moe_type}-experts_${num_experts}-tok_${num_experts_per_tok}"
upload_tag="MoEstureized"
is_sft=True
is_quantized=False
is_peft=False
max_length=4096
batch_size=16
gradient_accumulation_steps=1

python $path/upload_all_to_hf_hub.py \
    dataset_name=$dataset_name \
    strategy=$strategy \
    moe_type=$moe_type \
    num_experts=$num_experts \
    num_experts_per_tok=$num_experts_per_tok \
    model_detail=$model_detail \
    upload_tag=$upload_tag \
    is_sft=$is_sft \
    is_quantized=$is_quantized \
    is_peft=$is_peft \
    max_length=$max_length \
    batch_size=$batch_size \
    gradient_accumulation_steps=$gradient_accumulation_steps
