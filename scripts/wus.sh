#!/bin/bash

path="src/scaling"
upload_user="HuggingFaceTB"
model_type="SmolLM2-360M-Instruct"
precision="bf16"
wus_hidden_scale=2
wus_scaling_method="concat"
wus_attention_scaling="heads"
num_safetensors=1

python $path/wus.py \
    upload_user=$upload_user \
    model_type=$model_type \
    precision=$precision \
    wus_hidden_scale=$wus_hidden_scale \
    wus_scaling_method=$wus_scaling_method \
    wus_attention_scaling=$wus_attention_scaling \
    num_safetensors=$num_safetensors
