#!/bin/bash

path="src/scaling"
upload_user="HuggingFaceTB"
model_type="SmolLM2-360M-Instruct"
precision="bf16"
dus_hidden_layers=36
num_safetensors=1

python $path/dus.py \
    upload_user=$upload_user \
    model_type=$model_type \
    precision=$precision \
    dus_hidden_layers=$dus_hidden_layers \
    num_safetensors=$num_safetensors
