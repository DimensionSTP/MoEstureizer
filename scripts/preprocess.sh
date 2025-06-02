#!/bin/bash

path="src/preprocessing"
upload_user="HuggingFaceTB"
model_type="SmolLM2-360M-Instruct"

python $path/merge_tokenizer.py upload_user=$upload_user model_type=$model_type
python $path/merge_model.py upload_user=$upload_user model_type=$model_type
