#!/bin/bash

path="src/preprocessing"
upload_user="HuggingFaceTB"
model_type="SmolLM2-360M-Instruct"

python $path/specialize_reasoning.py upload_user=$upload_user model_type=$model_type
