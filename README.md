# LLM model scaling pipeline

## For (s)LLM model scaling

### Dataset
Tulu sampling dataset(preprocessed as instruction, input, and response)

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/merge-up-slm.git
cd merge-up-slm

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10 -y
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### .env file setting
```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
DEVICES={DEVICES}
HF_HOME={HF_HOME}
USER_NAME={USER_NAME}
```

### Train

* end-to-end
```shell
python main.py mode=train
```

### Examples of shell scipts

* full preprocessing
```shell
bash scripts/preprocess.sh
```

* dataset preprocessing
```shell
bash scripts/preprocess_dataset.sh
```

* train
```shell
bash scripts/train.sh
```

### Additional Options

* SFT train(masking input)
```shell
is_sft={True or False}
```

* Use preprocessed tokenizer option
```shell
is_preprocessed={True or False}
```

* Left padding option
```shell
left_padding={True or False}
```

* Pure decoder based LLM QLoRA 4-bit quantization option
```shell
is_quantized={True or False}
```

* Pure decoder based LLM LoRA or QLoRA PEFT option
```shell
is_peft={True or False}
```

* For LLM full fine-tuning(Continued Pretraining) in multi-GPU, recommended
```shell
strategy=deepspeed
```

* Upload user name and model name at HuggingFace Model card
```shell
upload_user={upload_user} 
model_type={model_type}
```

* Set data and target max length for model training and generation
```shell
max_length={max_length} 
```


__If you want to change main config, use --config-name={config_name}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__