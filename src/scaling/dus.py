import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["HF_HOME"] = os.environ.get("HF_HOME")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import json

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from safetensors.torch import save_file

from tqdm import tqdm

from huggingface_hub import HfApi, HfFolder

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def depth_upscale(
    config: DictConfig,
) -> None:
    dus_hiddens = config.dus_hidden_layers
    model_name = f"{config.model_type}-DUS-{dus_hiddens}layers"
    repo_id = f"{config.user_name}/{model_name}"

    save_dir = f"{config.connected_dir}/dus/{model_name}"
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    if config.precision == 32 or config.precision == "32":
        safetensors_dtype = torch.float32
        torch_dtype = "float32"
    elif config.precision == 16 or config.precision == "16":
        safetensors_dtype = torch.float16
        torch_dtype = "float16"
    elif config.precision == "bf16":
        safetensors_dtype = torch.bfloat16
        torch_dtype = "bfloat16"
    else:
        raise ValueError(f"Invalid precision type: {config.precision}")

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)
    tokenizer.save_pretrained(save_dir)
    model_config = AutoConfig.from_pretrained(config.pretrained_model_name)
    original_hidden_layers = model_config.num_hidden_layers

    model_config._name_or_path = repo_id
    model_config.torch_dtype = torch_dtype
    model_config.num_hidden_layers = dus_hiddens
    model_config.save_pretrained(save_dir)

    model = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name)

    difference = dus_hiddens - original_hidden_layers
    lower_arrangement = list(range(0, (original_hidden_layers + difference) // 2))
    upper_arrangement = list(
        range((original_hidden_layers - difference) // 2, original_hidden_layers)
    )
    layer_arrangement = lower_arrangement + upper_arrangement

    state_dict = model.state_dict().copy()
    layer_keys_template = [
        key.replace(".0.", ".{}.") for key in model.state_dict() if ".0." in key
    ]

    for dus_layer, original_layer in enumerate(layer_arrangement):
        for key in layer_keys_template:
            state_dict[key.format(dus_layer)] = model.state_dict()[
                key.format(original_layer)
            ]

    keys = list(state_dict.keys())
    num_splits = config.num_safetensors if hasattr(config, "num_safetensors") else 1
    split_size = len(keys) // num_splits
    total_size = sum(
        param.numel() * param.element_size() for param in state_dict.values()
    )
    index_dict = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": {},
    }

    for i in tqdm(range(num_splits)):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_splits - 1 else len(keys)
        safe_tensors_name = f"model-{i+1:05d}-of-{num_splits:05d}.safetensors"
        part_state_dict = {
            k: state_dict[k].to(safetensors_dtype) for k in keys[start_idx:end_idx]
        }
        part_state_dict_mapping = {
            k: safe_tensors_name for k in keys[start_idx:end_idx]
        }
        index_dict["weight_map"].update(part_state_dict_mapping)
        save_file(
            part_state_dict,
            f"{save_dir}/{safe_tensors_name}",
            metadata={
                "format": "pt",
            },
        )
    with open(f"{save_dir}/model.safetensors.index.json", "w") as f:
        json.dump(
            index_dict,
            f,
            indent=2,
        )

    api = HfApi()
    token = HfFolder.get_token()

    api.create_repo(
        repo_id=repo_id,
        token=token,
        private=True,
        repo_type="model",
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=repo_id,
        folder_path=save_dir,
        commit_message=f"Upload {model_name} model",
        token=token,
        repo_type="model",
    )


if __name__ == "__main__":
    depth_upscale()
