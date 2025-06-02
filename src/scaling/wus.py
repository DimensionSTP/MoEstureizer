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
def width_upscale(
    config: DictConfig,
) -> None:
    scaling_factor = config.wus_hidden_scale
    scaling_method = config.wus_scaling_method
    attention_scaling = config.wus_attention_scaling
    model_name = f"{config.model_type}-WUS-{scaling_method}-{attention_scaling}-{scaling_factor}x"
    repo_id = f"{config.user_name}/{model_name}"

    save_dir = f"{config.connected_dir}/wus/{model_name}"
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
    original_hidden_size = model_config.hidden_size
    original_intermediate_size = model_config.intermediate_size
    original_num_attention_heads = model_config.num_attention_heads
    original_num_key_value_heads = getattr(
        model_config,
        "num_key_value_heads",
        model_config.num_attention_heads,
    )
    original_vocab_size = model_config.vocab_size

    scaled_hidden_size = int(original_hidden_size * scaling_factor)
    scaled_intermediate_size = int(original_intermediate_size * scaling_factor)

    if attention_scaling == "heads":
        scaled_num_attention_heads = int(original_num_attention_heads * scaling_factor)
        scaled_num_key_value_heads = int(original_num_key_value_heads * scaling_factor)
        head_dim = original_hidden_size // original_num_attention_heads
    elif attention_scaling == "dims":
        scaled_num_attention_heads = original_num_attention_heads
        scaled_num_key_value_heads = original_num_key_value_heads
        head_dim = int(
            (original_hidden_size // original_num_attention_heads) * scaling_factor
        )
    else:
        raise ValueError(f"Invalid attention scaling method: {attention_scaling}")

    model_config._name_or_path = repo_id
    model_config.torch_dtype = torch_dtype
    model_config.hidden_size = scaled_hidden_size
    model_config.intermediate_size = scaled_intermediate_size
    model_config.num_attention_heads = scaled_num_attention_heads
    model_config.num_key_value_heads = scaled_num_key_value_heads

    if hasattr(model_config, "head_dim"):
        model_config.head_dim = head_dim

    model_config.save_pretrained(save_dir)

    model = AutoModelForCausalLM.from_pretrained(config.pretrained_model_name)
    state_dict = model.state_dict()
    new_state_dict = {}

    for key, tensor in tqdm(state_dict.items(), desc="Scaling weights"):
        if "embed_tokens" in key or "lm_head" in key:
            if tensor.dim() == 2:
                new_tensor = torch.zeros(
                    original_vocab_size,
                    scaled_hidden_size,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )

                if scaling_method == "interleaving":
                    new_tensor = (
                        tensor.unsqueeze(-1)
                        .expand(
                            -1,
                            -1,
                            int(scaling_factor),
                        )
                        .reshape(
                            tensor.shape[0],
                            -1,
                        )
                    )
                elif scaling_method == "concat":
                    new_tensor = torch.cat(
                        [tensor] * int(scaling_factor),
                        dim=1,
                    )
                else:
                    raise ValueError(f"Invalid scaling method: {scaling_method}")
            else:
                new_tensor = tensor.clone()

        elif (
            "q_proj" in key
            or "k_proj" in key
            or "v_proj" in key
            or "o_proj" in key
            or "gate_proj" in key
            or "up_proj" in key
            or "down_proj" in key
        ):
            old_out_dim, old_in_dim = tensor.size()
            new_out_dim = int(old_out_dim * scaling_factor)
            new_in_dim = int(old_in_dim * scaling_factor)

            new_tensor = torch.zeros(
                new_out_dim,
                new_in_dim,
                dtype=tensor.dtype,
                device=tensor.device,
            )

            if scaling_method == "interleaving":
                new_tensor = (
                    tensor.unsqueeze(-1)
                    .unsqueeze(-1)
                    .expand(
                        -1,
                        -1,
                        int(scaling_factor),
                        int(scaling_factor),
                    )
                    .reshape(
                        new_out_dim,
                        new_in_dim,
                    )
                )
            elif scaling_method == "concat":
                new_tensor = torch.cat(
                    [
                        torch.cat(
                            [tensor] * int(scaling_factor),
                            dim=1,
                        )
                    ]
                    * int(scaling_factor),
                    dim=0,
                )
            else:
                raise ValueError(f"Invalid scaling method: {scaling_method}")

        elif tensor.dim() == 1:
            new_size = int(tensor.size(0) * scaling_factor)
            new_tensor = torch.zeros(
                new_size,
                dtype=tensor.dtype,
                device=tensor.device,
            )

            if scaling_method == "interleaving":
                new_tensor = (
                    tensor.unsqueeze(-1)
                    .expand(
                        -1,
                        int(scaling_factor),
                    )
                    .reshape(-1)
                )
            elif scaling_method == "concat":
                new_tensor = torch.cat([tensor] * int(scaling_factor))
            else:
                raise ValueError(f"Invalid scaling method: {scaling_method}")

        else:
            new_tensor = tensor.clone()

        new_state_dict[key] = new_tensor

    keys = list(new_state_dict.keys())
    num_splits = config.num_safetensors if hasattr(config, "num_safetensors") else 1
    split_size = len(keys) // num_splits
    total_size = sum(
        param.numel() * param.element_size() for param in new_state_dict.values()
    )
    index_dict = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": {},
    }

    for i in tqdm(range(num_splits), desc="Saving model"):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_splits - 1 else len(keys)
        safe_tensors_name = f"model-{i+1:05d}-of-{num_splits:05d}.safetensors"
        part_state_dict = {
            k: new_state_dict[k].to(safetensors_dtype) for k in keys[start_idx:end_idx]
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
    width_upscale()
