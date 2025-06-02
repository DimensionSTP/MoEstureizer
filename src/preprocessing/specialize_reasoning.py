import dotenv

dotenv.load_dotenv(
    override=True,
)

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def specialize_reasoning(
    config: DictConfig,
) -> None:
    reasoning_tokenizer = AutoTokenizer.from_pretrained(config.reasoning_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    reasoning_chat_template = reasoning_tokenizer.chat_template

    for reasoning_token in config.reasoning_tokens:
        tokenizer.add_tokens(reasoning_token)

    special_tokens_map = tokenizer.special_tokens_map
    assistant_start = special_tokens_map.get(
        "additional_special_tokens", ["<|im_start|>assistant\n"]
    )[0]
    assistant_end = special_tokens_map.get("eos_token", "<|im_end|>")
    user_start = special_tokens_map.get(
        "additional_special_tokens", ["<|im_start|>user\n"]
    )[0]
    system_start = special_tokens_map.get(
        "additional_special_tokens", ["<|im_start|>system\n"]
    )[0]

    modified_chat_template = (
        reasoning_chat_template.replace(
            "<|im_start|>assistant",
            assistant_start,
        )
        .replace(
            "<|im_end|>",
            assistant_end,
        )
        .replace(
            "<|im_start|>user",
            user_start,
        )
        .replace(
            "<|im_start|>system",
            system_start,
        )
    )

    tokenizer.chat_template = modified_chat_template

    tokenizer.save_pretrained(config.custom_data_encoder_path)


if __name__ == "__main__":
    specialize_reasoning()
