import dotenv

dotenv.load_dotenv(
    override=True,
)

import os

from transformers import AutoTokenizer

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def merge_tokenizer(
    config: DictConfig,
) -> None:
    korean_tokenizer = AutoTokenizer.from_pretrained(config.korean_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    def is_korean(token):
        for char in token:
            if (
                "\uac00" <= char <= "\ud7a3"
                or "\u1100" <= char <= "\u11ff"
                or "\u3130" <= char <= "\u318f"
            ):
                return True
        return False

    korean_tokenizer_tokens = korean_tokenizer.get_vocab().keys()
    korean_tokens = [token for token in korean_tokenizer_tokens if is_korean(token)]

    tokenizer_tokens = tokenizer.get_vocab().keys()
    new_tokens = [token for token in korean_tokens if token not in tokenizer_tokens]
    new_tokens_length = len(new_tokens)
    max_multiple_of_128 = (new_tokens_length // 128) * 128
    new_tokens = new_tokens[:max_multiple_of_128]
    tokenizer.add_tokens(new_tokens)

    os.makedirs(
        config.custom_data_encoder_path,
        exist_ok=True,
    )
    tokenizer.save_pretrained(config.custom_data_encoder_path)


if __name__ == "__main__":
    merge_tokenizer()
