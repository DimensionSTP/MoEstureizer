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

from huggingface_hub import HfApi, HfFolder

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def upload_to_hf_hub(
    config: DictConfig,
) -> None:
    save_dir = f"{config.output_dir}/checkpoint-{config.step}"
    api = HfApi()
    token = HfFolder.get_token()

    repo_id = f"{config.user_name}/{config.model_detail}-{config.upload_tag}"
    api.create_repo(
        repo_id=repo_id,
        token=token,
        private=True,
        repo_type="model",
        exist_ok=True,
    )

    api.upload_folder(
        repo_id=f"{config.user_name}/{config.model_detail}-{config.upload_tag}",
        folder_path=save_dir,
        commit_message=f"Upload checkpoint-{config.step}",
        token=token,
        repo_type="model",
    )


if __name__ == "__main__":
    upload_to_hf_hub()
