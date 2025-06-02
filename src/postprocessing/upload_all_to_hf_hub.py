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

from tqdm import tqdm

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="../../configs/",
    config_name="sft.yaml",
)
def upload_to_hf_hub(
    config: DictConfig,
) -> None:
    base_dir = config.output_dir
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

    for step_dir in tqdm(sorted(os.listdir(base_dir))):
        save_dir = os.path.join(
            base_dir,
            step_dir,
        )
        if os.path.isdir(save_dir):
            revision = step_dir

        api.create_branch(
            repo_id=f"{config.user_name}/{config.model_detail}-{config.upload_tag}",
            branch=revision,
            token=token,
            repo_type="model",
            exist_ok=True,
        )

        api.upload_folder(
            repo_id=f"{config.user_name}/{config.model_detail}-{config.upload_tag}",
            folder_path=save_dir,
            commit_message=f"Upload revision {revision}",
            token=token,
            repo_type="model",
            revision=revision,
        )


if __name__ == "__main__":
    upload_to_hf_hub()
