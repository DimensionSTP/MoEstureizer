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

import sys

filtered_argv = []
for arg in sys.argv:
    if arg.startswith("--local_rank"):
        continue
    if arg.startswith("--node_rank"):
        continue
    if arg.startswith("--world_rank"):
        continue
    if arg.startswith("--master_addr"):
        continue
    if arg.startswith("--master_port"):
        continue
    filtered_argv.append(arg)
sys.argv = filtered_argv

import hydra
from omegaconf import DictConfig

from src.pipelines import *


@hydra.main(
    config_path="configs/",
    config_name="sft.yaml",
)
def main(
    config: DictConfig,
) -> None:
    if config.mode == "train":
        return train(config)
    elif config.mode == "test":
        return test(config)
    else:
        raise ValueError(f"Invalid execution mode: {config.mode}")


if __name__ == "__main__":
    main()
