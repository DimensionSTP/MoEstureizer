import os

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

import pandas as pd

import torch

from transformers import set_seed

import wandb

from ..utils import SetUp


def train(
    config: DictConfig,
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(
            project=config.project_name,
            name=config.logging_name,
        )

    if "seed" in config:
        set_seed(config.seed)

    if config.devices is not None:
        if isinstance(config.devices, int):
            num_gpus = min(config.devices, torch.cuda.device_count())
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        elif isinstance(config.devices, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = config.devices
        elif isinstance(config.devices, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.devices))

    setup = SetUp(config)

    train_dataset = setup.get_train_dataset()
    val_dataset = setup.get_val_dataset() if config.use_validation else None

    model = setup.get_model()
    data_encoder = setup.get_data_encoder()

    training_arguments = setup.get_training_arguments()

    ds_config = setup.get_ds_config()
    if ds_config:
        training_arguments.deepspeed = ds_config

    trainer_config = OmegaConf.to_container(
        config.trainer,
        resolve=True,
    )
    trainer_config.pop(
        "_target_",
        None,
    )

    TrainerClass = get_class(config.trainer._target_)

    trainer = TrainerClass(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=data_encoder,
        **trainer_config,
    )

    try:
        trainer.train(
            resume_from_checkpoint=(
                config.resume_from_checkpoint if config.resume_training else None
            )
        )
        trainer.save_model()

        if local_rank == 0:
            wandb.run.alert(
                title="Training Complete",
                text="Training process has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if local_rank == 0:
            wandb.run.alert(
                title="Training Error",
                text="An error occurred during training",
                level="ERROR",
            )
        raise e


def test(
    config: DictConfig,
) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        wandb.init(
            project=config.project_name,
            name=config.logging_name,
        )

    if "seed" in config:
        set_seed(config.seed)

    if config.devices is not None:
        if isinstance(config.devices, int):
            num_gpus = min(config.devices, torch.cuda.device_count())
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(num_gpus)))
        elif isinstance(config.devices, str):
            os.environ["CUDA_VISIBLE_DEVICES"] = config.devices
        elif isinstance(config.devices, list):
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.devices))

    setup = SetUp(config)

    test_loader = setup.get_test_loader()

    model = setup.get_model()
    data_encoder = setup.get_data_encoder()

    try:
        results = []
        for batch in test_loader:
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

            generations = data_encoder.batch_decode(
                outputs,
                skip_special_tokens=True,
            )
            labels = batch["labels"]

            for generation, label in zip(generations, labels):
                results.append(
                    {
                        "generation": generation,
                        "label": label,
                    }
                )

        if local_rank == 0:
            df = pd.DataFrame(results)
            df.to_csv(
                config.test_output_path,
                index=False,
            )

            wandb.log({"test_results": wandb.Table(dataframe=df)})

            wandb.run.alert(
                title="Testing Complete",
                text="Testing process has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if local_rank == 0:
            wandb.run.alert(
                title="Testing Error",
                text="An error occurred during testing",
                level="ERROR",
            )
        raise e
