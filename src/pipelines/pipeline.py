import os

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

import pandas as pd

import torch
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import set_seed

import wandb

from tqdm import tqdm

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
                text=f"Training process on {config.dataset_name} has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if local_rank == 0:
            wandb.run.alert(
                title="Training Error",
                text=f"An error occurred during training on {config.dataset_name}: {e}",
                level="ERROR",
            )
        raise e


def test(
    config: DictConfig,
) -> None:
    world_size = torch.cuda.device_count()
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    if local_rank == 0:
        wandb.init(
            project=config.project_name,
            name=config.model_detail,
        )

    if "seed" in config:
        set_seed(config.seed)

    setup = SetUp(config)

    test_dataset = setup.get_test_dataset()
    sampler = (
        DistributedSampler(
            test_dataset,
            shuffle=False,
        )
        if world_size > 1
        else None
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=setup.num_workers,
        pin_memory=True,
        sampler=sampler,
    )

    model = setup.get_model()
    data_encoder = setup.get_data_encoder()

    model.to(local_rank)
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
        )

    try:
        results = []
        with torch.inference_mode():
            for batch in tqdm(
                test_loader,
                desc=f"Test {config.dataset_name}",
                disable=(local_rank != 0),
            ):
                input_ids = batch["input_ids"].to(local_rank)
                attention_mask = batch["attention_mask"].to(local_rank)

                if world_size > 1:
                    generate_func = model.module.generate
                else:
                    generate_func = model.generate

                outputs = generate_func(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                ).cpu()

                instructions = data_encoder.batch_decode(
                    batch["input_ids"],
                    skip_special_tokens=True,
                )

                generations = data_encoder.batch_decode(
                    outputs[:, batch["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )

                labels = batch["labels"]

                for instruction, generation, label in zip(
                    instructions, generations, labels
                ):
                    results.append(
                        {
                            "instruction": instruction,
                            "generation": generation,
                            "label": label,
                        }
                    )

        if world_size > 1:
            all_results = [None] * world_size
            dist.gather_object(
                results,
                all_results if local_rank == 0 else None,
                dst=0,
            )
            if local_rank == 0:
                results = [item for sublist in all_results for item in sublist]

        if local_rank == 0:
            os.makedirs(
                config.test_output_dir,
                exist_ok=True,
            )
            test_output_path = os.path.join(
                config.test_output_dir,
                f"{config.test_output_name}.json",
            )

            df = pd.DataFrame(results)
            df.to_json(
                test_output_path,
                orient="records",
                indent=2,
                force_ascii=False,
            )

            wandb.log({"test_results": wandb.Table(dataframe=df)})

            wandb.run.alert(
                title="Testing Complete",
                text=f"Testing process on {config.dataset_name} has successfully finished.",
                level="INFO",
            )
    except Exception as e:
        if local_rank == 0:
            wandb.run.alert(
                title="Testing Error",
                text=f"An error occurred during testing on {config.dataset_name}: {e}",
                level="ERROR",
            )
        raise e
    finally:
        if world_size > 1:
            dist.destroy_process_group()
