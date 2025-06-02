from typing import Union

import os

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from ..models import *


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.data_type = self.config.data_type
        self.num_cpus = os.cpu_count()
        self.num_fit_workers = min(
            self.num_cpus,
            (config.devices * config.workers_ratio),
        )
        self.num_workers = (
            self.num_cpus if config.use_all_workers else self.num_fit_workers
        )

    def get_train_dataset(self) -> Dataset:
        train_dataset: Dataset = instantiate(
            self.config.dataset[self.data_type],
            split=self.config.split.train,
        )
        return train_dataset

    def get_val_dataset(self) -> Dataset:
        val_dataset: Dataset = instantiate(
            self.config.dataset[self.data_type],
            split=self.config.split.val,
        )
        return val_dataset

    def get_model(
        self,
    ) -> Union[
        MixLlamaForCausalLM,
        DeepMixLlamaForCausalLM,
        GatedMixLlamaForCausalLM,
        PreTrainedModel,
    ]:
        device_map = None
        quantization_config = None
        if self.config.is_quantized:
            device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}
            quantization_config = BitsAndBytesConfig(**self.config.quantization_config)

        if self.config.mix_type == "mix":
            model = MixLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.model_path,
                model_names=self.config.model_names,
                initialize=self.config.initialize,
                quantization_config=quantization_config,
            )

            if hasattr(self.config, "is_frozen_train") and self.config.is_frozen_train:
                model.freeze_branch_models()

        elif self.config.mix_type == "gated_mix":
            model = GatedMixLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.model_path,
                model_names=self.config.model_names,
                initialize=self.config.initialize,
                router_aux_loss_coef=self.config.router_aux_loss_coef,
                quantization_config=quantization_config,
            )

            if hasattr(self.config, "is_frozen_train") and self.config.is_frozen_train:
                model.freeze_branch_models()

        elif self.config.mix_type == "deep_mix":
            model = DeepMixLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.model_path,
                model_names=self.config.model_names,
                initialize=self.config.initialize,
                mix_interval=self.config.mix_interval,
                quantization_config=quantization_config,
            )

            if hasattr(self.config, "is_frozen_train") and self.config.is_frozen_train:
                if hasattr(model, "freeze_branch_models"):
                    model.freeze_branch_models()

        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name,
                device_map=device_map,
                quantization_config=quantization_config,
            )

        if self.config.is_quantized and self.config.quantization_config.get(
            "load_in_4bit",
            False,
        ):
            model = prepare_model_for_kbit_training(model)

        if self.config.is_peft:
            peft_config = LoraConfig(**self.config.peft_config)
            model = get_peft_model(model, peft_config)

        if self.config.gradient_checkpointing and self.config.mix_type != "deep_mix":
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs,
            )

        return model

    def get_data_encoder(self) -> PreTrainedTokenizer:
        data_encoder = AutoTokenizer.from_pretrained(
            self.config.pretrained_model_name,
            use_fast=True,
        )

        if data_encoder.chat_template is None:
            reference_data_encoder = AutoTokenizer.from_pretrained(
                self.config.reference_data_encoder_name
            )
            data_encoder.chat_template = reference_data_encoder.chat_template

        if data_encoder.pad_token_id is None:
            data_encoder.pad_token_id = data_encoder.eos_token_id
        if self.config.left_padding:
            data_encoder.padding_side = "left"
        else:
            data_encoder.padding_side = "right"

        return data_encoder

    def get_training_arguments(self) -> TrainingArguments:
        training_arguments: TrainingArguments = instantiate(
            self.config.training_arguments,
            dataloader_num_workers=self.num_workers,
        )
        return training_arguments

    def get_ds_config(self) -> DictConfig:
        if self.config.strategy == "deepspeed":
            ds_config = OmegaConf.to_container(
                self.config.deepspeed,
                resolve=True,
            )
            return ds_config
        return None
