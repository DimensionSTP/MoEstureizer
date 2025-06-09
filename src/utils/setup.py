from typing import Union

import os

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import torch
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

        if config.precision in [32, "32"]:
            self.torch_dtype = torch.float32
        elif config.precision in [16, "16"]:
            self.torch_dtype = torch.float16
        elif config.precision == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = "auto"

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

    def get_test_dataset(self) -> Dataset:
        test_dataset: Dataset = instantiate(
            self.config.test_dataset[self.data_type],
        )
        return test_dataset

    def get_model(self) -> Union[
        MoEsturizedLlamaForCausalLM,
        PreTrainedModel,
    ]:
        quantization_config = None
        device_map = None
        if self.config.is_quantized:
            device_map = {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}
            quantization_config = BitsAndBytesConfig(**self.config.quantization_config)

        if self.config.moe_type == "moesturized":
            model = MoEsturizedLlamaForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name,
                initialize=self.config.initialize,
                num_experts=self.config.num_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
                norm_topk_prob=self.config.norm_topk_prob,
                router_aux_loss_coef=self.config.router_aux_loss_coef,
                output_hidden_states=False,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.config.attn_implementation,
                quantization_config=quantization_config,
                device_map=device_map,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=self.config.pretrained_model_name,
                output_hidden_states=False,
                torch_dtype=self.torch_dtype,
                attn_implementation=self.config.attn_implementation,
                quantization_config=quantization_config,
                device_map=device_map,
            )

        if self.config.is_quantized and self.config.quantization_config.get(
            "load_in_4bit",
            False,
        ):
            model = prepare_model_for_kbit_training(model)

        if self.config.is_peft:
            if self.config.moe_type == "moesturized" and self.config.selective_peft:
                model = self._apply_selective_peft(model)
            else:
                peft_config = LoraConfig(**self.config.peft_config)
                model = get_peft_model(model, peft_config)

        return model

    def _apply_selective_peft(
        self,
        model: MoEsturizedLlamaForCausalLM,
    ) -> MoEsturizedLlamaForCausalLM:
        router_gate_params = []
        for name, param in model.named_parameters():
            if (
                "gate" in name
                and "gate_proj" not in name
                and "gate_up_proj" not in name
                and "mlp" not in name
            ):
                router_gate_params.append(name)
                param.requires_grad = True

        peft_config = LoraConfig(**self.config.peft_config)

        if peft_config.target_modules == "all-linear":
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            target_modules = []
            for module in peft_config.target_modules:
                if not (module in ["gate"] or "router" in module.lower()):
                    target_modules.append(module)

        peft_config.target_modules = target_modules

        model = get_peft_model(model, peft_config)

        for name, param in model.named_parameters():
            if (
                "gate" in name
                and "gate_proj" not in name
                and "gate_up_proj" not in name
                and "mlp" not in name
            ):
                param.requires_grad = True

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
