from typing import Tuple

import torch
from torch import nn
from torch.nn import ModuleList

from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class MixtureOfFeedForward(nn.Module):
    def __init__(
        self,
        configs: Tuple[PretrainedConfig],
    ) -> None:
        super().__init__()
        self.configs = configs
        self.main_config = configs[0]
        self.hidden_size = sum(config.hidden_size for config in configs)
        self.intermediate_size = sum(config.intermediate_size for config in configs)

        self.gate_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=self.main_config.mlp_bias,
            dtype=self.main_config.torch_dtype,
        )
        self.up_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=self.main_config.mlp_bias,
            dtype=self.main_config.torch_dtype,
        )
        self.down_proj = nn.Linear(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            bias=self.main_config.mlp_bias,
            dtype=self.main_config.torch_dtype,
        )

        self.act_fn = ACT2FN[self.main_config.hidden_act]

        self.pre_moff_layernorms = ModuleList(
            LlamaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            for config in configs
        )
        self.post_moff_layernorm = LlamaRMSNorm(
            self.hidden_size,
            eps=self.main_config.rms_norm_eps,
        )

    def forward(
        self,
        xs: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        residual = torch.cat(
            xs,
            dim=-1,
        )

        xs = tuple(self.pre_moff_layernorms[i](xs[i]) for i in range(len(xs)))
        x = torch.cat(
            xs,
            dim=-1,
        )

        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        down_proj = down_proj + residual
        down_proj = self.post_moff_layernorm(down_proj)
        return down_proj


class MixtureOfInterFeedForward(nn.Module):
    def __init__(
        self,
        configs: Tuple[PretrainedConfig],
    ) -> None:
        super().__init__()
        self.configs = configs
        self.main_config = configs[0]
        self.hidden_size = sum(config.hidden_size for config in configs)
        self.intermediate_size = sum(config.intermediate_size for config in configs)
        self.gate_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=self.main_config.mlp_bias,
            dtype=self.main_config.torch_dtype,
        )
        self.up_proj = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            bias=self.main_config.mlp_bias,
            dtype=self.main_config.torch_dtype,
        )
        self.down_projs = ModuleList(
            nn.Linear(
                in_features=self.intermediate_size,
                out_features=config.hidden_size,
                bias=config.mlp_bias,
                dtype=config.torch_dtype,
            )
            for config in configs
        )
        self.act_fn = ACT2FN[self.main_config.hidden_act]

        self.pre_moff_layernorms = ModuleList(
            LlamaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            for config in configs
        )
        self.post_moff_layernorms = ModuleList(
            LlamaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            for config in configs
        )

    def forward(
        self,
        xs: Tuple[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        residuals = xs

        xs = tuple(self.pre_moff_layernorms[i](xs[i]) for i in range(len(xs)))
        x = torch.cat(
            xs,
            dim=-1,
        )

        intermediate = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        down_projs = tuple(down_proj(intermediate) for down_proj in self.down_projs)
        down_projs = tuple(
            down_proj + residual for down_proj, residual in zip(down_projs, residuals)
        )
        down_projs = tuple(
            self.post_moff_layernorms[i](down_proj)
            for i, down_proj in enumerate(down_projs)
        )
        return down_projs
