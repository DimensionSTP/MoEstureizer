from typing import Tuple

import torch
from torch import nn
from torch.nn import ModuleList

from transformers.configuration_utils import PretrainedConfig


class LMHeadRouter(nn.Module):
    def __init__(
        self,
        configs: Tuple[PretrainedConfig],
    ) -> None:
        super().__init__()
        self.configs = configs
        self.main_config = configs[0]
        self.num_models = len(configs)

        self.hidden_sizes = [config.hidden_size for config in configs]
        self.total_hidden_size = sum(self.hidden_sizes)

        self.router = nn.Parameter(
            torch.tensor(
                [size / (2 * self.total_hidden_size) for size in self.hidden_sizes]
                + [0.5],
                dtype=self.main_config.torch_dtype,
            )
        )

        self.lm_heads = ModuleList(
            [
                nn.Linear(
                    in_features=config.hidden_size,
                    out_features=config.vocab_size,
                    bias=False,
                    dtype=config.torch_dtype,
                )
                for config in configs
            ]
        )

        self.combined_lm_head = nn.Linear(
            in_features=self.total_hidden_size,
            out_features=self.main_config.vocab_size,
            bias=False,
            dtype=self.main_config.torch_dtype,
        )

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        combined_hidden: torch.Tensor,
        slice_indices=None,
    ) -> torch.Tensor:
        if slice_indices is not None:
            logits = [
                self.lm_heads[i](hidden_states[i][:, slice_indices, :])
                for i in range(self.num_models)
            ]
            combined_logits = self.combined_lm_head(
                combined_hidden[:, slice_indices, :]
            )
        else:
            logits = [
                self.lm_heads[i](hidden_states[i]) for i in range(self.num_models)
            ]
            combined_logits = self.combined_lm_head(combined_hidden)

        all_logits = logits + [combined_logits]

        router_weights = torch.softmax(
            self.router,
            dim=0,
        )

        weighted_logits = sum(
            weight * logit
            for weight, logit in zip(
                router_weights,
                all_logits,
            )
        )
        return weighted_logits


class LMHeadGatedRouter(nn.Module):
    def __init__(
        self,
        configs: Tuple[PretrainedConfig],
    ) -> None:
        super().__init__()
        self.configs = configs
        self.main_config = configs[0]
        self.num_models = len(configs)

        self.hidden_sizes = [config.hidden_size for config in configs]
        self.total_hidden_size = sum(self.hidden_sizes)

        self.gate = nn.Linear(
            in_features=self.total_hidden_size,
            out_features=self.num_models + 1,
            bias=False,
            dtype=self.main_config.torch_dtype,
        )

        self.lm_heads = ModuleList(
            [
                nn.Linear(
                    in_features=config.hidden_size,
                    out_features=config.vocab_size,
                    bias=False,
                    dtype=config.torch_dtype,
                )
                for config in configs
            ]
        )

        self.combined_lm_head = nn.Linear(
            in_features=self.total_hidden_size,
            out_features=self.main_config.vocab_size,
            bias=False,
            dtype=self.main_config.torch_dtype,
        )

    def forward(
        self,
        hidden_states: Tuple[torch.Tensor],
        combined_hidden: torch.Tensor,
        slice_indices=None,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if slice_indices is not None:
            logits = [
                self.lm_heads[i](hidden_states[i][:, slice_indices, :])
                for i in range(self.num_models)
            ]
            combined_logits = self.combined_lm_head(
                combined_hidden[:, slice_indices, :]
            )
            gate_input = combined_hidden[:, slice_indices, :]
        else:
            logits = [
                self.lm_heads[i](hidden_states[i]) for i in range(self.num_models)
            ]
            combined_logits = self.combined_lm_head(combined_hidden)
            gate_input = combined_hidden

        all_logits = logits + [combined_logits]

        gate_logits = self.gate(gate_input)
        gate_probs = torch.softmax(
            gate_logits,
            dim=-1,
        )

        weighted_logits = sum(
            gate_probs[..., i : i + 1] * logit for i, logit in enumerate(all_logits)
        )

        aux_loss = self._compute_load_balancing_loss(
            gate_logits=gate_logits,
            gate_probs=gate_probs,
        )

        if output_router_logits:
            return weighted_logits, aux_loss, gate_logits
        else:
            return weighted_logits, aux_loss

    def _compute_load_balancing_loss(
        self,
        gate_logits: torch.Tensor,
        gate_probs: torch.Tensor,
    ) -> torch.Tensor:
        num_experts = gate_logits.size(-1)

        top_expert_indices = torch.argmax(
            gate_probs,
            dim=-1,
        )
        expert_freq = torch.zeros(
            num_experts,
            device=gate_probs.device,
            dtype=gate_probs.dtype,
        )
        for i in range(num_experts):
            expert_freq[i] = (top_expert_indices == i).float().mean()

        expert_avg_prob = gate_probs.mean(dim=(0, 1))

        load_balancing_loss = num_experts * torch.sum(expert_freq * expert_avg_prob)

        return load_balancing_loss
