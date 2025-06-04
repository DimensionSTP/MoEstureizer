from typing import Tuple, List, Any, Optional, Union

from functools import partial
import os
import json

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn import ModuleList

from transformers import AutoModelForCausalLM

from transformers.cache_utils import DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.generation import GenerationMixin

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    KwargsForCausalLM,
)

from transformers.utils import can_return_tuple, logging


logger = logging.get_logger(__name__)


class LlamaSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.gate = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.num_experts,
            bias=False,
        )
        self.experts = ModuleList([LlamaMLP(config) for _ in range(self.num_experts)])

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(
            -1,
            hidden_dim,
        )

        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(
            router_logits,
            dim=1,
            dtype=torch.float,
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(
                dim=-1,
                keepdim=True,
            )
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = F.one_hot(
            selected_experts,
            num_classes=self.num_experts,
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hidden_states[None, top_x].reshape(
                -1,
                hidden_dim,
            )
            current_hidden_states = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )

            final_hidden_states.index_add_(
                0,
                top_x,
                current_hidden_states.to(hidden_states.dtype),
            )
        final_hidden_states = final_hidden_states.reshape(
            batch_size,
            sequence_length,
            hidden_dim,
        )

        return final_hidden_states, router_logits


class MoEsturizedLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(
        self,
        config: LlamaConfig,
        layer_idx: int,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
        )

        self.mlp = LlamaSparseMoeBlock(config=config)

        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(
                self.config,
                "output_router_logits",
                False,
            )
        )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class MoEsturizedLlamaModel(LlamaModel):
    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__(config=config)

        for i in range(len(self.layers)):
            self.layers[i] = MoEsturizedLlamaDecoderLayer(
                config=config,
                layer_idx=i,
            )

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> MoeModelOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(
                self.config,
                "output_router_logits",
                False,
            )
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k: int = 3,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits],
            dim=0,
        )

    routing_weights = F.softmax(
        concatenated_gate_logits,
        dim=-1,
    )

    _, selected_experts = torch.topk(
        routing_weights,
        top_k,
        dim=-1,
    )

    expert_mask = F.one_hot(
        selected_experts,
        num_experts,
    )

    if attention_mask is None:
        tokens_per_expert = torch.mean(
            expert_mask.float(),
            dim=0,
        )

        router_prob_per_expert = torch.mean(
            routing_weights,
            dim=0,
        )
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (
            batch_size * sequence_length
        )

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand(
                (
                    num_hidden_layers,
                    batch_size,
                    sequence_length,
                    top_k,
                    num_experts,
                )
            )
            .reshape(
                -1,
                top_k,
                num_experts,
            )
            .to(compute_device)
        )

        tokens_per_expert = torch.sum(
            expert_mask.float() * expert_attention_mask,
            dim=0,
        ) / torch.sum(
            expert_attention_mask,
            dim=0,
        )

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand(
                (
                    num_hidden_layers,
                    batch_size,
                    sequence_length,
                    num_experts,
                )
            )
            .reshape(
                -1,
                num_experts,
            )
            .to(compute_device)
        )

        router_prob_per_expert = torch.sum(
            routing_weights * router_per_expert_attention_mask,
            dim=0,
        ) / torch.sum(
            router_per_expert_attention_mask,
            dim=0,
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))

    return overall_loss * num_experts


class MoEsturizedLlamaForCausalLM(LlamaForCausalLM, GenerationMixin):
    def __init__(
        self,
        config: LlamaConfig,
    ) -> None:
        super().__init__(config=config)
        self.post_init()

        self.model = MoEsturizedLlamaModel(config=config)

        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.router_aux_loss_coef = config.router_aux_loss_coef

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> MoeCausalLMOutputWithPast:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else getattr(
                self.config,
                "output_router_logits",
                False,
            )
        )

        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                self.vocab_size,
                **kwargs,
            )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def save_pretrained(
        self,
        save_directory: str,
        **kwargs: Any,
    ) -> None:
        """Save the model weights and configuration to a directory."""
        os.makedirs(
            save_directory,
            exist_ok=True,
        )

        config_dict = self.config.to_dict()
        config_dict.update(
            {
                "num_experts": self.num_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "norm_topk_prob": self.norm_topk_prob,
                "router_aux_loss_coef": self.router_aux_loss_coef,
            }
        )

        config_path = os.path.join(
            save_directory,
            "config.json",
        )
        with open(config_path, "w") as f:
            json.dump(
                config_dict,
                f,
                indent=2,
            )

        super().save_pretrained(
            save_directory,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        initialize: bool = False,
        num_experts: int = 8,
        num_experts_per_tok: int = 3,
        norm_topk_prob: bool = False,
        router_aux_loss_coef: float = 0.001,
        *args: Any,
        **kwargs: Any,
    ) -> "MoEsturizedLlamaForCausalLM":
        if initialize:
            base_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )

            config = base_model.config
            config.num_experts = num_experts
            config.num_experts_per_tok = num_experts_per_tok
            config.norm_topk_prob = norm_topk_prob
            config.router_aux_loss_coef = router_aux_loss_coef

            moe_model = cls(config)

            moe_model.load_state_dict(
                base_model.state_dict(),
                strict=False,
            )

            layers = len(moe_model.model.layers)
            for i, layer in enumerate(moe_model.model.layers):
                if isinstance(layer.mlp, LlamaSparseMoeBlock):
                    original_mlp = base_model.model.layers[i].mlp

                    for expert in layer.mlp.experts:
                        expert.load_state_dict(original_mlp.state_dict())

                    torch.nn.init.xavier_uniform_(
                        layer.mlp.gate.weight,
                        gain=0.02,
                    )

            return moe_model
        else:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
