# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Callable, Optional, Union

import torch
from torch import nn
from dataclasses import dataclass

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import check_model_inputs
from .configuration_cllama import CLlamaConfig

from transformers import AutoModel, AutoModelForCausalLM, PreTrainedTokenizer
from ..utils import *


logger = logging.get_logger(__name__)

@dataclass
class BaseModelOutputsWithGistStates(BaseModelOutputWithPast):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    gist_hidden: Optional[tuple[tuple[torch.FloatTensor, ...]]] = None # [layers, batch, num_gist, hidden]


@dataclass 
class CausalLMOutputWithGistStates(CausalLMOutputWithPast): 
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    gist_hidden: Optional[tuple[tuple[torch.FloatTensor, ...]]] = None
    critic_outputs: Optional[tuple[torch.FloatTensor, torch.FloatTensor]] = None # (positive_states, negative_states)



@use_kernel_forward_from_hub("RMSNorm")
class CLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class CLlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: CLlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, gist_token_position_ids=[]):

        # modify position ids based on gist_token_position_ids: 
        if len(gist_token_position_ids) > 0:
            pass
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CLlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class CLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: CLlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CLlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: CLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CLlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = CLlamaMLP(config)
        self.input_layernorm = CLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class CLlamaPreTrainedModel(PreTrainedModel):
    config: CLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": CLlamaDecoderLayer,
        "attentions": CLlamaAttention,
    }


@auto_docstring
class CLlamaModel(CLlamaPreTrainedModel):
    def __init__(self, config: CLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [CLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CLlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = CLlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.gist_token_id = None
        self.compress_mode = False
        self.gist_masking = False

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )


        if position_ids is None:
            # PREV CODE
            # position_ids = cache_position.unsqueeze(0)
            # NEW CODE
            position_ids = torch.arange(attention_mask.shape[1], device=attention_mask.device)[None, :] \
                * torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
            num_zeros = attention_mask.shape[-1] - (attention_mask).sum(dim=-1, keepdim=True)
            position_ids -= num_zeros
            position_ids = torch.where(position_ids < 0, 1, position_ids).int()


        # Compress mode will assume the model has expanded vocab.
        # whether gist mask is turned on or not is determined by the self.gist_masking
        if self.compress_mode:

            batch_size, seq_len = attention_mask.shape
            num_new_toks = input_ids.shape[1]

            assert self.gist_token_id is not None, "In compression mode, but GIST token ID is None..."
            if not use_cache: 
                pass 
                causal_mask = get_causal_mask(attention_mask, self.config.num_attention_heads, seq_len)

                # Turn on gist masking
                if self.gist_masking:
                    first_idx = get_first_idx_of_token(input_ids, self.gist_token_id)
                    last_idx = get_last_idx_of_token(input_ids, self.gist_token_id)

                    # print("LAST_IDX_SHAPE", last_idx.shape)
                    # print("ATTENTION_MASK_SHAPE", attention_mask.shape)
                    # print("POSITION_IDS_SHAPE", position_ids.shape)

                    for b in range(batch_size): 
                        # If there exists gist tokens in the input, then fix mask
                        if (input_ids[b] == self.gist_token_id).sum() > 0:
                            if last_idx[b] + 1 < seq_len:
                                causal_mask[b, :, last_idx[b]+1:, :first_idx[b]] = False
                                
                                position_ids[b, last_idx[b]:] -= last_idx[b] - (attention_mask[b]==0).sum()
                    position_ids = torch.where(input_ids == self.gist_token_id, 0, position_ids)
                causal_mask
            else: 
                num_new_toks = input_ids.shape[1]
                batch_size, seq_len = attention_mask.shape
                causal_mask = get_causal_mask_with_cache(past_key_values, attention_mask, self.config.num_attention_heads, num_new_toks, seq_len)

                # turn on gist masking
                if self.gist_masking:
                    cache_len = past_key_values.get_seq_length() 
                    if cache_len == 0: # First input
                        # These attributes are set for each generation batch. 
                        self.first_idx = get_first_idx_of_token(input_ids, self.gist_token_id)
                        self.last_idx = get_last_idx_of_token(input_ids, self.gist_token_id)
                        self.num_gist_tok_per_batch = (input_ids == self.gist_token_id).sum(dim=-1) #[batch,]
                        self.relative_offset = attention_mask.shape[1] - attention_mask.sum(dim=-1)
                    else: 
                        num_new_gist_toks = (input_ids == self.gist_token_id).sum(dim=-1) 
                        # first and last gist positions in new input
                        maybe_last_gist = get_last_idx_of_token(input_ids, self.gist_token_id)                    
                        maybe_first_gist = get_first_idx_of_token(input_ids, self.gist_token_id)                    
                        # update where the first and last gist tokens are in the full sequence
                        self.last_idx = torch.where(num_new_gist_toks > 0, cache_len + maybe_last_gist, self.last_idx)
                        self.first_idx = torch.where(self.num_gist_tok_per_batch == 0, cache_len + maybe_first_gist, self.first_idx) 

                        self.num_gist_tok_per_batch += num_new_gist_toks 


                    for b in range(batch_size): 
                        # If there exists gist tokens in the input, then fix mask
                        if self.num_gist_tok_per_batch[b] > 0:
                            if self.last_idx[b] + 1 < seq_len:
                                causal_mask[b, :, self.last_idx[b]+1-cache_len:, :self.first_idx[b]] = False
                                # mess with position IDS
                                position_ids[b, self.last_idx[b]+1-cache_len:] = position_ids[b, self.last_idx[b]+1-cache_len:] - self.last_idx[b] + self.relative_offset[b]
                    position_ids = torch.where(input_ids == self.gist_token_id, 0, position_ids)
                    
                

            
            causal_mask
                
        else: 
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            batch_size, seq_len = attention_mask.shape
            num_new_toks = input_ids.shape[1]

            # Implementation of 4D causal attention mask 
            # CURRENTLY DOES NOT WORK WITH BEAM SEARCH
            # [batch, heads, new_inputs, total_seq_len] 
            if not use_cache: 
                causal_mask = get_causal_mask(attention_mask, self.config.num_attention_heads, seq_len)
            else: 
                causal_mask = get_causal_mask_with_cache(past_key_values, attention_mask, self.config.num_attention_heads, num_new_toks, seq_len)

            # causal_mask = None # remove later

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output_hidden_states = [hidden_states] # [layer + 1, batch, seq_len, hidden]

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            output_hidden_states.append(hidden_states) 

        hidden_states = self.norm(hidden_states)

        # extract gist states
        if self.compress_mode:
            gist_states = [[] for _ in self.layers] #[layer, [batch, [num_gist, hidden]]] 

            first_gist = get_first_idx_of_token(input_ids, self.gist_token_id) # [batch]
            last_gist = get_last_idx_of_token(input_ids, self.gist_token_id) #[batch]
            
            # loop through batch
            for i in range(len(first_gist)): 
                has_gist = (input_ids[i] == self.gist_token_id).sum()
                if has_gist: 
                    gist_states_slice = [layer_state[i, first_gist[i]:last_gist[i]+1] for layer_state in output_hidden_states] #[layer, [num_gist, hidden]] for batch i
                else: 
                    # if there are no gist tokens, then for that batch, all layers get None
                    gist_states_slice = [None for _ in gist_states]
                for j in range(len(gist_states)): 
                    gist_states[j].append(gist_states_slice[j])


        # TODO: Do this
        return BaseModelOutputsWithGistStates(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=output_hidden_states, 
            gist_hidden=gist_states
        )

    def set_compress_mode(self, mode: bool) -> bool: 
        self.compress_mode = mode
        return self.compress_mode
    
    def set_gist_token_id(self, gist_token_id: int): 
        self.gist_token_id = gist_token_id

    def set_gist_masking(self, gist_masking: bool): 
        self.gist_masking = gist_masking


@auto_docstring
class CLlamaForCausalLM(CLlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = CLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self.compress_mode = False # allows support for gist token, i.e., <GIST> is part of vocab
        self.gist_token_id = None  # gist token id
        self.gist_masking = False  # turns on or off attention masking for gist. if compress_mode is false, this does nothing.

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithGistStates(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            gist_hidden=outputs.gist_hidden
        )
    def enable_compression_mode(self, tokenizer: PreTrainedTokenizer, 
                                gist_masking: Optional[bool]=True) -> PreTrainedTokenizer: 
        """
        1. expand the vocab size of model and tokenizer 
        2. initialize embeddings 
        3. enable flags for compression 
        """
        tokenizer.add_special_tokens({"additional_special_tokens":["<GIST>"]})
        self.resize_token_embeddings(len(tokenizer))

        self.gist_token_id = tokenizer.convert_tokens_to_ids("<GIST>")
        self.model.set_gist_token_id(self.gist_token_id)    

        self.gist_masking = gist_masking
        self.model.set_gist_masking(gist_masking) 

        # initialize word embeddings by averaging the embeddings
        with torch.no_grad():
            self.model.embed_tokens.weight[self.gist_token_id] = self.model.embed_tokens.weight[:self.gist_token_id].mean(dim=0)
            self.lm_head.weight[self.gist_token_id] = self.lm_head.weight[:self.gist_token_id].mean(dim=0)

        self.compress_mode=True
        self.model.set_compress_mode(True)



        return tokenizer
    
    def unlock_embeddings(self): 
        self.model.embed_tokens.weight.requires_grad = True

    def get_gist_token_id(self): 
        return self.gist_token_id




@auto_docstring
class MICLlamaForCausalLM(CLlamaForCausalLM): 
    _tied_weights_keys = ["lm_head.weight"]#, "critic.0.weight", "critic.0.bias", "critic.1.weight", "critic.1.bias"]
    _tp_plan = {"lm_head": "colwise_rep"}#, 
                # "critic": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    mi_model = True, 


    def __init__(self, config): 
        super().__init__(config) 
        #configure linear classifiers
        self.config.num_hidden_layers
        self.critic = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # new args: 
        paraphrase_input_ids=None,  
        paraphrase_attention_mask=None, 
        **kwargs: Unpack[TransformersKwargs]) -> Any: 

        causal_lm_outputs: CausalLMOutputWithGistStates = super().forward(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_values=past_key_values, 
            inputs_embeds=inputs_embeds, 
            labels=labels, 
            use_cache=use_cache, 
            cache_position=cache_position, 
            logits_to_keep=logits_to_keep, 
            **kwargs)
        
        # default stream
        # if paraphrase_input_ids is None: 
        return causal_lm_outputs 

        assert not use_cache, f"if paraphrase is provided, use_cache must be turned off"
        assert self.compress_mode, f"for MI compression, the model must be set to compress mode (i.e., it must support expanded vocab)"
        assert self.critic is not None, f"you must add a critic network"
        # otherwise, do replacement 
        del causal_lm_outputs.hidden_states

        # here, we compute critic outputs using input and paraphrase tokens.
        neg_critic_outputs = self.get_neg_critic_outputs(paraphrase_input_ids=paraphrase_input_ids, 
                                               paraphrase_attention_mask=paraphrase_attention_mask, 
                                               input_gist_states=causal_lm_outputs.gist_hidden)

        pos_critic_outputs = self.get_pos_critic_outputs(input_gist_states=causal_lm_outputs.gist_hidden)



        return CausalLMOutputWithGistStates(
            loss=causal_lm_outputs.loss, 
            logits=causal_lm_outputs.logits, 
            past_key_values=causal_lm_outputs.past_key_values, 
            hidden_states=None, 
            gist_hidden=causal_lm_outputs.gist_hidden, 
            critic_outputs=(pos_critic_outputs, neg_critic_outputs), 
        )


    def get_pos_critic_outputs(
        self, 
        input_gist_states: Optional[List[List[torch.FloatTensor]]] = None, 
        input_hidden: Optional[Tuple[torch.FloatTensor]] = None, # [layer+1, batch, seq_len, hidden]
        input_attention_mask: Optional[Tuple[torch.FloatTensor]] = None, 
        first_gist_index: Optional[torch.Tensor] = None, 
        last_gist_index: Optional[torch.Tensor] = None, 
        return_gist_states: Optional[bool] = False
    ) -> Any: 
        """
        
        Note: input_gist_states has shape: [layer+1, batch, num_gist, hidden]. 
        we assume that in a batch, all seqs have the same number of gist tokens for training.
        """
        assert self.critic is not None, f"Please add a critic network"
        assert not (input_gist_states is None and input_hidden is None), f"one of input_gist_states or input_hidden must be defined."
        
        # if input_gist_states is None: 
        assert first_gist_index is not None and last_gist_index is not None,  f"If we need to extract gist states from input, then we need to have bounds"
        assert input_hidden is not None and input_attention_mask is not None
        # first, make forward pass
        batch_size = first_gist_index.shape[0]

        position_ids = torch.arange(
            input_attention_mask.shape[1], device=input_attention_mask.device)[None, :] \
            * torch.ones((input_attention_mask.shape[0], 1), device=input_attention_mask.device)
        num_zeros = input_attention_mask.shape[-1] - (input_attention_mask).sum(dim=-1, keepdim=True)
        position_ids -= num_zeros
        position_ids = torch.where(position_ids < 0, 1, position_ids).int()
        position_embeddings = self.model.rotary_emb(input_hidden[0], position_ids)
    
        
        causal_mask = get_causal_mask(
            attention_mask=input_attention_mask, 
            num_attn_heads=self.config.num_attention_heads, 
            total_seq_len=input_attention_mask.shape[1])


        hidden_states = [input_hidden[0]]
        for i, layer in enumerate(self.model.layers): 
            hid = layer(
                hidden_states = input_hidden[i], 
                attention_mask = causal_mask,  # nothing special with this attention mask, 
                position_ids = position_ids, 
                position_embeddings = position_embeddings, 
                use_cache = False
            )
            hidden_states.append(hid)


        # extract hidden states from input_hidden
        input_gist_states = [[] for _ in self.model.layers] #[layer, [batch, [num_gist, hidden]]] 
        for i in range(len(first_gist_index)): 
            gist_states_slice = [layer_state[i, first_gist_index[i]:last_gist_index[i]+1] for layer_state in input_hidden[:-1]] #[layer, [num_gist, hidden]] for batch i
            for j in range(len(input_gist_states)): 
                input_gist_states[j].append(gist_states_slice[j])

        output_gist_states = [[] for _ in self.model.layers] #[layer, [batch, [num_gist, hidden]]] 
        for i in range(len(first_gist_index)): 
            gist_states_slice = [layer_state[i, first_gist_index[i]:last_gist_index[i]+1] for layer_state in hidden_states[1:]] #[layer, [num_gist, hidden]] for batch i
            for j in range(len(output_gist_states)): 
                output_gist_states[j].append(gist_states_slice[j])

        
        critic_outputs = []
        
        for i in range(len(output_gist_states)): 
            stacked_gist = torch.stack(output_gist_states[i], dim=0) #[batch, num_gist, hidden] 
            critic_output = self.critic(stacked_gist) #[batch, num_gist, 1]

            critic_outputs.append(critic_output)
        
        if return_gist_states: 
            return torch.stack(critic_outputs, dim=0), input_gist_states

        return torch.stack(critic_outputs, dim=0)

    def get_neg_critic_outputs(
        self, 
        paraphrase_input_ids: Optional[torch.Tensor] = None, 
        paraphrase_hidden: Optional[torch.Tensor] = None, 
        paraphrase_attention_mask: Optional[torch.Tensor] = None, 
        input_gist_states: torch.Tensor = None, 
    ) -> Any: 
        """
        first, get paraphrase hidden states by forward prop, if paraphrase_hidden is None.

        
        then, for each layer (independently): 
            1. input paraphrase input embeds
            2. input gist states from positive input
            3. obtain gist output for paraphrase + input_gist_states 

        """
        assert self.critic is not None, f"Please add a critic network"
        assert not (paraphrase_input_ids is None and paraphrase_hidden is None), f"one of paraphrase_input_ids or paraphrase_hidden must be defined"
        
        if paraphrase_hidden is None:
            paraphrase_outputs = self(input_ids=paraphrase_input_ids, 
                        attention_mask=paraphrase_attention_mask)

            paraphrase_hidden = paraphrase_outputs.hidden_states 

        # assumption: all sequences have the same number of gist tokens 
        # assumption: all layers have the same batch_size (the obvious default)
        # here we prepare inputs that would be identical across layers.
        num_gist = input_gist_states[0][0].shape[0] 
        batch_size = len(input_gist_states[0])
        causal_mask = get_causal_mask(
            attention_mask=torch.cat(
                (paraphrase_attention_mask, torch.ones((batch_size, num_gist), device=paraphrase_attention_mask.device)), 
                dim=-1), 
            num_attn_heads=self.config.num_attention_heads, 
            total_seq_len=paraphrase_attention_mask.shape[-1] + num_gist)
        # update position_ids with positions of gist tokens
        position_ids = torch.arange(
            paraphrase_attention_mask.shape[1], device=paraphrase_attention_mask.device)[None, :] \
            * torch.ones((paraphrase_attention_mask.shape[0], 1), device=paraphrase_attention_mask.device)
        num_zeros = paraphrase_attention_mask.shape[-1] - (paraphrase_attention_mask).sum(dim=-1, keepdim=True)
        position_ids -= num_zeros
        position_ids = torch.where(position_ids < 0, 1, position_ids).int()
        # print("POSITION IDS SHAPE", position_ids.shape)
        # print("BATCH, NUM_GIST", batch_size, num_gist)
        position_ids = torch.cat((position_ids, torch.zeros((batch_size, num_gist), device=paraphrase_attention_mask.device)), dim=1).int()
        position_embeddings = self.model.rotary_emb(paraphrase_hidden[0], position_ids)
        
        critic_outputs = [] # [layer, batch, num_gist, 1]

        for i, layer in enumerate(self.model.layers): 
            stacked_gist_states = torch.stack(input_gist_states[i], dim=0) # [batch, num_gist, hidden]
            concat_states = torch.cat((paraphrase_hidden[i], stacked_gist_states), dim=1) #[batch, seq_len + num_gist, hidden]
            
            
            hidden_states = layer(hidden_states=concat_states, 
                attention_mask=causal_mask, 
                position_ids=position_ids, 
                position_embeddings=position_embeddings, 
                use_cache=False, 
            )

            gist_output = hidden_states[:, paraphrase_attention_mask.shape[-1]:]
            assert gist_output.shape[1] == num_gist 
            critic_output = self.critic(gist_output)
            
            critic_outputs.append(critic_output)
        
        return torch.stack(critic_outputs, dim=0)

    def add_critic_network(self, critic: nn.Module):
        self.critic = critic




class CLlamaForSequenceClassification(GenericForSequenceClassification, CLlamaPreTrainedModel): ...


class CLlamaForQuestionAnswering(GenericForQuestionAnswering, CLlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class CLlamaForTokenClassification(GenericForTokenClassification, CLlamaPreTrainedModel): ...


AutoModel.register(CLlamaConfig, CLlamaModel)
AutoModelForCausalLM.register(CLlamaConfig, CLlamaForCausalLM)


__all__ = [
    "CLlamaForCausalLM",
    "CLlamaModel",
    "CLlamaPreTrainedModel",
    "CLlamaForSequenceClassification",
    "CLlamaForQuestionAnswering",
    "CLlamaForTokenClassification", 
    "MICLlamaForCausalLM"
]
