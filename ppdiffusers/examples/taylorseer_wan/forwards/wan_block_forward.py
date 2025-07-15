# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from cache_functions import cal_type
from taylorseer_utils import derivative_approximation, taylor_cache_init, taylor_formula

from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from ppdiffusers.models.modeling_outputs import Transformer2DModelOutput
from ppdiffusers.models.transformer_wan import WanTransformerBlock
from ppdiffusers.utils import (
    USE_PEFT_BACKEND,
    logger,
    scale_lora_layers,
    unscale_lora_layers,
)


def wan_block_forward(
    self: WanTransformerBlock,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor,
    temb: paddle.Tensor,
    rotary_emb: paddle.Tensor,
    current: dict,
    cache_dic: dict,
) -> paddle.Tensor:
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table + temb.astype(paddle.float32)
    ).chunk(6, axis=1)
    # current['type'] = 'full'
    if current["type"] == "full":
        # 1. Self-attention
        current["module"] = "self-attention"
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_hidden_states = (self.norm1(hidden_states.astype(paddle.float32)) * (1 + scale_msa) + shift_msa).cast(
            hidden_states.dtype
        )
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=attn_output)
        hidden_states = (hidden_states.astype(paddle.float32) + attn_output * gate_msa).cast(hidden_states.dtype)

        # 2. Cross-attention
        current["module"] = "cross-attention"
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_hidden_states = self.norm2(hidden_states.astype(paddle.float32)).cast(hidden_states.dtype)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=attn_output)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        current["module"] = "ffn"
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_hidden_states = (self.norm3(hidden_states.astype(paddle.float32)) * (1 + c_scale_msa) + c_shift_msa).cast(
            hidden_states.dtype
        )
        ff_output = self.ffn(norm_hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=ff_output)
        hidden_states = (hidden_states.astype(paddle.float32) + ff_output.astype(paddle.float32) * c_gate_msa).cast(
            hidden_states.dtype
        )
    else:
        distance = current["step"] - current["activated_steps"][-1]
        current["module"] = "self-attention"
        attn_output = taylor_formula(
            derivative_dict=cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]],
            distance=distance,
        )
        hidden_states = (hidden_states.astype(paddle.float32) + attn_output * gate_msa).cast(hidden_states.dtype)
        current["module"] = "cross-attention"
        attn_output = taylor_formula(
            derivative_dict=cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]],
            distance=distance,
        )
        hidden_states = hidden_states + attn_output
        # 3. Feed-forward
        current["module"] = "ffn"
        ff_output = taylor_formula(
            derivative_dict=cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]],
            distance=distance,
        )
        hidden_states = (hidden_states.astype(paddle.float32) + ff_output.astype(paddle.float32) * c_gate_msa).cast(
            hidden_states.dtype
        )
        # hidden_states = wan_attention_cache_forward(sa_dict=self.attn1.state_dict(),ca_dict=self.attn2.state_dict(),ffn_dict=self.ffn.state_dict(),e=(hidden_states,encoder_hidden_states),x=hidden_states,distance=current['distance'])

    return hidden_states
