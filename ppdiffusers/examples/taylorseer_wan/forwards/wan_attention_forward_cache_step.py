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

import torch
import torch.cuda.amp as amp
from diffusers.configuration_utils import ConfigMixin, register_to_config
from wan.modules.model import WanAttentionBlock

from .wan_attention_cache_forward import wan_attention_cache_forward


def wan_attention_forward_cache_step(
    self: WanAttentionBlock,
    x,
    e,
    layer_cache_dict,
    distance,
):
    r"""
    Args:
        x(Tensor): Shape [B, L, C]
        e(Tensor): Shape [B, 6, C]
        seq_lens(Tensor): Shape [B], length of each sequence in batch
        grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
        freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
    """
    assert e.dtype == torch.float32
    with amp.autocast(dtype=torch.float32):
        e = (self.modulation + e).chunk(6, dim=1)
    assert e[0].dtype == torch.float32

    x = wan_attention_cache_forward(
        sa_dict=layer_cache_dict["self-attention"],
        ca_dict=layer_cache_dict["cross-attention"],
        ffn_dict=layer_cache_dict["ffn"],
        e=e,
        x=x,
        distance=distance,
    )

    return x
