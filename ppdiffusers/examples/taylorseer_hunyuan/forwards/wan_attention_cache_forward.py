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

from typing import Dict

import paddle
import paddle.cuda.amp as amp
from wan.taylorseer.taylorseer_utils import (
    derivative_approximation,
    taylor_cache_init,
    taylor_formula,
)


@torch.compile
def wan_attention_cache_forward(
    sa_dict: Dict, ca_dict: Dict, ffn_dict: Dict, e: tuple, x: paddle.Tensor, distance: int
):

    seer_sa = taylor_formula(derivative_dict=sa_dict, distance=distance)
    seer_ca = taylor_formula(derivative_dict=ca_dict, distance=distance)
    seer_ffn = taylor_formula(derivative_dict=ffn_dict, distance=distance)

    x = cache_add(x, seer_sa, seer_ca, seer_ffn, e)

    return x


def cache_add(x, sa, ca, ffn, e):
    with amp.autocast(dtype=paddle.float32):
        x = x + sa * e[2]
    x = x + ca
    with amp.autocast(dtype=paddle.float32):
        x = x + ffn * e[5]
    return x
