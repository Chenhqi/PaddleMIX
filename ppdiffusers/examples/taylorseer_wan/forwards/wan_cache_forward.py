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
from wan.modules import WanModel


@torch.compile
def wan_cache_forward(
    self: WanModel, e: torch.Tensor, cond_cache_dict: dict, distance: int, x: torch.Tensor
) -> torch.Tensor:

    for i, block in enumerate(self.blocks):
        x = block.cache_step_forward(x, e=e, layer_cache_dict=cond_cache_dict[i], distance=distance)

    return x
