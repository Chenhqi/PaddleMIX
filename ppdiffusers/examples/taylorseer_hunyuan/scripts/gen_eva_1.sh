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

# CUDA_VISIBLE_DEVICES=5 python generation_wan_video.py \
# --inference_step 50 \
# --seed 42 \
# --dataset 'vbench' \
# --repeat 0 \
# --firstblock_predicterror_taylor


CUDA_VISIBLE_DEVICES=4 python generation_wan_video.py \
--inference_step 50 \
--seed 42 \
--dataset 'vbench' \
--repeat 0 \
--taylorseer

CUDA_VISIBLE_DEVICES=4 python generation_wan_video.py \
--inference_step 50 \
--seed 42 \
--dataset 'vbench' \
--repeat 0 \
--teacache


# CUDA_VISIBLE_DEVICES=2 python evaluation.py \
# --inference_step 50 \
# --seed 124 \
# --training_path /root/paddlejob/workspace/env_run/test_data/coco1k/1k \
# --generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/origin_50steps_coco1k \
# --speed_generation_path /root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_bf16/taylorseer_coco1k \
# --resolution 1024 