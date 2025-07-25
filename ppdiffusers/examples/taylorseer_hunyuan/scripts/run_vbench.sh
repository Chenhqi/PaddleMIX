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

# 设置模型子目录名称
NAME="taylorseer_fs5_N5"

# 设置基础路径
BASE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan"
VIDEO_PATH="${BASE}/${NAME}"
SAVE_PATH="${BASE}/eval/${NAME}"

# 执行评估任务
CUDA_VISIBLE_DEVICES=4 python vbench/run_vbench.py --video_path "$VIDEO_PATH" --save_path "$SAVE_PATH"
CUDA_VISIBLE_DEVICES=4 python vbench/cal_vbench.py --score_dir "$SAVE_PATH"



# # 设置模型子目录名称
# NAME="firstpredict_fs5_cnt2_rel0.15_bO2"

# # 设置基础路径
# BASE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan"
# VIDEO_PATH="${BASE}/${NAME}"
# SAVE_PATH="${BASE}/eval/${NAME}"

# # 执行评估任务
# CUDA_VISIBLE_DEVICES=4 python vbench/run_vbench.py --video_path "$VIDEO_PATH" --save_path "$SAVE_PATH"
# CUDA_VISIBLE_DEVICES=4 python vbench/cal_vbench.py --score_dir "$SAVE_PATH"




# 设置模型子目录名称
NAME="firstpredict_fs5_cnt5_rel0.36_bO3"

# 设置基础路径
BASE="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_wan"
VIDEO_PATH="${BASE}/${NAME}"
SAVE_PATH="${BASE}/eval/${NAME}"

# 执行评估任务
CUDA_VISIBLE_DEVICES=4 python vbench/run_vbench.py --video_path "$VIDEO_PATH" --save_path "$SAVE_PATH"
CUDA_VISIBLE_DEVICES=4 python vbench/cal_vbench.py --score_dir "$SAVE_PATH"
