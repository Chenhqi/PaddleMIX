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

import json
import os
import re

# ===== 配置路径 =====
prompt_json_path = "VBench_full_info.json"  # 包含 prompt_en 的 JSON 文件路径
video_dir = "/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed_hunyuan/taylorseer"  # 视频所在目录
video_suffix = ".mp4"

# 合法化文件名（与第一次相同）
def sanitize_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = name.strip().replace("\n", "").replace("\r", "")
    return name


# ===== 加载 prompt_en 列表 =====
with open(prompt_json_path, "r", encoding="utf-8") as f:
    prompts = json.load(f)

# ===== 重新命名所有文件为统一后缀 =====
for idx, item in enumerate(prompts):
    prompt_text = item["prompt_en"]
    safe_prompt = sanitize_filename(prompt_text)

    # 查找当前旧文件名（之前的格式：<prompt>-<idx>.mp4）
    old_name = f"{safe_prompt}-{idx}{video_suffix}"
    new_name = f"{safe_prompt}-0{video_suffix}"

    old_path = os.path.join(video_dir, old_name)
    new_path = os.path.join(video_dir, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {old_name} → {new_name}")
    else:
        print(f"⚠️ File not found: {old_name}")
