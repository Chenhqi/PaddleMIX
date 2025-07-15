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

import os
import subprocess

# 设置输入和输出目录
video_dir = "/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_wan"  # 替换为你的 MP4 文件夹路径
output_dir = "/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed/origin_wan_webp"  # 输出 WebP 动图路径

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(video_dir):
    if filename.endswith(".mp4"):
        input_path = os.path.join(video_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".webp"
        output_path = os.path.join(output_dir, output_filename)

        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            "fps=16,scale=832:480:flags=lanczos",  # 固定分辨率 & 帧率
            "-loop",
            "0",  # 无限循环
            "-preset",
            "picture",
            output_path,
        ]

        # print(f"Converting {filename} → {output_filename}")
        subprocess.run(cmd)

print("✅ 所有 MP4 转 WebP 动图完成！")
