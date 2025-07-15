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

import tqdm
from videosys.utils.utils import set_seed


def generate_func(pipeline, prompt_list, output_dir, loop: int = 5, kwargs: dict = {}):
    kwargs["verbose"] = False
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            video = pipeline.generate(prompt, seed=l, **kwargs).video[0]
            pipeline.save_video(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"))


def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list
