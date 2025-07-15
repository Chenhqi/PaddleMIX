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

import argparse

import torch
from vbench import VBench

full_info_path = "./vbench/VBench_full_info.json"

dimensions = [
    "subject_consistency",
    "imaging_quality",
    "background_consistency",
    "motion_smoothness",
    "overall_consistency",
    "human_action",
    "multiple_objects",
    "spatial_relationship",
    "object_class",
    "color",
    "aesthetic_quality",
    "appearance_style",
    "temporal_flickering",
    "scene",
    "temporal_style",
    "dynamic_degree",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    kwargs = {}
    kwargs["imaging_quality_preprocessing_mode"] = "longer"  # use VBench/evaluate.py default

    for dimension in dimensions:
        my_VBench = VBench(torch.device("cuda"), full_info_path, args.save_path)
        my_VBench.evaluate(
            videos_path=args.video_path,
            name=dimension,
            local=False,
            read_frame=False,
            dimension_list=[dimension],
            mode="vbench_standard",
            **kwargs,
        )
