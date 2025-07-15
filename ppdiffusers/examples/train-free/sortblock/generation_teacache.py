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
import os
import pickle
import sys
import time

import paddle
import pandas as pd
import torch
from numpy import average, imag
from tqdm import tqdm

# from tgates import TgateSDXLLoader, TgateSDLoader,TgateFLUXLoader,TgatePixArtAlphaLoader
from ppdiffusers import (
    CogVideoXPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    FluxPipeline,
    LCMScheduler,
    PixArtAlphaPipeline,
    PyramidAttentionBroadcastConfig,
    StableDiffusionXLPipeline,
    StableVideoDiffusionPipeline,
    UNet2DConditionModel,
    apply_pyramid_attention_broadcast,
)

# from diffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
from ppdiffusers.utils import export_to_video, load_image

sys.stdout.isatty = lambda: False


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of TGATE V2.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="the input prompts",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="the dir of input image to generate video",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default="/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed",
        help="the path to save images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pixart",
        help="[pixart_alpha,sdxl,lcm_sdxl,lcm_pixart_alpha,svd]",
    )
    parser.add_argument(
        "--gate_step",
        type=int,
        default=10,
        help="When re-using the cross-attention",
    )
    parser.add_argument(
        "--sp_interval",
        type=int,
        default=5,
        help="The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).",
    )
    parser.add_argument(
        "--fi_interval",
        type=int,
        default=1,
        help="The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).",
    )
    parser.add_argument(
        "--warm_up",
        type=int,
        default=2,
        help="The time step to warm up the model inference",
    )
    parser.add_argument(
        "--inference_step",
        type=int,
        default=50,
        help="total inference steps",
    )
    parser.add_argument(
        "--deepcache",
        action="store_true",
        default=False,
        help="do deep cache",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation. Set for reproducible results.",
    )
    parser.add_argument(
        "--tgate",
        action="store_true",
        default=False,
        help="do add tgate",
    )
    parser.add_argument(
        "--origin",
        action="store_true",
        default=False,
        help="do add tgate",
    )
    parser.add_argument(
        "--sort_taylor",
        action="store_true",
        default=False,
        help="do add sort_taylor",
    )
    parser.add_argument(
        "--flux_schnell",
        action="store_true",
        default=False,
        help="do add flux_schnell",
    )
    parser.add_argument(
        "--teacache",
        action="store_true",
        default=False,
        help="do add teacache",
    )
    parser.add_argument(
        "--teacache_pab",
        action="store_true",
        default=False,
        help="do add teacache_pab",
    )
    parser.add_argument(
        "--sortblock",
        action="store_true",
        default=False,
        help="do add sortblock",
    )
    parser.add_argument(
        "--teacache_block",
        action="store_true",
        default=False,
        help="do add teacache_block",
    )
    parser.add_argument(
        "--teacache_pab_block",
        action="store_true",
        default=False,
        help="do add teacache_pab_block",
    )
    parser.add_argument(
        "--block",
        action="store_true",
        default=False,
        help="do add block",
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        default="/root/paddlejob/workspace/env_run/test_data/coco1k",
        help="the path of evaluation annotations",
    )

    args = parser.parse_args()
    return args


def read_prompts_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 按双换行（空行）分隔，每个段落是一个 prompt
    prompts = [p.strip() for p in content.split("\n\n") if p.strip()]
    return prompts


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    if args.prompt:
        saved_path = os.path.join(args.saved_path, "test.png")
    elif args.image:
        saved_path = os.path.join(args.saved_path, "test.mp4")
    # 读取 .tsv 文件（tab 分隔）
    df = pd.read_csv(
        os.path.join("/root/paddlejob/workspace/env_run/chq/PaddleMIX_my/ppdiffusers/examples/inference/coco1k.tsv"),
        sep="\t",
    )

    # 假设列名为 "prompt"，提取成 list
    all_prompts = df["caption_en"].tolist()

    # file_path = '/root/paddlejob/workspace/env_run/gxl/paddle_speed/ppdiffusers/examples/taylorseer_flux/prompts/prompt.txt'
    # all_prompts = read_prompts_from_file(file_path)
    # all_prompts = pickle.load(open(args.anno_path, "rb"))

    # Create generator if seed is provided
    generator = None
    if args.seed is not None:
        generator = paddle.Generator().manual_seed(args.seed)
    # 原始生成的
    if args.origin == True:
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
        saved_path = os.path.join(args.saved_path, "origin_1k_bf16")
        os.makedirs(saved_path, exist_ok=True)
        for i, prompt in enumerate(tqdm(all_prompts)):
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=args.inference_step,
                max_sequence_length=512,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))

    if args.sort_taylor == True:
        import paddle
        from forwards import SortTaylor_forward

        from ppdiffusers import DiffusionPipeline
        from ppdiffusers.utils import logging

        logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

        num_inference_steps = 50
        seed = 42
        prompt = "An image of a squirrel in Picasso style"

        pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
        # pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

        # TaylorSeer settings
        pipeline.transformer.__class__.num_steps = num_inference_steps

        pipeline.transformer.__class__.forward = SortTaylor_forward
        pipeline.transformer.current_block_residual = [None] * len(pipeline.transformer.transformer_blocks)
        pipeline.transformer.current_block_encoder_residual = [None] * len(pipeline.transformer.transformer_blocks)
        pipeline.transformer.current_single_block_residual = [None] * len(
            pipeline.transformer.single_transformer_blocks
        )
        pipeline.transformer.previous_block_residual = [None] * len(pipeline.transformer.transformer_blocks)
        pipeline.transformer.previous_single_block_residual = [None] * len(
            pipeline.transformer.single_transformer_blocks
        )
        pipeline.transformer.previous_encoder_block_residual = [None] * len(
            pipeline.transformer.single_transformer_blocks
        )
        pipeline.transformer.result_list = []
        pipeline.transformer.result_single_list = []
        pipeline.transformer.start = 900
        pipeline.transformer.end = 50
        pipeline.transformer.precentage = 1
        pipeline.transformer.step_Num = 1
        pipeline.transformer.step_Num2 = 9
        pipeline.transformer.beta = 0.1
        pipeline.transformer.count = 0

        saved_path = os.path.join(args.saved_path, "sortblock_taylor_900-50_9")
        os.makedirs(saved_path, exist_ok=True)
        total_time = 0
        for i, prompt in enumerate(tqdm(all_prompts)):
            start_time = time.time()
            image = pipeline(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=args.inference_step,
                max_sequence_length=512,
                generator=generator,
            ).images[0]
            image.save(os.path.join(saved_path, f"{i}.png"))
            end_time = time.time()
            # elapsed_time = end_time - start_time
            total_time += end_time - start_time
            # print(f"Elapsed time: {elapsed_time:.2f} seconds")
        average_time = total_time / len(all_prompts)
        print(f"Average time per image generation: {average_time:.2f} seconds")

    else:
        raise Exception("Please sepcify the model name!")
