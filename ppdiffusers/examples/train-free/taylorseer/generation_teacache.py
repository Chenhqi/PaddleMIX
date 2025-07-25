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
import json
import os
import pickle
import sys
import time

import paddle
import pandas as pd
# from forwards import SortTaylor_forward
from numpy import imag
from paddle.distributed import fleet, get_rank
from paddle.io import DataLoader, Dataset, DistributedBatchSampler
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

# import torch
# from diffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
from ppdiffusers.utils import export_to_video, load_image

sys.stdout.isatty = lambda: False


def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list


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
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation. Set for reproducible results.",
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
        "--hunyuan_sort",
        action="store_true",
        default=False,
        help="do add flux_schnell",
    )
    parser.add_argument(
        "--hunyuan_pab",
        action="store_true",
        default=False,
        help="do add flux_schnell",
    )
    parser.add_argument(
        "--wan_teacache",
        action="store_true",
        default=False,
        help="do add teacache",
    )
    parser.add_argument(
        "--origin_hunyuan",
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
        "--hunyuan_teacache",
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
        "--hunyuan_taylor",
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


class myDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.all_prompts = read_prompt_list(
            "/root/paddlejob/workspace/env_run/chq/PaddleMIX/ppdiffusers/examples/Wan2.1/VBench_full_info.json"
        )
        print(f"[RANK {get_rank()}] Loaded {len(self.all_prompts)} prompts")

    def __getitem__(self, idx):
        return self.all_prompts[idx], idx

    def __len__(self):
        return len(self.all_prompts)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    if args.prompt:
        saved_path = os.path.join(args.saved_path, "test.png")
    elif args.image:
        saved_path = os.path.join(args.saved_path, "test.mp4")
    # 读取 .tsv 文件（tab 分隔）
    # df = pd.read_csv(os.path.join(args.anno_path,"coco1k.tsv"), sep="\t")

    # # 假设列名为 "prompt"，提取成 list
    # all_prompts = df['caption_en'].tolist()

    all_prompts = read_prompt_list(
        "/root/paddlejob/workspace/env_run/chq/PaddleMIX/ppdiffusers/examples/Wan2.1/VBench_full_info.json"
    )
    # file_path = '/root/paddlejob/workspace/env_run/chq/prompt.txt'
    # all_prompts = read_prompts_from_file(file_path)
    # all_prompts = pickle.load(open(args.anno_path, "rb"))

    # Create generator if seed is provided
    generator = None
    if args.seed is not None:
        generator = paddle.Generator().manual_seed(args.seed)
    # 原始生成的

    if args.hunyuan_taylor == True:
        import os
        from forwards import (
            taylorseer_flux_double_block_forward,
            taylorseer_hunyuan_forward,
            taylorseer_flux_single_block_forward,
        )

        import paddle
        from paddlenlp.transformers import LlamaModel
        from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast

        from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        from ppdiffusers.utils import export_to_video

        os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu",
        )
        num_inference_steps = 50
        pipe.transformer.__class__.num_steps = num_inference_steps

        pipe.transformer.__class__.forward = taylorseer_hunyuan_forward

        for double_transformer_block in pipe.transformer.transformer_blocks:
            double_transformer_block.__class__.forward = taylorseer_flux_double_block_forward

        for single_transformer_block in pipe.transformer.single_transformer_blocks:
            single_transformer_block.__class__.forward = taylorseer_flux_single_block_forward

        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        saved_path = os.path.join(args.saved_path, "hunyuan_taylor_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=1)
        for i, prompt in enumerate(tqdm(train_loader)):
            for l in range(2):
                start_time = time.time()
                output = pipe(
                    prompt=prompt[0],
                    height=320,
                    width=512,
                    num_frames=61,
                    num_inference_steps=50,
                    generator=paddle.Generator().manual_seed(l),
                ).frames[0]
                export_to_video(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=15)
            # print(f"Elapsed time: {elapsed_time:.2f} seconds")
        # average_time = total_time / (len(all_prompts) * 2)
        # print(f"Average time per image generation: {average_time:.2f} seconds")
    if args.origin_hunyuan == True:
        import os

        import paddle
        from paddlenlp.transformers import LlamaModel
        from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast

        from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        from ppdiffusers.utils import export_to_video

        os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu",
        )
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        saved_path = os.path.join(args.saved_path, "origin_hunyuan_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=1)
        for i, prompt in enumerate(tqdm(train_loader)):
            for l in range(2):
                start_time = time.time()
                output = pipe(
                    prompt=prompt[0],
                    height=320,
                    width=512,
                    num_frames=61,
                    num_inference_steps=50,
                    generator=paddle.Generator().manual_seed(l),
                ).frames[0]
                export_to_video(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=15)


    if args.hunyuan_teacache == True:
        import os

        import paddle
        from paddlenlp.transformers import LlamaModel
        from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast

        from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        from ppdiffusers.utils import export_to_video
        from teacache_forward import teacache_forward

        os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")

        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu",
        )

        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 50
        pipe.transformer.rel_l1_thresh = 0.15 # 0.1 for 1.6x speedup, 0.15 for 2.1x speedup
        pipe.transformer.accumulated_rel_l1_distance = 0
        pipe.transformer.previous_modulated_input = None
        pipe.transformer.previous_residual = None
        pipe.transformer.__class__.forward = teacache_forward

        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        saved_path = os.path.join(args.saved_path, "hunyuan_teacache_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=1)
        for i, prompt in enumerate(tqdm(train_loader)):
            for l in range(2):
                start_time = time.time()
                output = pipe(
                    prompt=prompt[0],
                    height=320,
                    width=512,
                    num_frames=61,
                    num_inference_steps=50,
                    generator=paddle.Generator().manual_seed(l),
                ).frames[0]
                export_to_video(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=15)

    if args.hunyuan_sort == True:
        import os

        import paddle
        from paddlenlp.transformers import LlamaModel
        from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast

        from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        from ppdiffusers.utils import export_to_video
        from forwards import SortBlock_forward

        os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu",
        )
        num_inference_steps = 50
        pipe.transformer.__class__.num_steps = num_inference_steps
        pipe.transformer.__class__.forward = SortBlock_forward

        pipe.transformer.current_block_residual = [None] * len(pipe.transformer.transformer_blocks)
        pipe.transformer.current_block_encoder_residual = [None] * len(pipe.transformer.transformer_blocks)
        pipe.transformer.current_single_block_residual = [None] * len(pipe.transformer.single_transformer_blocks)
        pipe.transformer.previous_block_residual = [None] * len(pipe.transformer.transformer_blocks)
        pipe.transformer.previous_single_block_residual = [None] * len(pipe.transformer.single_transformer_blocks)
        pipe.transformer.previous_encoder_block_residual = [None] * len(pipe.transformer.single_transformer_blocks)
        pipe.transformer.result_list = []
        pipe.transformer.result_single_list = []
        pipe.transformer.start = 965
        pipe.transformer.end = 50
        pipe.transformer.precentage = 1
        pipe.transformer.step_Num = 1
        pipe.transformer.step_Num2 = 5
        pipe.transformer.beta = 0.1
        pipe.transformer.count = 0

        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        saved_path = os.path.join(args.saved_path, "hunyuan_sort_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=1)
        for i, prompt in enumerate(tqdm(train_loader)):
            for l in range(2):
                start_time = time.time()
                output = pipe(
                    prompt=prompt[0],
                    height=320,
                    width=512,
                    num_frames=61,
                    num_inference_steps=50,
                    generator=paddle.Generator().manual_seed(l),
                ).frames[0]
                export_to_video(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=15)
    if args.hunyuan_pab == True:
        import os
        from ppdiffusers import (
            PyramidAttentionBroadcastConfig,
            apply_pyramid_attention_broadcast,
        )
        import paddle
        from paddlenlp.transformers import LlamaModel
        from paddlenlp.transformers.llama.tokenizer_fast import LlamaTokenizerFast

        from ppdiffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
        from ppdiffusers.utils import export_to_video

        os.environ["SKIP_PARENT_CLASS_CHECK"] = "True"
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", paddle_dtype=paddle.bfloat16
        )
        tokenizer = LlamaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
        text_encoder = LlamaModel.from_pretrained(model_id, subfolder="text_encoder", dtype="float16")
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            paddle_dtype=paddle.float16,
            map_location="cpu",
        )
        config = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=4,
            temporal_attention_block_skip_range=5,
            cross_attention_block_skip_range=6,
            spatial_attention_timestep_skip_range=(50, 1000),
            current_timestep_callback=lambda: pipe._current_timestep,
        )
        apply_pyramid_attention_broadcast(pipe.transformer, config)
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        saved_path = os.path.join(args.saved_path, "hunyuan_pab_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=1)
        for i, prompt in enumerate(tqdm(train_loader)):
            for l in range(2):
                start_time = time.time()
                output = pipe(
                    prompt=prompt[0],
                    height=320,
                    width=512,
                    num_frames=61,
                    num_inference_steps=50,
                    generator=paddle.Generator().manual_seed(l),
                ).frames[0]
                export_to_video(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=15)

    if args.wan_sort == True:
        from forwards import SortBlock_forward

        from ppdiffusers import AutoencoderKLWan, WanPipeline
        from ppdiffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )
        from ppdiffusers.utils import export_to_video_2

        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

        flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.scheduler = scheduler
        pipe.transformer.__class__.forward = SortBlock_forward

        pipe.transformer.current_single_block_residual = [None] * len(pipe.transformer.blocks)
        pipe.transformer.previous_single_block_residual = [None] * len(pipe.transformer.blocks)
        pipe.transformer.result_single_list = []
        pipe.transformer.start = 900
        pipe.transformer.end = 50
        pipe.transformer.precentage = 1
        pipe.transformer.step_Num = 1
        pipe.transformer.step_Num2 = 5
        pipe.transformer.beta = 0.1
        pipe.transformer.count = 0

        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        saved_path = os.path.join(args.saved_path, "wan_sort_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=1)
        total_time = 0
        for i, prompt in enumerate(tqdm(train_loader)):
            for l in range(2):
                start_time = time.time()
                output = pipe(
                    prompt=prompt[0],
                    negative_prompt=negative_prompt,
                    height=480,
                    width=832,
                    num_frames=81,
                    guidance_scale=5.0,
                    generator=paddle.Generator().manual_seed(l),
                ).frames[0]
                end_time = time.time()
                total_time += end_time - start_time
                export_to_video_2(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=16)
            # print(f"Elapsed time: {elapsed_time:.2f} seconds")
        average_time = total_time / (len(all_prompts))
        print(f"Average time per image generation: {average_time:.2f} seconds")
    if args.wan_pab == True:
        from ppdiffusers import (
            AutoencoderKLWan,
            PyramidAttentionBroadcastConfig,
            WanPipeline,
            apply_pyramid_attention_broadcast,
        )
        from ppdiffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )
        from ppdiffusers.utils import export_to_video_2

        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)
        flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.scheduler = scheduler
        config = PyramidAttentionBroadcastConfig(
            spatial_attention_block_skip_range=5,
            temporal_attention_block_skip_range=7,
            cross_attention_block_skip_range=9,
            spatial_attention_timestep_skip_range=(100, 900),
            current_timestep_callback=lambda: pipe._current_timestep,
        )
        apply_pyramid_attention_broadcast(pipe.transformer, config)
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        saved_path = os.path.join(args.saved_path, "wan_pab_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(dataset, batch_sampler=sampler, shuffle=False, num_workers=1)
        total_time = 0
        for i, prompt in enumerate(tqdm(train_loader)):
            # print(prompt[0][0])
            for l in range(2):
                start_time = time.time()
                output = pipe(
                    prompt=prompt[0],
                    negative_prompt=negative_prompt,
                    height=480,
                    width=832,
                    num_frames=81,
                    guidance_scale=5.0,
                    generator=paddle.Generator().manual_seed(l),
                ).frames[0]
                end_time = time.time()
                total_time += end_time - start_time
                export_to_video_2(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=16)
            # print(f"Elapsed time: {elapsed_time:.2f} seconds")
        average_time = total_time / (len(all_prompts) * 2)
        print(f"Average time per image generation: {average_time:.2f} seconds")

    else:
        raise Exception("Please sepcify the model name!")
