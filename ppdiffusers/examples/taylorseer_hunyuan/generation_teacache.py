import os
import argparse
import pickle
from numpy import imag
import paddle
import pandas as pd
from tqdm import tqdm
# import torch
# from diffusers import FluxPipeline
from ppdiffusers.models.transformer_flux import FluxTransformer2DModel
# from tgates import TgateSDXLLoader, TgateSDLoader,TgateFLUXLoader,TgatePixArtAlphaLoader
from ppdiffusers import CogVideoXPipeline,PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
import time
from ppdiffusers import StableDiffusionXLPipeline, PixArtAlphaPipeline, StableVideoDiffusionPipeline
from ppdiffusers import UNet2DConditionModel, LCMScheduler,FluxPipeline
from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.utils import load_image, export_to_video
from ppdiffusers import DiffusionPipeline
import json,os
import sys
from paddle.distributed import fleet, get_rank
from paddle.io import Dataset, DistributedBatchSampler, DataLoader
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
        default='/root/paddlejob/workspace/env_run/gxl/output/PaddleMIX/inf_speed',
        help="the path to save images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='pixart',
        help="[pixart_alpha,sdxl,lcm_sdxl,lcm_pixart_alpha,svd]",
    )
    parser.add_argument(
        "--gate_step",
        type=int,
        default=10,
        help="When re-using the cross-attention",
    )
    parser.add_argument(
        '--sp_interval',
        type=int,
        default=5,
        help="The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).",
    )
    parser.add_argument(
        '--fi_interval',
        type=int,
        default=1,
        help="The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).",
    )
    parser.add_argument(
        '--warm_up',
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
        '--seed',
        type=int,
        default=None,
        help='Random seed for generation. Set for reproducible results.',
    )
    parser.add_argument(
        '--origin', 
        action='store_true', 
        default=False, 
        help='do add tgate',
    )
    parser.add_argument(
        '--sort_taylor', 
        action='store_true', 
        default=False, 
        help='do add sort_taylor',
    )
    parser.add_argument(
        '--wan_sort', 
        action='store_true', 
        default=False, 
        help='do add flux_schnell',
    )
    parser.add_argument(
        '--wan_pab', 
        action='store_true', 
        default=False, 
        help='do add flux_schnell',
    )
    parser.add_argument(
        '--wan_teacache', 
        action='store_true', 
        default=False, 
        help='do add teacache',
    )
    parser.add_argument(
        '--wan_taylor', 
        action='store_true', 
        default=False, 
        help='do add teacache',
    )
    parser.add_argument(
        '--origin_wan', 
        action='store_true', 
        default=False, 
        help='do add teacache_pab',
    )
    parser.add_argument(
        '--sortblock', 
        action='store_true', 
        default=False, 
        help='do add sortblock',
    )
    parser.add_argument(
        '--teacache_block', 
        action='store_true', 
        default=False, 
        help='do add teacache_block',
    )
    parser.add_argument(
        '--teacache_pab_block', 
        action='store_true', 
        default=False, 
        help='do add teacache_pab_block',
    )
    parser.add_argument(
        '--block', 
        action='store_true', 
        default=False, 
        help='do add block',
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        default='/root/paddlejob/workspace/env_run/test_data/coco1k',
        help="the path of evaluation annotations",
    )
    
    
    args = parser.parse_args()
    return args

def read_prompts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按双换行（空行）分隔，每个段落是一个 prompt
    prompts = [p.strip() for p in content.split('\n\n') if p.strip()]
    return prompts

class myDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.all_prompts = read_prompt_list("/root/paddlejob/workspace/env_run/chq/PaddleMIX/ppdiffusers/examples/Wan2.1/VBench_full_info.json")
        print(f"[RANK {get_rank()}] Loaded {len(self.all_prompts)} prompts")

    def __getitem__(self, idx):
        return self.all_prompts[idx],idx

    def __len__(self):
        return len(self.all_prompts)


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    if args.prompt:
        saved_path = os.path.join(args.saved_path, 'test.png')
    elif args.image:
        saved_path = os.path.join(args.saved_path, 'test.mp4')
    # 读取 .tsv 文件（tab 分隔）
    # df = pd.read_csv(os.path.join(args.anno_path,"coco1k.tsv"), sep="\t")

    # # 假设列名为 "prompt"，提取成 list
    # all_prompts = df['caption_en'].tolist()

    all_prompts = read_prompt_list("/root/paddlejob/workspace/env_run/chq/PaddleMIX/ppdiffusers/examples/Wan2.1/VBench_full_info.json")
    # file_path = '/root/paddlejob/workspace/env_run/chq/prompt.txt'
    # all_prompts = read_prompts_from_file(file_path)
    # all_prompts = pickle.load(open(args.anno_path, "rb"))
    
    # Create generator if seed is provided
    generator = None
    if args.seed is not None:
        generator = paddle.Generator().manual_seed(args.seed)
    # 原始生成的
    if args.origin == True:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16
        )
        saved_path = os.path.join(args.saved_path,"origin_1k_bf16")
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

    if args.sort_taylor == True :
        pipeline = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
        #pipeline.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
        pipeline.transformer.__class__.num_steps = 50
        pipeline.transformer.__class__.forward = SortTaylor_forward

        pipeline.transformer.current_block_residual = [None] *len(pipeline.transformer.transformer_blocks)
        pipeline.transformer.current_block_encoder_residual = [None] *len(pipeline.transformer.transformer_blocks)
        pipeline.transformer.current_single_block_residual = [None] *len(pipeline.transformer.single_transformer_blocks)
        pipeline.transformer.previous_block_residual = [None] *len(pipeline.transformer.transformer_blocks)
        pipeline.transformer.previous_single_block_residual = [None] *len(pipeline.transformer.single_transformer_blocks)
        pipeline.transformer.previous_encoder_block_residual = [None] *len(pipeline.transformer.single_transformer_blocks)
        pipeline.transformer.result_list = []
        pipeline.transformer.result_single_list = []
        pipeline.transformer.start = 850
        pipeline.transformer.end = 50
        pipeline.transformer.precentage = 1
        pipeline.transformer.step_Num = 1
        pipeline.transformer.step_Num2 = 5
        pipeline.transformer.beta = 0.1
        pipeline.transformer.count = 0

        saved_path = os.path.join(args.saved_path,"sortblock_taylor_850-50_5")
        os.makedirs(saved_path, exist_ok=True)
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
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time:.2f} seconds") 

    if args.origin_wan == True :
        from ppdiffusers import AutoencoderKLWan, WanPipeline
        from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
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
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        saved_path = os.path.join(args.saved_path,"origin_wan_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        drop_last=False,)
        train_loader = DataLoader(dataset,
                                batch_sampler=sampler,
                                shuffle=False,
                                num_workers=1)
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
                total_time+= (end_time - start_time)
                export_to_video_2(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=16) 
            # print(f"Elapsed time: {elapsed_time:.2f} seconds") 
        average_time = total_time / (len(all_prompts)*2)
        print(f"Average time per image generation: {average_time:.2f} seconds")
    if args.wan_teacache == True :
        from ppdiffusers import AutoencoderKLWan, WanPipeline
        from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from ppdiffusers.utils import export_to_video_2
        # from forwards import Teacache_forward
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
        pipe.transformer.__class__.forward = Teacache_forward
        pipe.transformer.enable_teacache = True
        pipe.transformer.cnt = 0
        pipe.transformer.num_steps = 100
        pipe.transformer.coefficients = [2.39676752e+03, -1.31110545e+03,  2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
        pipe.transformer.teacache_thresh = 0.08
        pipe.transformer.accumulated_rel_l1_distance_even = 0
        pipe.transformer.accumulated_rel_l1_distance_odd = 0
        pipe.transformer.previous_e0_even = None
        pipe.transformer.previous_e0_odd = None
        pipe.transformer.previous_residual_even = None
        pipe.transformer.previous_residual_odd = None
        pipe.transformer.use_ref_steps = False
        pipe.transformer.cutoff_steps = 100
        pipe.transformer.ret_steps = 10

        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        saved_path = os.path.join(args.saved_path,"teacache_wan_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        drop_last=False,)
        train_loader = DataLoader(dataset,
                                batch_sampler=sampler,
                                shuffle=False,
                                num_workers=1)
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
                total_time+= (end_time - start_time)
                export_to_video_2(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=16) 
            # print(f"Elapsed time: {elapsed_time:.2f} seconds") 
        average_time = total_time / (len(all_prompts)*2)
        print(f"Average time per image generation: {average_time:.2f} seconds")

    if args.wan_sort == True :
        from ppdiffusers import AutoencoderKLWan, WanPipeline
        from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from ppdiffusers.utils import export_to_video_2
        # from forwards import SortBlock_forward
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
        saved_path = os.path.join(args.saved_path,"wan_sort_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        drop_last=False,)
        train_loader = DataLoader(dataset,
                                batch_sampler=sampler,
                                shuffle=False,
                                num_workers=1)
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
                total_time+= (end_time - start_time)
                export_to_video_2(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=16) 
            # print(f"Elapsed time: {elapsed_time:.2f} seconds") 
        average_time = total_time / (len(all_prompts))
        print(f"Average time per image generation: {average_time:.2f} seconds")
    if args.wan_pab == True :
        from ppdiffusers import AutoencoderKLWan, WanPipeline,PyramidAttentionBroadcastConfig,apply_pyramid_attention_broadcast
        from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
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
        saved_path = os.path.join(args.saved_path,"wan_pab_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        drop_last=False,)
        train_loader = DataLoader(dataset,
                                batch_sampler=sampler,
                                shuffle=False,
                                num_workers=1)
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
                total_time+= (end_time - start_time)
                export_to_video_2(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=16) 
            # print(f"Elapsed time: {elapsed_time:.2f} seconds") 
        average_time = total_time / (len(all_prompts)*2)
        print(f"Average time per image generation: {average_time:.2f} seconds")

    if args.wan_taylor == True :   
        from ppdiffusers import AutoencoderKLWan, WanPipeline
        from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from ppdiffusers.utils import export_to_video_2
        from forwards import wan_forward,wan_block_forward,wan_pipeline
        import time
        strategy = fleet.DistributedStrategy()
        fleet.init(is_collective=True, strategy=strategy)
        # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

        flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
        )
        pipe.scheduler = scheduler
        pipe.__class__.__call__ = wan_pipeline
        pipe.transformer.__class__.forward = wan_forward

        for double_transformer_block in pipe.transformer.blocks:
            double_transformer_block.__class__.forward = wan_block_forward
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        saved_path = os.path.join(args.saved_path,"wan_taylor_vbench")
        os.makedirs(saved_path, exist_ok=True)
        dataset = myDataset(1000)
        sampler = DistributedBatchSampler(dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        drop_last=False,)
        train_loader = DataLoader(dataset,
                                batch_sampler=sampler,
                                shuffle=False,
                                num_workers=1)
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
                total_time+= (end_time - start_time)
                export_to_video_2(output, os.path.join(saved_path, f"{prompt[0][0]}-{l}.mp4"), fps=16) 
            # print(f"Elapsed time: {elapsed_time:.2f} seconds") 
        average_time = total_time / (len(all_prompts)*2)
        print(f"Average time per image generation: {average_time:.2f} seconds")

    else:
        raise Exception('Please sepcify the model name!')

    