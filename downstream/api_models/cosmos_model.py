"""
conda env create --file downstream/api_models/env_config/cosmos.yaml
conda activate cosmos-predict2

pip install cosmos_guardrail
pip install -r downstream/api_models/env_config/cosmos.txt
pip install flash-attn==2.6.3 --no-build-isolation (optional)
"""

import os
import os.path as osp
import sys
import argparse
from PIL import Image

import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from diffusers import Cosmos2VideoToWorldPipeline
from diffusers.models import (
    CosmosTransformer3DModel,          # class name in your config.json
)
import torch
from torchvision import transforms
from utils.logger import setup_logger
from downstream.utils.saver import save_predict
from downstream.utils.worker_manager import (
    read_pickled_data,
    write_pickled_data,
    read_pickled_data_non_blocking,
    receiver_for_worker,
    worker_main,
)
from downstream.api_models import (
    DiffuserModel,
)

# get HF_TOKEN from environment variable if exists
HF_TOKEN = os.getenv("HF_TOKEN")


def load_local_pipeline(transformer_path: str,
                 base_model_id: str,
                 weight_dtype: torch.dtype,
                 progress_bar: bool = True):
    # load transformer
    transformer = CosmosTransformer3DModel.from_pretrained(
        transformer_path,
        subfolder=None,                 # already point to .../transformer
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    print(f'Transformer weight Loaded from {transformer_path}')

    # base pipe (fast local cache)
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
        base_model_id,
        transformer=transformer,
        torch_dtype=weight_dtype,
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
        # local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=not progress_bar)
    return pipe


class CosmosModel(DiffuserModel):
    """
    https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World/tree/main
    For the 720P model, the input image should be 1280×704; for the 480P model, use 832×480.
    The input video should consist of 5 frames, each with a resolution of 1280×704 for the 720P model,
    or 832×480 for the 480P model.
    # Available checkpoints: nvidia/Cosmos-Predict2-2B-Video2World, nvidia/Cosmos-Predict2-14B-Video2World
    """
    def _load_pipe(self):
        base_id = "nvidia/Cosmos-Predict2-2B-Video2World"

        # 2) optionally patch only specific modules from ft_dir
        if self.args.ft_dir:
            pipe = load_local_pipeline(
                base_model_id=base_id,
                weight_dtype=torch.bfloat16,
                transformer_path=os.path.join(self.args.ft_dir, "transformer"),
            )
        else:
            # 1) load EVERYTHING from the fastest source
            pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
                base_id,
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN,
                low_cpu_mem_usage=True,
            )

        return pipe.to(self.args.device)

    def _load_pipe_args(self):
        """Load the other necessary arguments for the pipeline when call pipe(*args)."""
        negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
        return dict(
            # guidance_scale=8.0,
            negative_prompt=[negative_prompt],
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
            fps=self.args.fps,
        )


def test_sample(args):
    print(args)
    navigator = CosmosModel(args=args)
    input_dict = {
        "b_action": [
            [1] * 2,
            # [2]*1 + [1] * 13,
        ],
        "save_dirs": [
            "downstream/api_models/test_sample/debug/",
        ],
    }
    return_dict = navigator.inference_batch(input_dict)
    print(f"[cosmos_worker] return_dict: {return_dict}")
    # print(navigator.pipe.scheduler.config)
    # print(navigator.pipe.transformer.config)
    # print(navigator.pipe.config)
    sys.exit(0)


if __name__ == "__main__":
    # The last argument is the w_fd we pass from main
    pipe_fd = int(sys.argv[-1])
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--width", type=int, default=480)   #640
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--task_type", type=str, default="navigation", choices=["navigation", "manipulation", "freetext"])
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="05_Cosmos_Debug")
    parser.add_argument("--ft_dir", type=str,
                        default=None)
                        # default="outputs/cosmos/converted_ft_cosmos2_2b_video2world_navigation")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_output_frames", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(sys.argv[1:-1])
    # args = parser.parse_args()

    if args.debug:
        args.width = 480
        args.height = 480
        # 832×480
        args.num_frames = 25
        args.num_inference_steps = 25
        test_sample(args)

    log_path = osp.join(
        args.log_dir,
        f"{args.exp_id}",
        "cosmos_worker",
        f"worker{args.device}.log",
    )
    setup_logger(log_path)
    print(f"[cosmos_worker] All Args:\n {args}")

    navigator = CosmosModel(args=args)
    print(f"[cosmos_worker] CosmosModel loaded successfully!")

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)


    worker_main(pipe_fd, do_some_tasks)
