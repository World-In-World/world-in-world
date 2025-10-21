"""
conda create -n hunyuan python=3.12
conda activate hunyuan

pip install accelerate==1.6.0
pip install bitsandbytes==0.45.5
pip install torch==2.7.0
pip install triton==3.3.0
pip install numpy==1.26.2
pip install opencv-python==4.9.0.80
pip install tokenizers==0.21.1
pip install -U ftfy imageio-ffmpeg imageio
pip install git+https://github.com/huggingface/diffusers
pip install git+https://github.com/huggingface/transformers

CUDA_VISIBLE_DEVICES=0 python downstream/api_models/hunyuan_model.py
CUDA_VISIBLE_DEVICES=1 python downstream/api_models/hunyuan_model.py
"""

import os
import os.path as osp
import sys
import argparse
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from diffusers import HunyuanVideoImageToVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import load_image

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


class HunyuanModel(DiffuserModel):
    """
    https://huggingface.co/hunyuanvideo-community
    https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuan_video
    Performs best at resolutions of 480, 720, 960, 1280
    """

    def _load_pipe(self):
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            self.args.model_slug,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            self.args.model_slug,
            transformer=transformer,
            torch_dtype=torch.float16,
        )
        pipe.vae.enable_tiling()
        pipe.to(self.args.device)
        return pipe

    def _load_pipe_args(self):
        """Load the other necessary arguments for the pipeline when call pipe(*args)."""
        negative_prompt = "Bright tones, overexposed, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs"
        return dict(
            guidance_scale=8.0,
            negative_prompt=negative_prompt,
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
        )


def test_sample(args):
    print(args)
    navigator = HunyuanModel(args=args)
    input_dict = {
        "b_action": [
            [1] * 2,
            # [2]*1 + [1] * 13,
        ],
        "save_dirs": [
            "/home/jzhan423/scratchayuille1/jzhan423/igenex_code/PLOT/converted_pano",
        ],
    }
    return_dict = navigator.inference_batch(input_dict)
    print(f"[hunyuan_worker] return_dict: {return_dict}")
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
    parser.add_argument("--exp_id", type=str, default="05_Hunyuan_Debug")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument(
        "--model_slug",
        type=str,
        default="hunyuanvideo-community/HunyuanVideo-I2V",
        # default="hunyuanvideo-community/HunyuanVideo-I2V-33ch",
    )
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_output_frames", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(sys.argv[1:-1])
    # args = parser.parse_args()

    if args.debug:
        args.width = 480
        args.height = 480
        args.num_frames = 99
        args.num_inference_steps = 30
        test_sample(args)

    log_path = osp.join(
        args.log_dir,
        f"{args.exp_id}",
        "hunyuan_worker",
        f"worker{args.device}.log",
    )
    setup_logger(log_path)
    print(f"[hunyuan_worker] All Args:\n {args}")

    navigator = HunyuanModel(args=args)
    print(f"[hunyuan_worker] HunyuanModel loaded successfully!")

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)


    worker_main(pipe_fd, do_some_tasks)
