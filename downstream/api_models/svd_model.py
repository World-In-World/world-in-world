#!/usr/bin/env python3
"""
Inference script for *Stable Video Diffusion* (image-to-video).
conda activate cosmos-predict2
"""
import argparse
import os
import os.path as osp
import sys
from typing import Dict

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import numpy as np
from PIL import Image

from utils.logger import setup_logger
from downstream.utils.worker_manager import (
    read_pickled_data,
    write_pickled_data,
    read_pickled_data_non_blocking,
    receiver_for_worker,
    worker_main,
)
from downstream.api_models import DiffuserModel, images_to_tensor

HF_TOKEN = os.environ.get("HF_TOKEN", None)  # only used if model is gated


class SVD(DiffuserModel):
    """
    A minimal DiffuserModel wrapper around Stable Video Diffusion.
    """

    def _load_pipe(self):
        """
        Loads StableVideoDiffusionPipeline, optionally applying
        LoRA / adapter weights from --ft_dir.
        """
        dtype = self.args.weight_dtype
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.args.ckpt_path,
            torch_dtype=dtype,
            variant="fp16",
            token=HF_TOKEN,
            low_cpu_mem_usage=True,
        )

        # Send the whole pipeline to the desired device
        return pipe.to(self.args.device)

    def _load_pipe_args(self) -> Dict:
        """
        Collects keyword arguments passed to the pipeline __call__().
        SVD infers resolution from the conditioning image, so we do not
        pass width / height.  Most users only tweak num_frames and
        """
        return dict(
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
        )

    def postprocess_frames(self, pipe_images):
        video_tensors = images_to_tensor(
            pipe_images,
            uni_samp=self.args.num_output_frames,
            save_size=(self.args.out_width, self.args.out_height),
        )
        return video_tensors

    def generate_pipe_images(self, txt_list, img_list):
        pipe_images = self.pipe(
            # generator=generator,
            image=img_list,
            **self.pipe_args,
        ).frames

        return pipe_images


# --------------------------------------------------------------------------- #
# Debug / Worker                                                              #
# --------------------------------------------------------------------------- #
def test_sample(args):
    print(args)
    navigator = SVD(args=args)
    input_dict = {
        "b_action": [
            [1] * 2
        ],
        "save_dirs": [
            "downstream/api_models/test_sample/debug",
        ],
    }
    return_dict = navigator.inference_batch(input_dict)
    print(f"[SVD_worker] return_dict: {return_dict}")
    sys.exit(0)


if __name__ == "__main__":
    # The last argument is the w_fd we pass from main
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--task_type", type=str, default="navigation", choices=["navigation", "manipulation", "freetext"])
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="05.05_SVD")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e")
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--num_output_frames", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    parser.add_argument("--debug", action="store_true")
    if '--debug' in sys.argv:
        args = parser.parse_args()
        test_sample(args)
    else:
        pipe_fd = int(sys.argv[-1])
        args = parser.parse_args(sys.argv[1:-1])

    args.world_model_name = "svd"

    log_path = osp.join(
        args.log_dir,
        f"{args.exp_id}",
        "SVD_worker",
        f"worker{args.device}.log",
    )
    setup_logger(log_path)
    print(f"[SVD_worker] All Args:\n {args}")

    navigator = SVD(args=args)
    print(f"[SVD_worker] SVD loaded successfully!")

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)

    worker_main(pipe_fd, do_some_tasks)
