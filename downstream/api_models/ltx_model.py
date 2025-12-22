"""
conda env create --file downstream/api_models/env_config/LTXvideo.yaml
conda activate LTXvideo

pip install -r downstream/api_models/env_config/LTXvideo.txt

# Additional requirements for LTXVideo model:
pip install 'accelerate>=0.26.0' 'bitsandbytes'
"""
import argparse
import os
import os.path as osp
import sys

import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.models import LTXVideoTransformer3DModel
from safetensors.torch import load_file
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm

from utils.logger import setup_logger
from downstream.utils.worker_manager import (
    read_pickled_data,
    write_pickled_data,
    read_pickled_data_non_blocking,
    receiver_for_worker,
    worker_main,
)
from downstream.api_models import DiffuserModel


HF_TOKEN = os.environ.get("HF_TOKEN", None)  # only used if needed by HF hub

# "a-r-r-o-w/LTX-Video-0.9.1-diffusers"
# "./pretrained_models/LTX-Video"
def load_local_pipeline(
    base_model_id: str,
    weight_dtype: torch.dtype,
    transformer_path: str,
    progress_bar: bool = True,
):
    """
    Load base LTXImageToVideoPipeline then override transformer/unet from local ft dirs,
    but do it in a single from_pretrained() call (no in-place patching).
    """
    transformer = LTXVideoTransformer3DModel.from_pretrained(
        transformer_path,
        subfolder=None,                 # already point to .../transformer
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    print(f"[LTXvideo] Transformer loaded from {transformer_path}")

    pipe = LTXImageToVideoPipeline.from_pretrained(
        base_model_id,
        transformer=transformer,
        torch_dtype=weight_dtype,
        token=HF_TOKEN,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    pipe.set_progress_bar_config(disable=not progress_bar)
    return pipe


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class LTXvideo(DiffuserModel):
    def _load_pipe(self):
        base_id = self.args.ckpt_path

        if self.args.ft_dir:
            transformer_dir = os.path.join(self.args.ft_dir, "transformer")
            # unet_dir = os.path.join(self.args.ft_dir, "unet")  # optional
            pipe = load_local_pipeline(
                base_model_id=base_id,
                weight_dtype=torch.bfloat16,
                transformer_path=transformer_dir,
            )
        else:
            pipe = LTXImageToVideoPipeline.from_pretrained(
                base_id,
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN,
                low_cpu_mem_usage=True,
            )

        return pipe.to(self.args.device)

    def _load_pipe_args(self):
        negative_prompt = (
            "Bright tones, overexposed, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, three legs"
        )
        return dict(
            # negative_prompt=negative_prompt,
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
        )


# --------------------------------------------------------------------------- #
# Debug / Worker                                                              #
# --------------------------------------------------------------------------- #
def test_sample(args):
    print(args)
    navigator = LTXvideo(args=args)
    input_dict = {
        "b_action": [
            [1] * 2
        ],
        "save_dirs": [
            "downstream/api_models/test_sample/debug",
        ],
    }
    return_dict = navigator.inference_batch(input_dict)
    print(f"[LTX_worker] return_dict: {return_dict}")
    sys.exit(0)


if __name__ == "__main__":
    # The last argument is the w_fd we pass from main
    pipe_fd = int(sys.argv[-1])
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--task_type", type=str, default="navigation", choices=["navigation", "manipulation", "freetext"])
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="05.05_LTX")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str, default="a-r-r-o-w/LTX-Video-0.9.1-diffusers")
    parser.add_argument("--ft_dir", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_output_frames", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(sys.argv[1:-1])
    # args = parser.parse_args()
    # add world_model_name of this inference script in args:
    if args.ft_dir:
        args.world_model_name = "FTltx"
    else:
        args.world_model_name = "ltx"

    if args.debug:
        test_sample(args)

    log_path = osp.join(
        args.log_dir,
        f"{args.exp_id}",
        "LTX_worker",
        f"worker{args.device}.log",
    )
    setup_logger(log_path)
    print(f"[LTX_worker] All Args:\n {args}")

    navigator = LTXvideo(args=args)
    print(f"[LTX_worker] LTXvideo loaded successfully!")

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)

    worker_main(pipe_fd, do_some_tasks)
