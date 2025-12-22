"""
Wan2.2 TI2V-5B Model Integration (Image-to-Video Only)
This model requires an image as input and generates video based on image + text prompt.
Text-only generation is not supported.
https://github.com/huggingface/diffusers/pull/12006

conda env create --file downstream/api_models/env_config/wan22.yaml
conda activate wan22
pip install -r downstream/api_models/env_config/wan22.txt

pip install diffusers transformers torch torchvision
"""

import sys
import os

import argparse
import logging
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from utils.logger import setup_logger
from downstream.utils.worker_manager import worker_main
from downstream.api_models import DiffuserModel, images_to_tensor
from diffusers.utils import load_image
import json

try:
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan, ModularPipeline
    from diffusers.utils import export_to_video, load_image
    print(f"Diffusers version: {__import__('diffusers').__version__}")
except ImportError as e:
    print(f"Error importing diffusers: {e}")
    sys.exit(1)

class Wan22TI2V5BModel(DiffuserModel):
    """
    Wan2.2 TI2V-5B Model - 5B parameters, image-to-video only (image input required)
    """
    def _load_pipe(self):
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            torch_dtype=torch.bfloat16
        )

        # Set expand_timesteps=True to bypass missing image_processor/image_encoder
        print(f"Original expand_timesteps: {pipe.config.expand_timesteps}")
        pipe.config.expand_timesteps = True
        print(f"Modified expand_timesteps: {pipe.config.expand_timesteps}")

        pipe.enable_model_cpu_offload(device=self.args.device)

        self.wan_image_processor = ModularPipeline.from_pretrained(
            "YiYiXu/WanImageProcessor",
            trust_remote_code=True
        )
        return pipe

    def _load_pipe_args(self):
        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

        return dict(
            negative_prompt=negative_prompt,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
            guidance_scale=5.0,  # TI2V-5B recommended guidance scale
            generator=torch.Generator(device="cuda").manual_seed(42),
        )


    def postprocess_frames(self, pipe_images):
        video_tensors = images_to_tensor(
            pipe_images,
            uni_samp=self.args.num_output_frames,
            save_size=(self.args.out_width, self.args.out_height),
        )
        return video_tensors

    def generate_pipe_images(self, txt_list, img_list):
        pipe_images = []
        for i, txt in enumerate(txt_list):
            img = img_list[i]
            pipe_args = self.pipe_args.copy()
            pipe_args["prompt"] = txt

            processed_image = self.wan_image_processor(
                image=img,
                max_area=self.args.width * self.args.height,
                output="processed_image"
            )

            pipe_args["image"] = processed_image
            pipe_args["height"] = processed_image.height
            pipe_args["width"] = processed_image.width
            # print(f"Processing image: {processed_image.size}")

            output = self.pipe(**pipe_args).frames[0]
            if isinstance(output, list):
                pipe_images.append(output)
            else:
                frame_list = []
                for t in range(len(output)):
                    frame = output[t]
                    if isinstance(frame, torch.Tensor):
                        if frame.min() < 0:
                            frame = (frame + 1) / 2
                        frame = torch.clamp(frame, 0, 1)
                        pil_img = transforms.ToPILImage()(frame)
                    elif isinstance(frame, np.ndarray):
                        if frame.dtype in (np.float32, np.float64):
                            frame = (frame * 255).astype(np.uint8)
                        pil_img = Image.fromarray(frame)
                    else:
                        pil_img = frame
                    frame_list.append(pil_img)
                pipe_images.append(frame_list)

        return pipe_images


def test_sample(args):
    """Test function - using local cond_rgb.png and action_seq.json"""
    navigator = Wan22TI2V5BModel(args=args)
    input_dict = {
        "b_action": [
            [1] * 2,
            # [2]*1 + [1] * 13,
        ],
        "save_dirs": [
            "downstream/api_models/test_sample/test/",
        ],
    }


    return_dict = navigator.inference_batch(input_dict)
    print(f"[wan22_ti2v_worker] return_dict: {return_dict}")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V-5B Inference script")
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--task_type", type=str, default="navigation", choices=["navigation", "manipulation", "freetext"])
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="wan22_ti2v_debug")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_output_frames", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=50)  # TI2V-5B recommended steps
    parser.add_argument("--debug", action="store_true")
    if '--debug' in sys.argv:
        args = parser.parse_args()
        test_sample(args)
    else:
        pipe_fd = int(sys.argv[-1])
        args = parser.parse_args(sys.argv[1:-1])

    args.world_model_name = "wan22"

    log_path = os.path.join(args.log_dir, f"{args.exp_id}", "wan22_ti2v_worker", f"worker{args.device}.log")
    setup_logger(log_path)
    print(f"[wan22_ti2v_worker] All Args:\n {args}")

    navigator = Wan22TI2V5BModel(args=args)
    print(f"[wan22_ti2v_worker] Wan22TI2V5BModel loaded successfully!")

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)

    worker_main(pipe_fd, do_some_tasks)