"""
Wan2.2 I2V (A14B) Model Integration
"""

import sys
import os
import os.path as osp
import argparse
import logging
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from utils.logger import setup_logger, log_worker_identity
from downstream.utils.worker_manager import worker_main
from downstream.api_models import DiffuserModel, images_to_tensor, process_input_dict, process_output_dict
from safetensors.torch import load_file
from diffusers.utils import load_image
import json

try:
    from diffusers import WanImageToVideoPipeline
    from diffusers.models import WanTransformer3DModel
    from diffusers.utils import export_to_video, load_image
    print(f"Diffusers version: {__import__('diffusers').__version__}")
except ImportError as e:
    print(f"Error importing diffusers: {e}")
    sys.exit(1)

def load_local_pipeline(
    transformer_path: str,
    base_model_id: str,
    weight_dtype: torch.dtype,
    progress_bar: bool = True
):
    """Load transformer from local ft_dir and base pipeline from hub
    Note: For Wan2.2-I2V-A14B with MoE architecture, this loads the main transformer.
    The model has two experts (high-noise and low-noise) that are handled internally.
    """
    # load transformer
    transformer = WanTransformer3DModel.from_pretrained(
        transformer_path,
        subfolder=None,                 # already point to .../transformer
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True,
    )
    print(f'Transformer weight Loaded from {transformer_path}')

    # For MoE models, also check for transformer_2 (second expert)
    transformer_2_path = transformer_path.replace("/transformer", "/transformer_2")
    transformer_2 = None
    if os.path.exists(transformer_2_path):
        try:
            transformer_2 = WanTransformer3DModel.from_pretrained(
                transformer_2_path,
                subfolder=None,
                torch_dtype=weight_dtype,
                low_cpu_mem_usage=True,
            )
            print(f'Second Transformer (MoE) weight Loaded from {transformer_2_path}')
        except Exception as e:
            print(f'Warning: Could not load second transformer from {transformer_2_path}: {e}')

    # base pipe (fast local cache)
    pipe_kwargs = {
        "transformer": transformer,
        "torch_dtype": weight_dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }

    # Add second transformer if available (for MoE models)
    if transformer_2 is not None:
        pipe_kwargs["transformer_2"] = transformer_2

    pipe = WanImageToVideoPipeline.from_pretrained(
        base_model_id,
        **pipe_kwargs
    )
    pipe.set_progress_bar_config(disable=not progress_bar)
    return pipe


class Wan22I2VModel(DiffuserModel):
    """
    Wan2.2 I2V Model - Aligned with official pipeline expectations
    """

    def _load_pipe(self):
        base_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

        # Load with optional fine-tuned weights
        if self.args.ft_dir:
            pipe = load_local_pipeline(
                base_model_id=base_id,
                weight_dtype=torch.bfloat16,
                transformer_path=os.path.join(self.args.ft_dir, "transformer"),
            )
        else:
            pipe = WanImageToVideoPipeline.from_pretrained(
                base_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )

        return pipe.to(self.args.device)

    def _load_pipe_args(self):
        # negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        return dict(
            # negative_prompt=negative_prompt,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
            guidance_scale=3.5,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )

    def inference_batch(self, input_dict):
        b_action, save_dirs, return_objects, txt_list, img_list = (
            process_input_dict(input_dict, self.args.task_type, self.args.world_model_name)
        )

        # 3. run the pipeline
        pipe_images = self.generate_pipe_images(txt_list, img_list)
        video_tensors = images_to_tensor(
            pipe_images,
            uni_samp=self.args.num_output_frames,
            save_size=(self.args.out_width, self.args.out_height),
        )

        out = process_output_dict(b_action, save_dirs, return_objects, video_tensors)
        return out

    def generate_pipe_images(self, txt_list, img_list):
        pipe_images = []
        for i, txt in enumerate(txt_list):
            img = None
            if img_list and i < len(img_list) and img_list[i] is not None:
                img = img_list[i]

            pipe_args = self.pipe_args.copy()
            pipe_args["prompt"] = txt

            if img is not None:
                mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
                aspect_ratio = img.height / img.width
                max_area = 480 * 832
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
                img = img.resize((width, height))
                pipe_args["image"] = img
                pipe_args["height"] = height
                pipe_args["width"] = width
            else:
                logging.warning("No image provided for I2V generation.")

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
    print(args)

    input_dir = "downstream/api_models/test_sample/debug"
    image_path = os.path.join(input_dir, 'cond_rgb.png')
    json_path = os.path.join(input_dir, 'action_seq.json')

    # Create output directory
    output_dir = 'downstream/api_models/test_sample/wan22_i2v_debug'
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        b_action = json.load(f)
    input_dict = {
        "b_action": [b_action],
        "save_dirs": [output_dir],
    }

    navigator = Wan22I2VModel(args=args)
    raw_img = Image.open(image_path).convert("RGB")

    aspect_ratio = raw_img.height / raw_img.width
    mod_value = navigator.pipe.vae_scale_factor_spatial * navigator.pipe.transformer.config.patch_size[1]
    args.height = round(np.sqrt(480 * 832 * aspect_ratio)) // mod_value * mod_value
    args.width = round(np.sqrt(480 * 832 / aspect_ratio)) // mod_value * mod_value
    resized_img = raw_img.resize((args.width, args.height))

    # Save resized image to the output directory
    resized_img.save(os.path.join(output_dir, 'cond_rgb.png'))

    return_dict = navigator.inference_batch(input_dict)
    print(f"[wan22_A14B_worker] return_dict: {return_dict}")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wan2.2 I2V Inference script")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--task_type", type=str, default="navigation", choices=["navigation", "manipulation", "freetext"])
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="wan22_debug")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ft_dir", type=str, default=None)
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--num_output_frames", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--debug", action="store_true")

    if '--debug' in sys.argv:
        args = parser.parse_args()
        args.num_frames = 25
        args.num_inference_steps = 10
        test_sample(args)
    else:
        pipe_fd = int(sys.argv[-1])
        args = parser.parse_args(sys.argv[1:-1])

    args.world_model_name = "wan22"

    log_path = osp.join(
        args.log_dir,
        f"{args.exp_id}",
        "wan22_worker",
        f"worker{args.device}.log",
    )
    setup_logger(log_path)
    log_worker_identity(args.device)  # <--- add this line
    print(f"[wan22_worker] All Args:\n {args}")

    navigator = Wan22I2VModel(args=args)
    print(f"[wan22_worker] Wan22I2VModel loaded successfully!")

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)

    worker_main(pipe_fd, do_some_tasks)
