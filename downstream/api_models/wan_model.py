"""
conda env create --file downstream/api_models/env_config/wan.yaml

pip install -r downstream/api_models/env_config/wan.txt
pip install flash_attn==2.7.4.post1 # install flash_attn at the second step
"""


import argparse
import torch
from diffusers import AutoencoderKLWan, WanTransformer3DModel, WanImageToVideoPipeline
from transformers import UMT5EncoderModel, CLIPVisionModel
from diffusers.utils import export_to_video, load_image, load_video
import numpy as np
from pathlib import Path
import os
import imageio
from PIL import Image
import random
from utils.logger import setup_logger
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
from tqdm import tqdm
import sys
import os.path as osp
from downstream.utils.saver import save_predict
from torchvision import transforms


class WanVideo(DiffuserModel):
    def _load_pipe_args(self):
        # negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        return dict(
            # negative_prompt=negative_prompt,
            height=self.args.height,
            width=self.args.width,
            num_frames=self.args.num_frames,
            num_inference_steps=self.args.num_inference_steps,
        )

    def _load_pipe(self):       # * if not need to load lora (or the model has fused with lora), we should able to use this function
        model_id = self.args.ckpt_path  # e.g., "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"

        # Load encoders
        image_encoder = CLIPVisionModel.from_pretrained(
            model_id, subfolder="image_encoder", torch_dtype=torch.float32
        )
        text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)

        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        # or below is load fused transformer, wan_fused_transformer is the new subfolder which contains the fused transformer:
        # transformer = WanTransformer3DModel.from_pretrained(
        #     model_id, subfolder="wan_fused_transformer", torch_dtype=torch.bfloat16
        # )
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16
        )
        pipe.enable_model_cpu_offload()
        self.pipe = pipe

        return pipe

    def _load_pipe_(self):       # * if need to load lora, the current verion is to load lora from a separate folder, bot fused
        model_id = self.args.ckpt_path                    # base weights

        # --- 1. build the sub‑modules exactly as before --------------------------
        image_encoder = CLIPVisionModel.from_pretrained(
            model_id, subfolder="image_encoder", torch_dtype=torch.float32
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )

        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
        )

        # --- 2. attach LoRA *before* the pipeline is built -----------------------
        if getattr(self.args, "lora_dir", None):
            # `load_attn_procs()` inserts the LoRA weights only into the transformer
            transformer.load_attn_procs(self.args.lora_dir)
            print(f"[LoRA] Adapter loaded from {self.args.lora_dir}")

        # --- 3. assemble the pipeline -------------------------------------------
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()

        self.pipe = pipe
        return pipe

    def resize_for_wan(self, image, max_area=480 * 832):
        # Ensure image fits WAN's patching scheme
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return image.resize((width, height))

    def resize_img(self, image):
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # norm to [-1, 1]
        ])

        img_tensor = transform(image)
        return img_tensor

    def generate_pipe_images(self, txt_list, img_list):
        pipe_images = []
        for img, txt in zip(img_list, txt_list):
            pipe_imgs_list = self.pipe(
                # generator=generator,
                image=img,
                prompt=txt,
                output_type="pil",
                **self.pipe_args,
            ).frames
            pipe_images.extend(pipe_imgs_list)

        return pipe_images


def test_sample(args):
    print(args)
    navigator = WanVideo(args=args)
    input_dict = {
        "b_action": [
            [1] * 5 + [2] * 1 + [1] * 8,
            # [2]*2 + [1] * 12,
        ],
        "save_dirs": [
            # "/home/jzhan423/scratchayuille1/jzhan423/igenex_code/PLOT/converted_pano",
            # "/home/jzhan423/scratchayuille1/jzhan423/igenex_code/PLOT/converted_pano/perspective.png",
            "downstream/api_models/test_sample/test/",
        ],
    }
    return_dict = navigator.inference_batch(input_dict)
    print(f"[Wan_worker] return_dict: {return_dict}")
    sys.exit(0)


if __name__ == "__main__":
    # # The last argument is the w_fd we pass from main
    # pipe_fd = int(sys.argv[-1])
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--task_type", type=str, default="navigation", choices=["navigation", "manipulation", "freetext"])
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="05.11_Wan")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--ckpt_path", type=str,
                        default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    parser.add_argument("--lora_dir", type=str,
                        default=None, help="Folder that contains adapter_config.json and LoRA *.safetensors")
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--num_output_frames", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(sys.argv[1:-1])
    # args = parser.parse_args()
    args.world_model_name = "wan21"

    # if args.debug:
    #     test_sample(args)
    if '--debug' in sys.argv:
        test_sample(args)

    pipe_fd = int(sys.argv[-1])

    log_path = osp.join(
        args.log_dir,
        f"{args.exp_id}",
        "Wan_worker",
        f"worker{args.device}.log",
    )
    setup_logger(log_path)
    print(f"[Wan_worker] All Args:\n {args}")

    navigator = WanVideo(args=args)
    print(f"[Wan_worker] WanVideo loaded successfully!")

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)


    worker_main(pipe_fd, do_some_tasks)
