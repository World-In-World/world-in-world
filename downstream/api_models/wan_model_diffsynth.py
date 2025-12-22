"""
conda env create --file downstream/api_models/env_config/wan_diffsynth.yaml
conda activate wan_diffsynth
pip install -r downstream/api_models/env_config/wan_diffsynth.txt
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from PIL import Image
from diffsynth import save_video, load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

from downstream.api_models import DiffuserModel, process_input_dict, process_output_dict, images_to_tensor
from downstream.utils.worker_manager import worker_main
from utils.logger import setup_logger


class WanModel(DiffuserModel):
    """
    Wrapper around WanVideoPipeline with support for ft_method:
      - lora: inject LoRA from lora_path
      - full: load full fine-tuned dit checkpoint from checkpoint
      - no_ft: vanilla pre-trained model
    """

    def _load_pipe(self):
        if "Wan2.2-TI2V-5B" in self.args.model_id:
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=self.args.device,
                model_configs=[
                    ModelConfig(model_id=self.args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device=None,),
                    ModelConfig(model_id=self.args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device=None,),
                    ModelConfig(model_id=self.args.model_id, origin_file_pattern="Wan2.2_VAE.pth", offload_device=None,),
                    ModelConfig(model_id=self.args.model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device=None,),
                ],
            )
        elif "Wan2.2-I2V-A14B" in self.args.model_id:
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device="cuda",
                model_configs=[
                    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors", offload_device=None),
                    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id="Wan-AI/Wan2.2-I2V-A14B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ],
            )

        elif "Wan2.1-I2V-14B-720P" in self.args.model_id:
            pipe = WanVideoPipeline.from_pretrained(
                torch_dtype=torch.bfloat16,
                device=self.args.device,
                model_configs=[
                    ModelConfig(model_id=self.args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device=None,),
                    ModelConfig(model_id=self.args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                    ModelConfig(model_id=self.args.model_id, origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                    ModelConfig(model_id=self.args.model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
                ],
            )
        # Apply fine-tuning method
        ft = self.args.ft_method.lower()
        if ft == "lora":
            assert isinstance(self.args.lora_path, list) and 2 >= len(self.args.lora_path) > 0
            for path in self.args.lora_path:
                if "_low_" in path:
                    print(f"[wan_lora_worker] Loading LoRA weights from {path} into dit2 (low noise model)")
                    pipe.load_lora(pipe.dit2, path, alpha=1.0)
                else:
                    pipe.load_lora(pipe.dit, path, alpha=1.0)
        elif ft == "full":
            if not self.args.checkpoint:
                raise ValueError("ft_method is 'full' but --checkpoint was not provided.")
            # load full fine-tuned state dict into dit
            state_dict = load_state_dict(self.args.checkpoint)
            pipe.dit.load_state_dict(state_dict)
        elif ft == "no_ft":
            pass  # nothing to do
        else:
            raise ValueError(f"Unknown ft_method '{self.args.ft_method}'. Choose from lora/full/no_ft.")

        # pipe.enable_vram_management()
        return pipe.to(self.args.device)

    def _load_pipe_args(self):
        return dict(
            negative_prompt="low quality, blurry, deformed",
            num_inference_steps=self.args.num_inference_steps,
            tiled=True,
            height=self.args.height,
            width=self.args.width,
            num_frames=self.args.num_frames,
            seed=self.args.seed,
        )

    def inference_batch(self, input_dict):
        b_action, save_dirs, return_objects, txt_list, img_list = \
            process_input_dict(input_dict, self.args.task_type, self.args.world_model_name)

        pipe_images = []
        for prompt, img in zip(txt_list, img_list):
            pipe_args = self._load_pipe_args()
            pipe_args.update(
                prompt=prompt,
                input_image=img
            )
            res = self.pipe(**pipe_args)
            frames = res if isinstance(res, list) else res.frames
            pipe_images.append(frames)

        video_tensors = images_to_tensor(
            pipe_images,
            uni_samp=self.args.num_output_frames,
            save_size=(self.args.width, self.args.height),
        )
        print(f"[wan_{args.ft_method}_worker] video_tensors shape: {video_tensors.shape}")

        return process_output_dict(b_action, save_dirs, return_objects, video_tensors)


def test_sample(args):
    print(args)
    navigator = WanModel(args=args)
    input_dict = {
        "b_action": [
            [1] * 14,
            # [2]*2 + [1] * 12,
        ],
        "save_dirs": [
            "downstream/api_models/test_sample/debug",
        ],
    }
    return_dict = navigator.inference_batch(input_dict)
    print(f"[Wan_worker] return_dict: {return_dict}")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WanModel worker script with ft_method")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--task_type", type=str, default="navigation", choices=["navigation", "manipulation", "freetext"])
    parser.add_argument("--num_frames", type=int, default=13)
    parser.add_argument("--num_output_frames", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="wan2.2_worker")
    parser.add_argument("--ft_method", type=str, choices=["lora", "full", "no_ft"], default="no_ft",
                        help="Fine-tuning method to apply.")
    parser.add_argument("--lora_path", type=str, nargs='+',
                        default=[], # default="/home/tlu37/scratchdkhasha1/auzunog1/shared/jzhan423/models/wan_models/navigation_medium/Wan2.2-TI2V-5B_lora_navigation/epoch-0.safetensors",
                        help="Path to LoRA checkpoint (used if ft_method==lora)")
    parser.add_argument("--checkpoint", type=str,
                        # default="/home/tlu37/scratchdkhasha1/auzunog1/shared/jzhan423/models/wan_models/navigation_medium/Wan2.2-TI2V-5B_full_navigation/epoch-0.safetensors",
                        help="Path to full fine-tuned .safetensors (used if ft_method==full)")
    parser.add_argument("--debug", action="store_true")
    if '--debug' in sys.argv:
        args = parser.parse_args()
        test_sample(args)
    else:
        pipe_fd = int(sys.argv[-1])
        args = parser.parse_args(sys.argv[1:-1])

    # add world_model_name of this inference script in args:
    if "wan2.1" in args.model_id:
        args.world_model_name = "wan21"
    elif "wan2.2" in args.model_id:
        args.world_model_name = "wan22"
    else:
        raise ValueError(f"Unknown model_id: {args.model_id}")
    if args.lora_path:
        args.world_model_name = f"FT{args.world_model_name}"

    log_path = os.path.join(args.log_dir, args.exp_id, "FTwan_diffsynth_worker", f"worker{args.device}.log")
    setup_logger(log_path)
    print(f"[wan_{args.ft_method}_worker] All Args:\n {args}")

    navigator = WanModel(args=args)
    print(f"[wan_{args.ft_method}_worker] wan_{args.ft_method} loaded successfully!")

    def do_some_tasks(input_dict):
        return navigator.inference_batch(input_dict)

    worker_main(pipe_fd, do_some_tasks)