import os
import torch
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline

import sys
from pathlib import Path
# Ensure repository root is available for imports like `utils` and `downstream`
FILE_DIR = Path(__file__).resolve().parent
PARENT_DIR = FILE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from utils.svd_utils import (
    norm_image,
    get_action_ids,
    apply_discrete_conditioning_dropout,
)
from utils.dataset_utils import revert_pixel_values
from diffusers.utils import load_image
from jaxtyping import Float
from torch import Tensor
from typing import List
import argparse, sys
from downstream.utils.worker_manager import (
    worker_main,
)
from utils.logger import setup_logger
from downstream.api_models import process_input_dict, process_output_dict, images_to_tensor
import os.path as osp


def collect_inference_frames(device, weight_dtype, navigator, val_dataloader):
    ground_truth_frames, generated_frames, actions_seqs = [], [], []
    gt_frames_np = []
    for step, batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        # first, convert images to latent space.
        pixel_values = batch["pixel_values"].to(    #torch.Size([1, 8, 3, 256, 512]), range [-1, 1]
            device=device, non_blocking=True)
        past_obs = batch["past_obs"].to(            #torch.Size([1, 5, 3, 256, 512]), range [-1, 1]
            device=device, non_blocking=True)
        actions = batch["actions"].to(
            device=device, non_blocking=True)
        base_img_path = batch["frame_paths"][0][0]

        past_obs_pixel = norm_image(past_obs)                 #torch.Size([5, 3, 224, 224]), range [0, 1]
        reverted_array = revert_pixel_values(pixel_values)

        # get the PIL images from the reverted array.
        pil_img = Image.fromarray(reverted_array[0])

        #get action_ids and apply conditioning dropout (for classifier-free guidence)
        action_ids = get_action_ids(1, actions, navigator.action_strategy, weight_dtype)
        if navigator.action_strategy == 'action_block':
            _, _, action_ids_uc = apply_discrete_conditioning_dropout(
                encoder_hidden_states=None,
                conditional_latents=None,
                action_conditioning=torch.clone(action_ids),
                bsz=1,
                only_action_dropout=True,
            )
            action_ids = torch.cat([action_ids_uc, action_ids], dim=0)

        generated_frames_ = navigator.inference(    #tensor, range [0, 1]
            actions=action_ids,
            base_img_path=pil_img,
        )
        pixel_values_ = (pixel_values + 1.0) / 2.0     #norm it into [0, 1]
        ground_truth_frames.append(pixel_values_)
        generated_frames.append(generated_frames_)
        actions_seqs.append(actions)        #tensor shape of (1,25)
        gt_frames_np.append(reverted_array)

    return {
        "gt_frames": ground_truth_frames,
        "gt_frames_np": gt_frames_np,
        "gen_frames": generated_frames,
        "actions_seqs": actions_seqs,
    }


class Navigator:
    def __init__(self, args, device):
        self.generations = []
        self.image_width    = args.width
        self.image_height   = args.height
        self.num_past_obs   = args.num_past_obs
        self.num_frames     = args.num_frames
        self.action_strategy        = args.action_strategy
        self.task_type              = args.task_type
        self.action_input_channel          = args.action_input_channel

        # self.generator = torch.manual_seed(1)
        self.generator = torch.Generator(device=device).manual_seed(1)
        self.device = device


    def set_current_image(self, image):
        self.generations.append([image])

    def get_current_image(self):
        return self.generations[-1][-1]

    def clear_movements(self):
        self.generations = []

    def get_all_frames(self):
        flattened_frames = [frame for movement in self.generations for frame in movement]
        return flattened_frames


    def get_pipeline(self, unet_path, svd_path, weight_dtype, progress_bar=False):
        config_dict = {
            'pretrained_model_name_or_path': unet_path,
            'subfolder': 'unet',
            'torch_dtype': weight_dtype,
            'low_cpu_mem_usage': True,
            'action_strategy': self.action_strategy,
            'task_type': self.task_type,
            'num_frames': self.num_frames,
            'action_input_channel': self.action_input_channel,
        }
        try:
            unet = UNetSpatioTemporalConditionModel.from_pretrained(**config_dict)
        except Exception as e:
            print(f"Error when load Unetpretained fp32 version: {e}")
            print(f"Trying to load Unetpretained fp16 version...")
            unet = UNetSpatioTemporalConditionModel.from_pretrained(**config_dict, variant="fp16")
        print(f'Unet Loaded from {unet_path}')

        pipe = StableVideoDiffusionPipeline.from_pretrained(
            svd_path,
            unet=unet,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
            local_files_only=True,
            variant="fp16",
        )
        pipe.set_progress_bar_config(disable=not progress_bar)
        pipe.to(self.device)
        print('Pipeline Loaded')
        self.pipe = pipe
        return pipe

    def save_video(self, frames, save_path, fps=3):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = imageio.get_writer(save_path, fps=fps)

        frames = self.resize_gen_frames(frames)

        if len(frames) == 0:
            print('No Movement to Export.')
            return

        # Add images to the video
        for frame in frames:
            # Convert the PIL image to a numpy array
            frame = np.array(frame)
            writer.append_data(frame)

        # Close the writer to finalize the video
        writer.close()
        print(f'Video saved to: {save_path}')

    def save_video_stitch(self, gen_frames, gt_frames_np, save_path, fps=3):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = imageio.get_writer(save_path, fps=fps)
        # if isinstance(gen_frames[0], torch.Tensor):
        #     gen_frames = gen_frames.cpu().numpy()
        gen_frames = self.resize_gen_frames(gen_frames)

        above, bottom = gen_frames, gt_frames_np
        assert len(bottom) == len(above), "Different number of frames in the two videos."
        frames = [np.concatenate([b, a], axis=0) for b, a in zip(bottom, above)]

        # Add images to the video
        for frame in frames:
            writer.append_data(frame)

        writer.close()
        print(f'Stitched video saved to: {save_path}')

    def resize_gen_frames(self, gen_frames):
        """resize the generated frames to the desired size according to the type of the input frames"""
        if isinstance(gen_frames[0], Image.Image):
            assert gen_frames[0].dtype == np.uint8, "Expected a list of PIL images."
            gen_frames = [f.resize((self.image_width, self.image_height), Image.BICUBIC) for f in gen_frames]
            gen_frames = [np.array(f) for f in gen_frames]
        elif isinstance(gen_frames[0], torch.Tensor):
            assert gen_frames.dim() == 5 and gen_frames.size(0) == 1, "Expected a single video tensor with bs=1."
            gen_frames = gen_frames.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
            gen_frames = (gen_frames * 255).astype(np.uint8)
        return gen_frames


    def save_gif(self, frames, save_path, fps=3):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Calculate the duration of each frame in the GIF
        if save_path.endswith('.mp4'):
            save_path = save_path.replace('.mp4', '.gif')
        duration = int(1000 / fps)  # duration in milliseconds per frame
        frames = self.resize_gen_frames(frames)

        if len(frames) == 0:
            print('No Movement to Export.')
            return

        # Convert frames to PIL images and save as GIF with duration
        frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
        print(f'GIF saved to: {save_path}')

    def save_imgs(self, frames, save_path):
        """Input: a list of PIL images"""
        if save_path.endswith('.mp4'):
            save_path = save_path.replace('.mp4', '')
        os.makedirs(save_path, exist_ok=True)
        frames = [frame.resize((self.image_width, self.image_height), Image.BICUBIC) for frame in frames]

        for i, frame in enumerate(frames):
            frame.save(os.path.join(save_path, f'{i}.png'))
        print(f'PNG frames saved to: {save_path}')



    @torch.no_grad()
    def inference(self, actions, base_img_path, past_obs_pixel=None, out_width=None, out_height=None):

        with torch.inference_mode():
            if isinstance(base_img_path, str):
                base_img = load_image(base_img_path).resize((self.image_width, self.image_height))
            elif isinstance(base_img_path, torch.Tensor):
                base_img = base_img_path
                assert isinstance(past_obs_pixel, torch.Tensor), "Expected past_obs_pixel to be a tensor."
            elif isinstance(base_img_path, Image.Image):
                base_img = base_img_path
                assert past_obs_pixel is None
            elif isinstance(base_img_path, list):
                base_img = base_img_path
                assert past_obs_pixel is None
                assert isinstance(base_img_path[0], Image.Image)
                assert self.action_strategy == 'micro_cond'

            gen_clips: List[Image.Image] = self.pipe(
                base_img,
                height=self.image_height,
                width=self.image_width,
                num_frames=self.num_frames,
                fps=7,
                # decode_chunk_size=8,          # disable for enhance video quality but add VRAM usage
                motion_bucket_id=127,
                noise_aug_strength=0.02,        # can be tuned, 0.09
                num_inference_steps=30,
                added_action_ids=actions,
                past_obs_imgs=past_obs_pixel,
                generator=self.generator,
            ).frames

            if out_width is None and out_height is None:
                out_width, out_height = self.image_width, self.image_height
            video_tensors: Float[Tensor, 'b N C H W'] = images_to_tensor(gen_clips, save_size=(out_width, out_height))
            # if video_tensor.dim() == 4:
            #     video_tensor = video_tensor.unsqueeze(0)
            return video_tensors


if __name__ == '__main__':
    # The last argument is the w_fd we pass from main
    pipe_fd = int(sys.argv[-1])

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="03.15_usegenex")
    parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--num_past_obs", type=int, default=1)
    parser.add_argument("--task_type", type=str, default="navigation", choices=["navigation", "manipulation", "freetext"])

    # Unique arguments for the igenex:
    parser.add_argument("--action_strategy", type=str, default="micro_cond")
    parser.add_argument("--action_input_channel", type=int,
                        help="The number of input channels for action Embedder, 14 for 'navigation', 23 for 'manipulation'")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--unet_path", type=str,
                        default="/data/igenex_DSAI/jzhan423/igenex_code/outputs/02.25_2D_uni_full_noFPSid_3acm/seed_1_0227_172558/checkpoint-24000/unet")
    parser.add_argument("--svd_path", type=str,
                        default="/home/jchen293/igenex_code/checkpoints/models--stabilityai--stable-video-diffusion-img2vid-xt-1-1/snapshots/043843887ccd51926e3efed36270444a838e7861")
    parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    args = parser.parse_args(sys.argv[1:-1])

    log_path = osp.join(args.log_dir, f"{args.exp_id}", f"igenex_worker",f"worker{args.device}.log")
    setup_logger(log_path)
    print(f"[igenex worker] All Args:\n {args}")

    # Initialize global navigator instance.
    navigator = Navigator(args, device=args.device)
    print(f"[igenex worker] Initialized navigator with device {args.device}.")

    # 1. Initialize the inference pipeline if not already done.
    navigator.get_pipeline(
        unet_path=args.unet_path,
        svd_path=args.svd_path,
        weight_dtype=args.weight_dtype,
    )
    navigator.pipe.set_progress_bar_config(disable=False)

    def do_some_tasks(input_dict):
        """
        Processes a task (mimicking ARSolver.b_genex) using global objects.
        Expects input_dict with keys 'b_image' and 'b_action'. Uses the global
        'navigator' and 'args' initialized in main.
        """
        b_action, save_dirs, return_objects, txt_list, pil_images = \
            process_input_dict(input_dict, args.task_type, args.world_model_name)
        # b_action shape Int32[Tensor, "b 14"] or "b 14 8"
        # b_image shape [bs, 3, H, W] or [bs, 4, H, W], Uint8

        if not isinstance(b_action, torch.Tensor):
            b_action = torch.tensor(b_action, dtype=torch.float32)

        bs = b_action.shape[0]
        # 2. Encode action IDs.
        action_ids: Float[Tensor, "bs 14 14"] = get_action_ids(
            bs, b_action, args.action_strategy, args.weight_dtype
        )
        if navigator.action_strategy == "action_block":
            _, _, action_ids_uc = apply_discrete_conditioning_dropout(
                encoder_hidden_states=None,
                conditional_latents=None,
                action_conditioning=torch.clone(action_ids),
                bsz=bs,
                only_action_dropout=True,
            )
            action_ids = torch.cat([action_ids_uc, action_ids], dim=0)

        # 3. real inference:
        video_tensors: Float[Tensor, "bs 14 C H W"] = navigator.inference(
            actions=action_ids.to(args.device),
            base_img_path=pil_images,
            out_width=args.out_width,
            out_height=args.out_height,
        )
        return process_output_dict(b_action, save_dirs, return_objects, video_tensors)

    worker_main(pipe_fd, do_some_tasks)
