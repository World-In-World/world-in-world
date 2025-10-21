# %%
import os
from PIL import Image, ImageDraw
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from jaxtyping import Float, Int32, UInt8
import matplotlib.pyplot as plt
import json
import re
import imageio
import sys
import pickle
import argparse, sys
import pickle
import torch
from numpy.typing import NDArray
import os.path as osp
from downstream.utils.worker_manager import convert_to_python_variable, worker_main
from downstream.visualize import read_video_asframes
from downstream.utils.saver import save_video, clean_jpgs_under_folder
from utils.logger import setup_logger

from sam2.build_sam import build_sam2_video_predictor


def show_mask(mask, ax, obj_id=None, random_color=False, save_path=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    # If a save_path is provided, save the mask image to disk
    if save_path is not None:
        # Scale the image from [0, 1] to [0, 255] and convert to uint8 for saving
        mask_to_save = (mask_image * 255).astype(np.uint8)
        im = Image.fromarray(mask_to_save)
        im.save(save_path)
        print(f"Mask image saved to: {save_path}")


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def create_video_writer(fps, out_path, type='rgb'):
    if type == 'rgb':
        color = "yuv444p"
    elif type == 'gray':
        color = "gray"

    mask_writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-loglevel", "error", "-crf", "18", "-preset", "slow", "-pix_fmt", color]
    )

    return mask_writer


#---------------------- New helper functions ----------------------
def _is_video_list(file_names: List[str]) -> bool:
    return len(file_names) == 1 and file_names[0].lower().endswith(".mp4")


def _frames_from_mp4_with_reader(video_path: str, indices: List[int]) -> Dict[int, np.ndarray]:
    """
    Use user's read_video_asframes([video_path]) to load (T, C, H, W),
    then return only requested frames as HWC (RGB). Handles both list and np.ndarray returns.
    """
    vids = read_video_asframes([video_path])  # user's fn
    # Support both possible returns: list of (T,C,H,W) or np.ndarray (B,T,C,H,W)/(T,C,H,W)
    vid = vids[0]  # remove batch dimension if present

    T = vid.shape[0]
    frames: Dict[int, np.ndarray] = {}
    for idx in indices:
        if idx < 0 or idx >= T:
            raise IndexError(f"Requested frame {idx}, but video has {T} frames")
        chw = vid[idx]                      # (C, H, W)
        frames[idx] = np.transpose(chw, (1, 2, 0))  # -> (H, W, C)
    return frames

def _load_frames_for_indices(video_dir: str, file_names: List[str], indices: List[int]) -> Dict[int, np.ndarray]:
    """Return {frame_idx: np.ndarray(H,W,3)} from either JPG folder or single MP4."""
    if _is_video_list(file_names):
        assert len(file_names) == 1, "Expected a single video file in the list."
        video_path = os.path.join(video_dir, file_names[0])
        return _frames_from_mp4_with_reader(video_path, indices)

    # Folder of images
    frames: Dict[int, np.ndarray] = {}
    for idx in indices:
        fn = file_names[idx]
        frame_path = os.path.join(video_dir, fn)
        arr = np.array(Image.open(frame_path))
        if arr.ndim == 2:                   # grayscale -> RGB
            arr = np.repeat(arr[..., None], 3, axis=-1)
        frames[idx] = arr                   # (H, W, 3)
    return frames
#---------------------- End New helper functions ----------------------

class VideoPropagationSaver:
    """
    A helper class that saves:
      1) A color video (either blended with segmentation or pure RGB)
      2) One grayscale mask video *per object ID*
    """

    def __init__(self, fps=3, color_video_mode="blended"):
        """
        Args:
            fps (int): Frames per second for the output videos.
            color_video_mode (str): 'blended' or 'pure'
        """
        if color_video_mode not in ["blended", "pure"]:
            raise ValueError("color_video_mode must be 'blended' or 'pure'.")
        self.fps = fps
        self.color_video_mode = color_video_mode

    def _collect_all_obj_ids(self, video_segments):
        """
        Gathers all unique object IDs in the provided video_segments dictionary.

        video_segments is structured as:
            {
              frame_idx: {
                obj_id: binary_mask_array,
                obj_id2: binary_mask_array, ...
              },
              frame_idx2: {...},
              ...
            }

        Returns:
            A sorted list of all unique obj_ids.
        """
        obj_ids = set()
        for frame_idx, obj_dict in video_segments.items():
            obj_ids.update(obj_dict.keys())
        return sorted(obj_ids)

    def _save_color_video(
        self,
        frames_dict,
        video_segments,
        frame_indices,
        out_path
    ):
        """
        Saves a single color video (either 'blended' or 'pure').
        Args:
            frames_dict (dict): Maps frame_idx -> ndarray(H,W,3) in RGB order.
            video_segments (dict): See docstring of _collect_all_obj_ids.
            sorted_frame_indices (list): Sorted list of frame indices to process.
            out_path (str): Output path for the color video.
        Returns:
            str: Path to the saved color video.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = create_video_writer(self.fps, out_path, type='rgb')

        for frame_idx in frame_indices:
            original_img = frames_dict[frame_idx]  # (H,W,3) in RGB

            if self.color_video_mode == "pure":
                # Write the frame directly
                color_frame = original_img
            else:
                # 'blended' mode
                # Convert image to float for alpha blending
                frame_float = original_img.astype(np.float32) / 255.0

                # Overlay each mask with a distinct color
                for obj_id, mask_array in video_segments[frame_idx].items():
                    mask_bool = mask_array.squeeze().astype(bool)
                    color = np.array(plt.get_cmap("tab10")(obj_id % 10))  # RGBA in [0..1]
                    alpha = 0.6
                    c_rgb = color[:3].reshape((1, 1, 3))
                    frame_float[mask_bool, :] = (
                        alpha * c_rgb + (1 - alpha) * frame_float[mask_bool, :]
                    )
                color_frame = (frame_float * 255).astype(np.uint8)

            writer.append_data(color_frame)

        writer.close()
        print(f"==> Color video ({self.color_video_mode}) saved to: {out_path}")
        return out_path

    def _save_per_obj_mask_videos(
        self,
        frames_dict,
        video_segments,
        frame_indices,
        out_mask_path
    ):
        """
        Saves a separate grayscale mask video for each object ID.
        Args:
            frames_dict (dict): Maps frame_idx -> ndarray(H,W,3).
                                We only need shape info from these frames.
            video_segments (dict): frame_idx -> {obj_id: mask_array}
            sorted_frame_indices (list): Sorted list of frame indices to process.
            out_mask_path (str): Base path for mask videos,
                                 e.g. "propagation_mask.mp4" -> "propagation_mask_obj_{id}.mp4".
        Returns:
            dict: Mapping from obj_id -> mask video path for that object.
        """
        base, ext = os.path.splitext(out_mask_path)
        os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)

        # Find all object IDs
        all_obj_ids = self._collect_all_obj_ids(video_segments)

        # Create one writer per object ID
        obj_writers = {}
        obj_mask_paths = {}
        for obj_id in all_obj_ids:
            cur_path = f"{base}_obj_{obj_id}{ext}"
            obj_writers[obj_id] = create_video_writer(self.fps, cur_path, type="gray")
            obj_mask_paths[obj_id] = cur_path

        # Write frames for each object ID
        for frame_idx in frame_indices:
            original_img = frames_dict[frame_idx]  # just to get shape
            height, width = original_img.shape[:2]

            for obj_id in all_obj_ids:
                # Create a blank (0) mask
                mask_frame = np.zeros((height, width), dtype=np.uint8)
                # If this object is in the current frame, set mask pixels to 255
                if obj_id in video_segments[frame_idx]:
                    mask_bool = video_segments[frame_idx][obj_id].squeeze().astype(bool)
                    mask_frame[mask_bool] = 255

                obj_writers[obj_id].append_data(mask_frame)

        # Close all writers
        for obj_id, writer in obj_writers.items():
            writer.close()
            print(f"==> Grayscale mask video for obj_id={obj_id} saved to: {obj_mask_paths[obj_id]}")

        return obj_mask_paths

    def save_propagation_videos(
        self,
        video_dir: str,
        video_segments: dict,   # frame_idx -> {obj_id: mask_array}
        file_names: list,       # now robust (jpg list OR single mp4)
        out_path: str,
        out_mask_path: str = None,
    ) -> Tuple[str, Dict[int, str]]:
        """
        Main entry point to save:
          1) A color video (blended or pure) to `out_path`.
          2) A set of mask videos, one for each object ID, named:
             {out_mask_path base}_obj_{obj_id}.{ext}
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if out_mask_path is None:
            base, ext = os.path.splitext(out_path)
            out_mask_path = f"{base}_mask{ext}"

        sorted_frame_indices = sorted(video_segments.keys())

        # 🔁 uses your reader for mp4; PIL for jpgs
        frames_dict = _load_frames_for_indices(video_dir, file_names, sorted_frame_indices)

        if os.path.exists(out_path):
            print(f"==> Color video already exists at: {out_path}, skipping saving.")
            color_video_path = out_path
        else:
            color_video_path = self._save_color_video(
                frames_dict=frames_dict,
                video_segments=video_segments,
                frame_indices=sorted_frame_indices,
                out_path=out_path,
            )

        obj_mask_paths = self._save_per_obj_mask_videos(
            frames_dict=frames_dict,
            video_segments=video_segments,
            frame_indices=sorted_frame_indices,
            out_mask_path=out_mask_path,
        )

        return color_video_path, obj_mask_paths


def show_masked_imgs(video_dir, video_segments, file_names):
    indices = sorted(video_segments.keys())
    frames = _load_frames_for_indices(video_dir, file_names, indices)

    for idx in indices:
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {idx}")
        plt.imshow(frames[idx])  # HWC RGB
        for out_obj_id, out_mask in video_segments[idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

class SAM2VideoPredictor:
    def __init__(self, args):
        checkpoint = args.ckpt_path
        # model_cfg = args.cfg_path
        model_cfg = f"configs/sam2.1/{os.path.basename(args.cfg_path)}"
        self.model = build_sam2_video_predictor(model_cfg, checkpoint, device=args.device)
        self.saver = VideoPropagationSaver(fps=3, color_video_mode="pure")

    @torch.no_grad()
    def __call__(
        self,
        video_dir: str,
        bbox: Dict[str, int],
        frame_idx: int = 0,
        save_masks: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        inference_state = self.scan_tobe_processed_files(video_dir)

        if isinstance(bbox, dict):
            bboxs_tfm = np.array(
                [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]],
                dtype=np.float32,
            )
            # resize bbox_tfm into (1, 4) shape
            bboxs_tfm = np.tile(bboxs_tfm, (1, 1))
            obj_ids = [1]
        elif isinstance(bbox, np.ndarray) or isinstance(bbox, list):
            bboxs_tfm = np.array(bbox, dtype=np.float32)    # (N, 4) shape
            obj_ids = list(range(1, bboxs_tfm.shape[0] + 1))

        assert bboxs_tfm.ndim == 2 and bboxs_tfm.shape[1] == 4, \
            f"bbox_tfm should be of shape (N, 4), but got {bboxs_tfm.shape}"

        for bbox_coord, ann_obj_id in zip(bboxs_tfm, obj_ids):
            _, out_obj_ids, out_mask_logits = self.model.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,  # the frame index we interact with
                obj_id=ann_obj_id,
                box=bbox_coord,
                # points=points, # labels=labels,
            )

        video_segments = self.propagate_video(inference_state)
        if save_masks:
            out_path = os.path.join(video_dir, "gen_video.mp4")
            out_paths, out_mask_paths = self.saver.save_propagation_videos(
                video_dir=video_dir,
                video_segments=video_segments,
                file_names=self.file_names,
                out_path=out_path
            )
            paths: Dict = {"origin_rgb": out_paths} | out_mask_paths
        else:
            paths = {}

        self.model.reset_state(inference_state)
        return video_segments, paths


    def scan_tobe_processed_files(self, video_dir: str):
        # scan all the JPEG frame names in this directory
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        video_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".mp4"]
        ]
        # consistant with the code in official sam2:
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        if (len(frame_names) > 0 and len(video_names) == 0):
            self.file_names = frame_names
            inference_state = self.model.init_state(video_path=video_dir)
        elif (len(frame_names) == 0 and len(video_names) == 1):
            self.file_names = video_names
            inference_state = self.model.init_state(video_path=osp.join(video_dir, video_names[0]))
        else:
            raise ValueError(
                f"Expected either JPEG frames or a single MP4 video in {video_dir}, "
                f"but found {len(frame_names)} frames and {len(video_names)} videos."
            )
        return inference_state

    def propagate_video(self, inference_state):
        # 1. run propagation throughout the video and collect the results in a dict
        video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.model.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()

        return video_segments

        # 3. (Optional) return the video_masks in the format of {frame_idx: mask_array}
        # video_mask = {}
        # for frame_idx, obj_mask_dict in video_segments.items():
        #     assert len(obj_mask_dict) == 1
        #     mask = list(obj_mask_dict.values())[0]
        #     video_mask[frame_idx] = mask.tolist()
        # return video_mask


# %%
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SAM2 worker")
    p.add_argument("--log_dir", type=str, default="downstream/logs")
    p.add_argument("--exp_id", type=str, default="03.15_usegenex")
    p.add_argument("--ckpt_path", type=str, default="path/to/checkpoint")
    p.add_argument("--cfg_path", type=str, default="path/to/config")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--debug", action="store_true")
    return p

def make_logger(args):
    log_path = osp.join(args.log_dir, f"{args.exp_id}", "sam2_worker", f"worker{args.device}.log")
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    setup_logger(log_path)
    print(f"[sam2_worker] All Args:\n {args}")

def build_model(args) -> SAM2VideoPredictor:
    sam2 = SAM2VideoPredictor(args)
    print(f"[sam2_worker] Initialized SAM2VideoPredictor on {args.device}.")
    return sam2


if __name__ == "__main__":
    parser = build_arg_parser()     #1.
    if "--debug" in sys.argv:
        args = parser.parse_args()
        pipe_fd = None
        # test_sample(args)
    else:
        # The launcher passes the write-end FD as the last argv item
        pipe_fd = int(sys.argv[-1])
        args = parser.parse_args(sys.argv[1:-1])

    make_logger(args)               #2.
    sam2 = build_model(args)        #3.

    def do_some_tasks(input_dict: dict) -> dict:
        save_dirs = input_dict["save_dirs"]
        bbox_coords = input_dict["bbox_coords"]
        pred_frames = input_dict["pred_frames"] # UInt8[NDarray, "B T C H W"]

        out_paths = []
        # If you need video_segments per job, you can also collect them similar to out_paths.
        for save_dir, bbox_coord, frames in zip(save_dirs, bbox_coords, pred_frames):
            save_video(frames, save_path=os.path.join(save_dir, "gen_video.mp4"))

            # SAM2VideoPredictor.__call__(video_dir, bbox, ...)
            video_segments, paths = sam2(save_dir, bbox_coord)
            # video_segments = convert_to_python_variable(video_segments)

            out_paths.append(paths)
            # cleanup any temporary jpg frames dropped in the directory
            clean_jpgs_under_folder(save_dir)

        # Keep the same returned shape your server expects
        return {"save_dirs": out_paths} #, "video_segments": video_segments}


    # Hand over networking/IPC to the shared worker loop (like the Wan example)
    worker_main(pipe_fd, do_some_tasks)

# %%
