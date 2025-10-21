# %%
import argparse
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from jaxtyping import Float, Int32, UInt8
from typing import Any, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
from collections import defaultdict

import sys
import os.path as osp
import re
from sam2_model import SAM2VideoPredictor
from utils.logger import setup_logger
from downstream.utils.worker_manager import convert_to_python_variable, worker_main
from downstream.visualize import (
    read_video_asframes,
)
from ultralytics import YOLOWorld
from downstream.downstream_datasets import get_indoor_classes, BACKGROUND_CLASS
from downstream.detection.post_process import filter_detections, mask_subtract_contained
from utils.util import is_empty



def stack_masks_for_sv(masks_dict: dict) -> tuple[np.ndarray, list]:
    """
    Given a dictionary mapping obj_id -> mask_array,
    and stacks the mask arrays into a single NumPy array (N, H, W).
    Parameters
    ----------
    masks_dict : dict
        Keys are object IDs, values are mask arrays of shape (1, H, W)
        or (H, W).
    Returns
    -------
    masks_stacked : np.ndarray
        Boolean array of shape (N, H, W), where N is the number of detections.
    sorted_keys : list
        The sorted keys from masks_dict corresponding to each row in masks_stacked.
    """
    if not masks_dict:
        # Return empty arrays if masks_dict is empty
        return np.zeros((0,0,0))

    # For each key, .squeeze() if shape is (1, H, W), etc.
    masks_list = [masks_dict[k].squeeze() for k in list(masks_dict.keys())]

    # Stack along axis=0 => shape (N, H, W)
    masks_stacked = np.stack(masks_list, axis=0).astype(bool)
    return masks_stacked


def visulize_objs(
    save_dir: str,
    frames: list[np.ndarray],
    obj_mask_info: list[sv.Detections]
):
    """
    Visualize segmentation masks, bounding boxes, and labels for each frame using Supervision.
    Parameters
    ----------
    save_dir : str
        Directory where annotated frames will be saved as image files (e.g., PNG).
    frames : list of np.ndarray
        Each entry is a frame of shape (H, W, 3) in RGB format.
    obj_mask_info : list of sv.Detections
        A list of Supervision `Detections` objects, one per frame.
        Each `Detections` object is expected to have:
            - `xyxy` (bounding boxes)
            - `mask` (segmentation masks)
            - `class_id` (optional: numeric IDs)
            - `confidence` (optional: detection scores)
            - potentially a custom field for the text label, e.g. `detections.set_value("text_labels", ...)`
    Notes
    -----
    - `len(frames)` must match `len(obj_mask_info)`.
    - If a Detections object is empty for a frame, we assume there are no detections for that frame.
    """
    # Create Supervision annotators (customize as needed)
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.Color(255, 0, 0),  # bounding box color
    )
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=2,
        smart_position=True
    )

    if len(frames) != len(obj_mask_info):
        raise ValueError(
            f"Number of frames ({len(frames)}) does not match number of Detections ({len(obj_mask_info)})"
        )

    for i, detections in enumerate(obj_mask_info):
        # Convert the frame from RGB -> BGR for visualization
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

        if is_empty(detections):
            # No detections for this frame; just save the raw frame
            out_path = os.path.join(save_dir, f"frame_{i:04d}_annotated.png")
            cv2.imwrite(out_path, frame_bgr)
            print(f"No detections for frame {i}, saved raw frame at {out_path}")
            continue

        # If you stored textual labels inside the Detections (e.g. using .set_value("text_labels", ...)),
        # you can retrieve them like this:
        labels = []
        if hasattr(detections, "text_labels") and not is_empty(detections.text_labels):
            text_labels = detections.text_labels
            confidences = detections.confidence  # typically a 1D np.array
            if text_labels is not None and confidences is not None:
                # e.g. build "label 0.97" for each detection
                labels = [
                    f"{lbl} {conf:.2f}"
                    for lbl, conf in zip(text_labels, confidences)
                ]

        # Annotate the frame in the desired order: masks -> boxes -> labels
        annotated_frame = mask_annotator.annotate(
            scene=frame_bgr.copy(),
            detections=detections
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels  # Our custom label strings
        )

        # Save the annotated frame
        out_path = os.path.join(save_dir, f"frame_{i:04d}_annotated.png")
        cv2.imwrite(out_path, annotated_frame)
        print(f"Saved annotated frame at {out_path}")


class GroundingSAM2:
    def __init__(self, args, obj_classes):
        self.args = args
        self.device = args.device
        self.sam2 = SAM2VideoPredictor(args)

        cfg = "yolov8x-world.pt"
        self.detection_model = YOLOWorld(cfg)
        self.detection_model.set_classes(obj_classes)
        self.obj_classes = obj_classes
        print(f"Set detection classes length: {len(obj_classes)}")
        print(f"Load YOLO model {cfg} successful!")


    def detect_objects_in_video(self, video_frames, text_prompt=None):
        # Detect objects
        if text_prompt is not None:
            self.detection_model.set_classes(text_prompt)

        results = self.detection_model.predict(video_frames, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        det_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        det_class_labels = [
            self.obj_classes[class_id]
            for class_idx, class_id in enumerate(det_class_ids)
        ]
        xyxy_np = results[0].boxes.xyxy.cpu().numpy()

        results = [{
            "boxes": xyxy_np,
            "scores": confidences,
            "text_labels": det_class_labels,
            "class_ids": det_class_ids,
        }]
        return results


    def extract_object_masks(self, save_dir, results):
        for i in range(len(results)):
            bbox_ = results[i]["boxes"]
            if len(bbox_) == 0:
                segment = np.zeros((0,0,0))
                print(f"Warning: No detections for frame {i}.")
            else:
                segments, _ = self.sam2(
                    video_dir=save_dir,
                    bbox=bbox_,
                    save_masks=False,
                )
                assert len(segments) == 1
                segment = segments[0]   #segments (dict): frame_idx -> {obj_id: mask_array}
                segment = stack_masks_for_sv(segment)
                # v_frames: UInt8[NDArray, "B 14 C H W"] = read_video_asframes(
                #     video_path,
                # )
                # rgb_frames, mask_frames = v_frames[0], v_frames[1]

            results[i].update({"masks": segment})

        return results

    def compose_detected_prompt(self, text_prompt):
        # Compose the detected objects into a single text prompt
        composed_prompt = ""
        for obj in text_prompt:
            obj = obj.lower()
            composed_prompt += obj + ". "

        return composed_prompt.strip()      # remove the trailing space


    @torch.no_grad()
    def __call__(self, save_dir, text_prompt=None, save_visualize=False):
        # assert isinstance(text_prompt, list) or isinstance(text_prompt, str)
        # if isinstance(text_prompt, list):
        #     text_prompt = self.compose_detected_prompt(text_prompt)

        video_frames = []
        H, W = None, None
        # use re to match the file names, like "rgb_XXXX_000.jpg", "rgb_XXX_001.jpg", ... where XXXX is a str or num with any length
        # scan_pattern = r"rgb_\w+_\d{3}.\w"
        frame_names = [
            p
            for p in os.listdir(save_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        frame_paths = [os.path.join(save_dir, p) for p in frame_names]

        for img_path in frame_paths:
            img = Image.open(img_path).convert("RGB")
            # Check for size consistency
            if H is not None and W is not None and (H, W) != img.size:
                raise ValueError(f"Image size mismatch: {img.size} != {(H, W)}")
            else:
                H, W = img.size
            video_frames.append(np.array(img))  # directly convert to a NumPy array

        # run grounding dino
        results = self.detect_objects_in_video(video_frames, text_prompt)

        # run sam2 on the video frames
        results_ = self.extract_object_masks(save_dir, results)

        obj_mask_info = self.post_process_masks(results_, (H, W))

        if save_visualize:
            visulize_objs(
                save_dir=save_dir,
                frames=video_frames,  # list of frames (np.ndarray)
                obj_mask_info=obj_mask_info  # dictionary with keys "boxes", "masks", "scores", "text_labels"
            )

        # convert the final return into a dict:
        return {
            "masks": obj_mask_info[0].mask,
            "text_labels": obj_mask_info[0].text_labels,
            "boxes": obj_mask_info[0].xyxy,
        }


    def post_process_masks(self, results, HW):
        obj_mask_info = []
        for i in range(len(results)):
            curr_det = sv.Detections(
                xyxy=results[i]["boxes"],
                mask=results[i]["masks"],
                class_id=results[i]["class_ids"],
                confidence=results[i]["scores"],
            )

            # filter the detection by removing overlapping detections
            curr_det_ = filter_detections(
                image_HW=HW,
                detections=curr_det,
                all_text_labels=self.obj_classes,
            )

            if not is_empty(curr_det_):
                curr_det_.mask = mask_subtract_contained(curr_det_.xyxy, curr_det_.mask)
            obj_mask_info.append(curr_det_)

        return obj_mask_info


# %%
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Grounding-SAM2 worker")
    p.add_argument("--log_dir", type=str, default="downstream/logs")
    p.add_argument("--exp_id", type=str, default="03.27_debug_grounding_sam2")
    p.add_argument("--ckpt_path", type=str, default="path/to/checkpoint")
    p.add_argument("--cfg_path", type=str, default="path/to/config")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--debug", action="store_true")
    return p


def make_logger(args):
    log_path = osp.join(args.log_dir, f"{args.exp_id}", "grounding_sam2_worker", f"worker{args.device}.log")
    setup_logger(log_path)
    print(f"[grounding_sam2_worker] All Args:\n {args}")


def build_model(args) -> GroundingSAM2:
    # same content as before: build class list and model
    obj_classes = get_indoor_classes("scannet200")
    obj_classes = [o for o in obj_classes if o not in BACKGROUND_CLASS]
    grounding_sam2 = GroundingSAM2(args, obj_classes)
    print(f"[grounding_sam2_worker] Initialized GroundingSAM2 on {args.device}.")
    # keep your autocast enter behavior
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    return grounding_sam2

def test_sample(args):
    print(args)
    grounding_sam2 = build_model(args)
    input_dict = {
        "save_dirs": ["temp"],
        # "text_prompts": ["bed. table. chair."],
        # "bbox_coords": [[x1, y1, x2, y2], [x3, y3, x4, y4]]
    }
    input_dict["text_prompts"] = [None] * len(input_dict["save_dirs"])
    # out = do_some_tasks(input_dict)
    out = grounding_sam2(input_dict["save_dirs"][0], text_prompt=input_dict["text_prompts"][0])
    print(f"[grounding_sam2_worker] return_dict: {out}")


if __name__ == "__main__":
    parser = build_arg_parser()

    if "--debug" in sys.argv:
        args = parser.parse_args()
        test_sample(args)
        sys.exit(0)
    else:
        # launcher passes the write-end FD as last argv item
        pipe_fd = int(sys.argv[-1])
        args = parser.parse_args(sys.argv[1:-1])

    make_logger(args)
    grounding_sam2 = build_model(args)

    def do_some_tasks(input_dict: dict) -> dict:
        save_dirs = input_dict["save_dirs"]
        text_prompts = input_dict.get("text_prompts", [None] * len(save_dirs))

        obj_mask_infos = []
        for save_dir, text_prompt in zip(save_dirs, text_prompts):
            obj_mask_info = grounding_sam2(save_dir, text_prompt)
            obj_mask_infos.append(convert_to_python_variable(obj_mask_info))

        return {"obj_mask_infos": obj_mask_infos, "save_dirs": save_dirs}


    # hand over IPC to the shared worker loop (like the example)
    worker_main(pipe_fd, do_some_tasks)

