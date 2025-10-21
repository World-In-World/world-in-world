import json
import math
import os
import os.path as osp
import subprocess
import tempfile

import cv2
from tqdm import tqdm
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Tuple
import random

RED = (255, 0, 0)
GREEN = (0, 100, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
dark_colors = [
    # (0, 0, 0),       # Black
    (0, 0, 128),       # Navy Blue
    GREEN,       # Dark Green
    (139, 0, 0),       # Dark Red
    (139, 0, 139),     # Dark Magenta
    (75, 0, 130),      # Indigo
    (54, 69, 79),      # Charcoal
    (217, 82, 241),    # Magenta
    RED,
    (129, 96, 201),    # Purple
    (255, 140, 0),     # Dark Orange
    (255, 20, 147),    # Deep Pink
]


def read_video_asframes(video_paths: list) -> np.ndarray:
    """
    Reads one or more .mp4 videos from a list of file paths and separates them into frames.
    The function returns a numpy array with shape (B, num_frames, C, H, W) where:
    This function handles grayscale videos by adding a channel dimension if needed.
    Args:
        video_paths (list): A list of strings. Each string is the path to a .mp4 video file.
    Returns:
        np.ndarray: Video frames as a numpy array with shape (B, num_frames, C, H, W).
    """
    all_videos = []  # To hold the frames of each video
    if isinstance(video_paths, dict):
        video_paths = list(video_paths.values())

    for video_path in video_paths:
        reader = imageio.get_reader(video_path, 'ffmpeg')
        frames = []

        for frame in reader:
            # If the frame is grayscale, it will have shape (H, W) instead of (H, W, C)
            if frame.ndim == 2:
                frame = np.expand_dims(frame, axis=-1)  # Now shape becomes (H, W, 1)
            # Convert frame from (H, W, C) to (C, H, W)
            frame_chw = np.transpose(frame, (2, 0, 1))
            frames.append(frame_chw)

        # Stack frames along the first axis for this video: (num_frames, C, H, W)
        video_np = np.stack(frames, axis=0)
        all_videos.append(video_np)

    return all_videos


def draw_bbox_from_coord(x_min, x_max, y_min, y_max, rgb_obs, bbox_color, thickness):

    if isinstance(rgb_obs, torch.Tensor):
        rgb_obs = rgb_obs.numpy()
    if rgb_obs.dtype == np.float32:
        rgb_obs = (rgb_obs * 255).astype(np.uint8)

    # If the image is in channels-first format [3, H, W], convert it to channels-last [H, W, 3].
    if rgb_obs.ndim == 3 and rgb_obs.shape[0] == 3:
        rgb_obs = np.transpose(rgb_obs, (1, 2, 0))

    rgb_obs_with_bbox = rgb_obs.copy()
    print(f"OBJECT BBOX:")
    print(f"X RANGE: {x_min:.1f}, {x_max:.1f}")
    print(f"Y RANGE: {y_min:.1f}, {y_max:.1f}")
    cv2.rectangle(rgb_obs_with_bbox, (x_min, y_min), (x_max, y_max), bbox_color, thickness)
    return rgb_obs_with_bbox


def annotate_nav_path(
    pil_image,
    paths,
    label_idxs,
    chosen_action=None,
    scale_factor=1.0,
    font_size=20,
):
    """
    Annotate navigation paths on a list of PIL images using only the default PIL font.
    Args:
        pil_images (List[PIL.Image]): List of PIL images.
        paths (List[Tuple[Tuple[int,int], Tuple[int,int]]]):
            List of (start_px, end_px) pixel coordinates for each action.
        scale_factor (float): Scales line thickness, text size, etc.
        chosen_action (int): Highlights the action with this index in a different color (GREEN).
        base_font_size (int): Base font size before scaling.
    Returns:
        Tuple[List[PIL.Image], List[int]]:
            - The list of annotated PIL images.
            - A list of action indices drawn in order.
    """
    action_idxs = []
    draw = ImageDraw.Draw(pil_image)

    for i, (start_px, end_px) in enumerate(paths):
        # 1) Draw the path line (arrow body)
        sx, sy = map(int, start_px)
        ex, ey = map(int, end_px)
        curr_action_idx = label_idxs[i]
        line_width = math.ceil(5 * scale_factor)
        draw.line([(sx, sy), (ex, ey)], fill=RED, width=line_width)

        # 2) Prepare and measure the action label
        text = str(curr_action_idx)
        # Use textbbox to measure text with the default font at the chosen font_size
        # The returned bounding box is (left, top, right, bottom)
        text_bbox = draw.textbbox((0, 0), text, font_size=font_size)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # 3) Draw a circle at the end point sized to fit the text
        circle_radius = max(text_w, text_h) // 2 + math.ceil(10 * scale_factor)
        circle_bbox = [
            ex - circle_radius,
            ey - circle_radius,
            ex + circle_radius,
            ey + circle_radius,
        ]

        # Choose fill color depending on whether it's the chosen action
        if chosen_action is not None and curr_action_idx == chosen_action:
            fill_color = GREEN
        else:
            fill_color = WHITE

        outline_width = math.ceil(2 * scale_factor)
        draw.ellipse(circle_bbox, fill=fill_color, outline=RED, width=outline_width)

        # 4) Draw the text centered in the circle
        # By specifying anchor="mm", the text is drawn with its center at (ex, ey).
        draw.text((ex, ey), text, fill=BLACK, font_size=font_size, anchor="mm")

    return pil_image


def annotate_frame(
    image,
    text,
    text_color="red",
    bg_color="white",
    font_size=20,
    title_height=30,
    align="center",
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle([(0, 0), (width, title_height)], fill=bg_color)
    if align == "center":
        text_position = (width // 2, title_height // 2)
        anchor = "mm"
    elif align == "bottom":
        # Draw the rectangle at the bottom of the image
        draw.rectangle([(0, height - title_height), (width, height)], fill=bg_color)
        text_position = (width // 2, height - title_height // 2)
        anchor = "mm"
    elif align == "left":
        text_position = (title_height // 2, title_height // 2)
        anchor = "lm"
    elif align == "right":
        text_position = (width - title_height // 2, title_height // 2)
        anchor = "rm"
    draw.text(text_position, text, fill=text_color, anchor=anchor, font_size=font_size)
    return image


def annotate_frame_masks(
    image: Image.Image,
    masks: List[np.ndarray],
    obj_idxs: List[int],
    font_size: int = 20,
    contour_thickness: int = 1,
    rectangle_size: float = 1.3,  # scaling factor for the text background rectangle
    title: str = None,
    use_smooth: bool = False,  # switch to use a simple smooth filter
    contour_alpha: int = 128   # 0 = fully transparent, 255 = fully opaque
) -> Image.Image:
    """
    Annotate a frame with the given masks, placing an object index at the center of each mask.
    The contour of each mask will be drawn with partial transparency.
    Args:
        image (PIL.Image.Image): The image to annotate (RGB).
        masks (List[np.ndarray]): List of boolean masks (True = object pixel) for each object.
        obj_idxs (List[int]): List of object indices corresponding to each mask.
        font_size (int): Font size for the object index text.
        contour_thickness (int): Thickness (in pixels) of the mask outlines.
        rectangle_size (float): Scale factor for the text background rectangle size.
        title (str): Optional title string to draw on the image.
        use_smooth (bool): Whether to apply a simple "smooth" filter to mask edges pre-edge detection.
        contour_alpha (int): Alpha value (0-255) for the contours and rectangle background.
    Returns:
        PIL.Image.Image: Annotated image in RGB mode, with semi-transparent contours drawn.
    """
    assert len(masks) == len(obj_idxs), "Masks and object indices must have the same length."

    # 1) Convert base image to RGBA for alpha-compositing.
    base_rgba = image.convert("RGBA")

    # 2) Create an RGBA overlay (same size, all fully transparent initially).
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay)

    # Prepare contour/background colors for each object index.
    # We'll add alpha to each color when drawing.
    colors = [dark_colors[idx % len(dark_colors)] for idx in obj_idxs]

    for mask, obj_idx, color in zip(masks, obj_idxs, colors):
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)

        # Find foreground pixels
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            continue

        # -- (a) Convert mask to 8-bit PIL image [0 or 255]
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        if use_smooth:
            mask_img = mask_img.filter(ImageFilter.SMOOTH)

        # (b) Edge detection
        edges_img = mask_img.filter(ImageFilter.FIND_EDGES)
        edges = np.array(edges_img)

        # Expand edges for thickness
        edge_y, edge_x = np.where(edges > 0)
        # color: e.g. (R, G, B). We'll add alpha channel:
        # e.g. (r, g, b, 128) means 50% opaque.
        contour_color_rgba = (*color, contour_alpha)

        for px, py in zip(edge_x, edge_y):
            for dy in range(-contour_thickness, contour_thickness + 1):
                for dx in range(-contour_thickness, contour_thickness + 1):
                    draw_overlay.point((px + dx, py + dy), fill=contour_color_rgba)

        # -- (c) Place text at mask centroid, with a semi-transparent background rectangle
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))

        text = str(obj_idx)
        text_bbox = draw_overlay.textbbox((0, 0), text, font_size=font_size)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        text_w = int(text_w * rectangle_size)
        text_h = int(text_h * rectangle_size)

        rect_x0 = centroid_x - text_w // 2
        rect_y0 = centroid_y - text_h // 2
        rect_x1 = rect_x0 + text_w
        rect_y1 = rect_y0 + text_h

        # Draw the rectangle behind the text with the same alpha
        rect_color_alpha = 100
        rect_color_rgba = (*color, min(255, int(rect_color_alpha * 1.5)))
        draw_overlay.rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)],
                               fill=rect_color_rgba)

        # Draw text fully opaque or partially opaque, your call.
        # For fully opaque text, use alpha=255 (white).
        draw_overlay.text(
            (centroid_x, centroid_y),
            text,
            fill=(255, 255, 255, 255),
            anchor="mm",
            font_size=font_size,
        )
    # 4) Alpha-composite the overlay onto the base RGBA
    image_with_alpha = Image.alpha_composite(base_rgba, overlay)

    # 5) Convert back to RGB. The alpha is "baked in," so the semi-transparent
    #    contours look the same, but the final image is an RGB.
    final_image = image_with_alpha.convert("RGB")

    # 3) If there's a title, we can draw it on the overlay as well or on the base
    if title:
        final_image = annotate_frame(final_image, title)

    return final_image

def frames_to_video(frames, video_path, fps):
    """
    Write a GIF, MP4, or AVI from an in-memory list of PIL.Image objects.

    frames : Sequence[PIL.Image.Image]
    video_path : str                 # .gif / .mp4 / .avi
    fps : int
    """
    _, ext = osp.splitext(video_path)
    ext = ext.lower()

    if ext == ".gif":
        # ImageIO wraps Pillow here; duration is per-frame in milliseconds
        imageio.mimsave(
            video_path,
            [np.asarray(f) for f in frames],
            format="GIF",
            fps=fps,
            loop=0,
        )
    else:
        if ext not in (".mp4", ".avi"):
            raise ValueError("Only .gif, .mp4, and .avi are supported.")

        # ImageIO's ffmpeg writer accepts numpy arrays in (H, W, 3/4) uint8
        writer = imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",          # same codec as before
            pixelformat="yuv420p",    # chroma subsampling most players expect
            macro_block_size=None,    # keep odd dimensions; pads internally
            quality=7,                # ~CRF 23 equivalent
        )

        for frame in frames:
            writer.append_data(np.asarray(frame))

        writer.close()
    print(f"==> Video saved to: {video_path}")


def visualize_ar_baseline(
    datum_dir,
    key="rgb_bbox_front",
    label=None,
    answer_file_name="answerer.json",
    planner_file_name="planner.json",
    vis_order="answer_first",  # "answer_first", "planner_first", "answer_only", "planner_only"
):
    if not isinstance(key, str):
        assert isinstance(key, list)
        key = key[0]

    if label is None:
        label_files = [f for f in os.listdir(datum_dir) if f.startswith("LABEL=")]
        label = label_files[0].split("=")[1].split(".")[0]
    frames = []
    for idx, i in enumerate(sorted(os.listdir(datum_dir))):
        if not i.startswith("A"):
            continue
        action_dir = osp.join(datum_dir, i)

        # * get image
        image_path = osp.join(datum_dir, i, f"{key}.png")
        if os.path.exists(image_path):
            image = Image.open(image_path)
            # if the dir not exist jsons but the image exist, we still want to visualize it
            if not osp.exists(osp.join(action_dir, answer_file_name)) and \
               not osp.exists(osp.join(action_dir, planner_file_name)):
                frames.append(image)
        else:
            # reuse the previous img
            image = frames[-1]

        # * 2. Determine what to visualize based on vis_order
        def add_answerer_frame():
            answerer_path = osp.join(action_dir, answer_file_name)
            if osp.exists(answerer_path):
                with open(answerer_path, "r") as f:
                    answerer_output = json.load(f)
                if isinstance(answerer_output, list):  # For AEQA format
                    output1 = answerer_output[0]["Action Plan"]
                    output2 = answerer_output[0]["Chosen Landmark"]
                    view = answerer_output[0]["Chosen View"]
                    answer_annt = f"{i}: Chosen Landmark: <{output2}> | Chosen View: {view} | {output1} "
                elif isinstance(answerer_output, dict): # For AR format
                    best_answer, best_answer_score = list(answerer_output.items())[0]
                    answer_annt = f"{i}: {best_answer_score:.1%}: {best_answer}"

                answer_frame = annotate_frame(
                    image.copy(),
                    answer_annt,
                    text_color="green",
                    bg_color="black",
                )
                frames.append(answer_frame)

        def add_planner_frame():
            planner_path = osp.join(action_dir, planner_file_name)
            if osp.exists(planner_path):
                with open(planner_path, "r") as f:
                    planner_output = json.load(f)
                if isinstance(planner_output, dict):    # For AR format
                    best_action, best_action_score = list(planner_output.items())[0]
                    best_action = best_action[: best_action.find(" degrees")]
                    action_annt = f"{i}: {best_action_score:.1%}: {best_action}"
                elif isinstance(planner_output, list):  # For AEQA format
                    record = planner_output[0]
                    if isinstance(record, dict):
                        heading = next(iter(record.values()))   #assert the first key is heading
                    elif isinstance(record, list):
                        heading = record
                        if isinstance(planner_output[1], dict) and "gripper_states_aft_act" in planner_output[1]:
                            added_info = list(planner_output[1].values())[0]
                            additional = [round(item, 2) for item in added_info[0][:3]] + [added_info[1]]
                            heading = f"{heading}|{additional}"
                    action_annt = f"{i}: {heading}" # | Forward_num: <{num_step}>"

                frame = annotate_frame(
                    image.copy(),
                    action_annt,
                    text_color="red",
                    bg_color="black",
                )
                frames.append(frame)

        # Execute based on vis_order
        if vis_order == "answer_first":
            add_answerer_frame()
            add_planner_frame()
        elif vis_order == "planner_first":
            add_planner_frame()
            add_answerer_frame()
        elif vis_order == "answer_only":
            add_answerer_frame()
        elif vis_order == "planner_only":
            add_planner_frame()
        else:
            raise ValueError(f"Unknown vis_order: {vis_order}. Must be one of: 'answer_first', 'planner_first', 'answer_only', 'planner_only'")

    # * save frames
    frames_to_video(frames, osp.join(datum_dir, f"{label}.mp4"), fps=1)


if __name__ == "__main__":
    data_root = "downstream/states/AR_03.16_ar_filterDEBUG/YVUC4YcDtcY"
    for ep in tqdm(sorted(os.listdir(data_root))):
        if not ep.startswith("E"):
            continue
        print(f"visualizing {ep} ...")
        datum_dir = osp.join(data_root, ep)
        visualize_ar_baseline(datum_dir)
        # break
