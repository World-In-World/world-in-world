import torch
from torch import Tensor
from jaxtyping import Float, Int32, UInt8, Bool
from typing import Any, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
import numpy as np
import cv2
from scipy import ndimage
import re
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch.nn.functional as F
from torchvision.utils import save_image

from downstream.visualize import (
    read_video_asframes,
    annotate_frame_masks,
    annotate_frame,
)
from utils.svd_utils import (
    rotate_by_degrees,
    rotate_by_shift,
)

from habitat_data.equi2cube import convert_equi2per
from downstream.vlm import VIEW_ORDER, VIEW_ID_OFFSET
from copy import deepcopy


IGENEX_ACTION_IDS = {'forward': 1, 'turn_left': 2, 'turn_right': 3, 'stop': 4, 'placeholder': 0}


def filter_by_distance(landmark_pos, agent_position, dist_thr=2.4):
    dist = np.linalg.norm(
        np.array(landmark_pos) - np.array(agent_position)
    )
    if dist < dist_thr:
        return True
    else:
        return False

def compute_theta_deviation_from_depth(depth_img, hfov, dist_thr=2.4, area_ratio=0.02):
    """
    Identify the largest continuous region in 'depth_img' where depth > dist_thr.
    If its pixel area exceeds area_thr, compute the average x-coordinate (region center)
    and convert it to a small angle offset from the image center. Otherwise, return None.
    Args:
        depth_img (np.ndarray): Single-channel depth image of shape (H, W).
        hfov (float): Horizontal field of view in degrees.
        dist_thr (float): Minimum depth in meters to consider "large open area."
        area_thr (float): Minimum pixel area of the largest region to be valid.
    Returns:
        float or None:
            The angle offset in radians if a valid deep region is found;
            None if no region or insufficient area.
    Notes on Angle Offset:
        - If x_center > width/2, the offset is positive (object is on the agent's right).
        - If x_center < width/2, the offset is negative (object is on the agent's left).
    """
    # 1) Create a boolean mask of where depth > dist_thr.
    mask = (depth_img > dist_thr)

    # 2) Label connected components in the mask.
    labeled, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return None

    # 3) Find the largest connected component by area (pixel count).
    curr_max_area = 0
    best_label = 0
    for label_idx in range(1, num_labels + 1):
        area = np.sum(labeled == label_idx)
        if area > curr_max_area:
            curr_max_area = area
            best_label = label_idx

    # 4) If the largest region is smaller than 'area_thr', ignore it
    if curr_max_area < area_ratio * depth_img.size:
        return None

    # 5) Compute the average x-coordinate of that largest region.
    y_idxs, x_idxs = np.where(labeled == best_label)    #x is width, y is height
    # get x center from the median of x_idxs
    x_center = np.median(x_idxs)
    height, width = depth_img.shape

    # 6) Convert hfov (degrees) to radians.
    hfov_rad = np.radians(hfov)

    # 7) Calculate angle offset from the image center:
    #    If x_center > width/2, angle_offset is positive (object is on the right).
    #    If x_center < width/2, angle_offset is negative (object is on the left).
    angle_offset = (x_center - width / 2) / width * hfov_rad

    return angle_offset


def is_wrapped_by_width(pixel_coords, img_width, wrap_threshold=0.5):
    """
    Checks if the bounding box in equirectangular pixel space is "too wide,"
    implying it wraps around from left to right.

    Args:
        pixel_coords: (8, 2) array of corner pixel coords.
        img_width:    width of the panorama.
        wrap_threshold: fraction of img_width that triggers wrap detection.

    Returns:
        bool: True if it is likely a wrapped bounding box.
    """
    x_min = pixel_coords[:, 0].min()
    x_max = pixel_coords[:, 0].max()
    box_width = x_max - x_min

    # If the bounding box is > 50% of the panorama width, we consider it "wrapped."
    return (box_width > wrap_threshold * img_width)


def compute_2d_bbox_from_8_corners(pixel_coords):
    """
    Given 8 corner coordinates in pixel space (shape: (8, 2)),
    return the 2D bounding box in the form [x_min, y_min, x_max, y_max].
    """
    # pixel_coords: np.ndarray of shape (8, 2), each row: (x, y)
    # If needed, filter out corners that are invalid or behind camera, etc.

    x_coords = pixel_coords[:, 0]
    y_coords = pixel_coords[:, 1]

    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    return np.array([x_min, y_min, x_max, y_max])


def prepare_init_panos(rgb, rotate_degrees, rotate_type="by_degrees"):
    # {action_seq_id: degree}
    image_tensors = []

    for id, degrees in rotate_degrees.items():
        if rotate_type == "by_degrees":
            rgb_transformed = rotate_by_degrees(rgb, degrees)
        elif rotate_type == "by_shift":
            # if degrees > 0: shift > 0
            shift_pixel = int(degrees * rgb.shape[-1] / 360)
            # Rotate the image by the specified angle
            rgb_transformed = rotate_by_shift(rgb, shift_pixel)

        # action_seq = torch.tensor([1] * self.igenex_n_frame).unsqueeze(0)
        # Add a batch dimension to the image: [1, 4, H, W] (assuming original shape [4, H, W])
        image_tensors.append(rgb_transformed.unsqueeze(0))
        # action_tensors.append(action_seq)

    return image_tensors


def compose_turn_actions(sim_actions_space):
    turn_actions = {}
    for action_text, action_items in sim_actions_space.items():
        turn_degree = 0
        for action in action_items:
            # Use re to match the number after an underscore and a letter
            match = re.search(r"_[a-zA-Z]([0-9]+(?:\.[0-9]+)?)", action[1])
            num_str = match.group(1)
            if "turn" in action[0]:
                if "left" in action[0]:
                    turn_degree += float(num_str)
                elif "right" in action[0]:
                    turn_degree -= float(num_str)
            elif "forward" in action[0]:
                turn_degree = 0
        turn_actions[action_text] = turn_degree
    return turn_actions


def post_process_output_ar(output_dict, per_hfov, img_size: Tuple[int, int]=(384, 384)):
    """
    Return:
        rgbs_aligned_w_bbox: List[Tensor[N, 3, H, W]]
            containing aligned RGB frames with bounding boxes.
        rgbs_w_bbox: List[Tensor[N, 3, H, W]]
            containing RGB frames with bounding boxes.
        retain_idxs: List[List[int]]
            containing the retained indices of rgbs_aligned_w_bbox.
    """
    rgbs_aligned_w_bbox = []
    rgbs_w_bbox = []
    retain_idxs = []
    for i in range(len(output_dict["save_dirs"])):
        v_frames: UInt8[NDArray, "B 14 C H W"] = read_video_asframes(
            output_dict["save_dirs"][i],
        )
        rgb_frames, mask_frames = v_frames[0], v_frames[1]  #B=2
        mask_frames = (mask_frames / 255.0).astype(bool)
        # reduce The mask_frames to 1 channel
        mask_frames = mask_frames[:, 0:1]

        out1, _ = generate_aligned_bbox_frames(
            rgb_frames, mask_frames, per_hfov, img_size
        )
        out2_1, out2_2 = generate_bbox_cube_frames(
            rgb_frames, mask_frames, per_hfov, img_size
        )
        rgbs_aligned_w_bbox.append(out1)
        rgbs_w_bbox.append(out2_1)
        retain_idxs.append(out2_2)
    return rgbs_aligned_w_bbox, rgbs_w_bbox, retain_idxs

def post_process_output_ar_non_pano(output_dict, per_hfov,
                                    img_size: Tuple[int, int]=(384, 384), n_frame=14):
    rgbs_aligned_w_bbox = []
    rgbs_w_bbox = []
    retain_idxs = []
    for i in range(len(output_dict["save_dirs"])):
        v_frames: UInt8[NDArray, "B 14 C H W"] = read_video_asframes(
            output_dict["save_dirs"][i],
        )
        assert v_frames[0].shape[0] == v_frames[1].shape[0] == n_frame+1
        rgb_frames, mask_frames = v_frames[0], v_frames[1]  #B=2
        mask_frames = (mask_frames / 255.0).astype(bool)
        # reduce The mask_frames to 1 channel
        mask_frames = mask_frames[:, 0:1]

        rgbs_w_bbox_, retain_idxs_, bbox_coords = generate_bbox_imgs(
            rgb_frames, mask_frames,
        )
        rgbs_w_bbox_ = rgbs_w_bbox_[retain_idxs_]
        # Resize to (img_size, img_size) if needed
        # rgbs_w_bbox_ is a torch.Tensor of shape (B, C, H, W)
        rgbs_w_bbox_resized = F.interpolate(
            rgbs_w_bbox_,
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )

        rgbs_w_bbox.append(rgbs_w_bbox_resized[1:]) #because we insert a dummy frame at the beginning, so here we remove it
        retain_idxs.append(retain_idxs_)
    return deepcopy(rgbs_w_bbox), rgbs_w_bbox, retain_idxs


def post_process_output_aeqa_non_pano(output_dict, per_hfov, selected_idx, img_size: Tuple[int, int]=(384, 384)):
    rgbs_wo_mask = []
    h_pers=img_size[0]; w_pers=img_size[1]
    # assert h_pers == w_pers, "For non-pano AEQA, the img_size only supports square images now."

    for i in range(len(output_dict["save_dirs"])):
        # read the selected frame (1,C,H,W)
        # rgb_frame = ToTensor()(Image.open(select_img_path).convert("RGB")).unsqueeze(0)
        rgb_frame = output_dict["pred_frames"][i][selected_idx]     #pred_frames is [N, 14, C, H, W]
        rgb_frame = to_tensor(rgb_frame).permute((1, 2, 0)).unsqueeze(0)    # convert to (1, C, H, W)

        # 2) GET PERSPECTIVE VIEWS
        rgb_frame_resized = F.interpolate(
            rgb_frame,
            size=(h_pers, w_pers),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(0)
        # shape => [1, 1, 3, Hpers, Wpers]
        rgbs_wo_mask.append(rgb_frame_resized)

    return rgbs_wo_mask

def post_process_output_aeqa(output_dict, per_hfov, selected_idx, img_size: Tuple[int, int]=(384, 384)):
    """
    Return:
        rgbs_wo_mask: List[Tensor[N, 1, 3, Hpers, Wpers]] containing perspective
                      views of the selected RGB frames without masks.
    """
    rgbs_wo_mask = []

    for i in range(len(output_dict["save_dirs"])):
        # read the selected frame (1,C,H,W)
        rgb_frame = output_dict["pred_frames"][i][selected_idx]     #pred_frames is [B, 14, C, H, W]
        rgb_frame = to_tensor(rgb_frame).permute((1, 2, 0)).unsqueeze(0)    # convert to (1, C, H, W)

        # 2) GET PERSPECTIVE VIEWS FROM THE PANORAMA
        perspective_wo_mask = get_perspective_views(
            per_hfov, rgb_frame, h_pers=img_size[0], w_pers=img_size[1]
        )
        # shape => [4, 14, 3, Hpers, Wpers]

        rgbs_wo_mask.append(perspective_wo_mask)

    return rgbs_wo_mask


def post_process_output_ignav_non_pano(
    output_dict,
    per_hfov: float,          # kept for API symmetry; not used here
    start_idx: int,
    img_size: Tuple[int, int]=(384, 384)
) -> List[torch.Tensor]:
    """
    Select non-panoramic RGB frames, resize them, and return them in a shape that
    matches the panoramic front-view output of

    Parameters:
    save_dirs : Sequence[str]
        One or more folders containing JPEG frames named ``{frame_id}.jpg``.
    per_hfov : float
        Unused for non-panoramic data but retained for interface consistency.
    start_idx : int
        First frame index to keep; if absent, the newest frame is used.
    img_size : int, optional
        Output resolution (square).  Default: 384.

    Returns:
    List[torch.Tensor]
        Each entry has shape ``[1, 1, 3, img_size, img_size]`` so that downstream
        code can treat it analogously to a single front view with one strip.
    """
    rgbs_wo_mask: List[torch.Tensor] = []
    # assert img_size[0] == img_size[1], "For non-pano ignav, the img_size only supports square images now."

    for dir_path, pred_frames in zip(output_dict["save_dirs"], output_dict["pred_frames"]):
        # 1) Decide which frame(s) to keep for this directory.
        assert start_idx < len(pred_frames), f"Invalid start_idx {start_idx} for directory {dir_path}"
        kept_frames = pred_frames[start_idx:]  # keep all frames from start_idx to the end

        # 2) Load and stack the kept frames.
        rgb_frames = [to_tensor(f).permute((1, 2, 0)) for f in kept_frames]  # list of [3,H,W]
        rgb_frames = torch.stack(rgb_frames, dim=0)           # [N,3,H,W]
        rgb_frames = F.interpolate(                           # resize
            rgb_frames,
            size=(img_size[0], img_size[1]),  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        front_view = rgb_frames   # [N,3,H,W]
        rgbs_wo_mask.append(front_view)

    return rgbs_wo_mask

def post_process_output_ignav(output_dict, per_hfov, start_idx, img_size: Tuple[int, int]=(384, 384)):
    rgbs_wo_mask = []

    for dir_path, pred_frames in zip(output_dict["save_dirs"], output_dict["pred_frames"]):
        # 1) Decide which frame(s) to keep for this directory.
        assert start_idx < len(pred_frames), f"Invalid start_idx {start_idx} for directory {dir_path}"
        kept_frames = pred_frames[start_idx:]  # keep all frames from start_idx to the end

        # read the selected frame (1,C,H,W)
        rgb_frames = [to_tensor(f).permute((1, 2, 0)) for f in kept_frames]  # list of [3,H,W]
        rgb_frames = torch.stack(rgb_frames, dim=0)  # shape [N, C, H, W]

        persp_views = get_perspective_views(
            per_hfov, rgb_frames, h_pers=img_size[0], w_pers=img_size[1]
        )
        # shape => [4(view_num), 14, 3, Hpers, Wpers]
        rgbs_wo_mask.append(persp_views[0]) # shape [N, 3, Hpers, Wpers]

    return rgbs_wo_mask


def post_process_output_aeqa_(output_dict, bbox_obj_ids, per_hfov, selected_idx=None):
    rgbs_wo_mask = []
    rgbs_w_mask = []

    for i in range(len(output_dict["out_paths"])):
        # 0) READ VIDEO FRAMES
        # v_frames shape: [NumObj+1, 14, C, H, W]
        v_frames = read_video_asframes(output_dict["out_paths"][i])

        rgb_frames = v_frames[0]  # shape [14, C, H, W], C=3
        mask_frames = np.stack(v_frames[1:], axis=0)  # shape [NumObj, 14, C, H, W]
        # reduce to 1 channel
        mask_frames = mask_frames[:, :, 0]  # shape [NumObj, 14, H, W]
        if selected_idx:
            if rgb_frames.shape[0] <= selected_idx:
                selected_idx = rgb_frames.shape[0] - 1
            mask_frames = mask_frames[:, selected_idx: selected_idx + 1]
            rgb_frames = rgb_frames[selected_idx: selected_idx + 1]
        mask_frames = (mask_frames / 255.0).astype(bool)

        # 1) PREPARE PANORAMA TENSORS
        # We'll store the raw panorama as float [14, 3, H, W]:
        pano_wo_mask = torch.from_numpy(rgb_frames / 255.0).float()
        pano_mask_tensor = torch.from_numpy(mask_frames).float()
        # shape would be [NumObj, 14, H, W]

        # 2) GET PERSPECTIVE VIEWS FROM THE PANORAMA
        perspective_wo_mask = get_perspective_views(per_hfov, pano_wo_mask)
        # shape => [4, 14, 3, Hpers, Wpers]
        pano_mask_tensor_ = pano_mask_tensor.reshape(
            -1, 1, pano_mask_tensor.shape[-2], pano_mask_tensor.shape[-1]
        )
        per_H, per_W = perspective_wo_mask.shape[-2], perspective_wo_mask.shape[-1]
        perspective_masks = get_perspective_views(per_hfov, pano_mask_tensor_)
        perspective_masks = perspective_masks.reshape(
            4, *pano_mask_tensor.shape[:-2], per_H, per_W
        )
        perspective_masks = perspective_masks.bool()
        # => shape [4, NumObj, 14, Hpers, Wpers] if you adapt the function

        # We'll create an annotated clone of perspective_wo_mask:
        perspective_w_mask = perspective_wo_mask.clone()

        # 3) ANNOTATE EACH PERSPECTIVE FRAME WITH MASK/BOX OVERLAYS
        # For each frame f_idx and each perspective view view_idx,
        # we can convert to PIL, call "annotate_frame_masks," and write back.
        for view_idx in range(perspective_w_mask.shape[0]):     # 4 views
            for f_idx in range(perspective_w_mask.shape[1]):   # 14 frames
                # Convert to PIL
                pil_img = to_pil_image(perspective_w_mask[view_idx, f_idx])
                annotated_img = annotate_frame_masks(
                    pil_img,
                    masks=perspective_masks[view_idx, :, f_idx],
                    obj_idxs=bbox_obj_ids[i],
                    font_size=16,
                    contour_thickness=0,
                )
                perspective_w_mask[view_idx, f_idx] = to_tensor(annotated_img)

        rgbs_wo_mask.append(perspective_wo_mask)
        rgbs_w_mask.append(perspective_w_mask)

    return rgbs_w_mask, rgbs_wo_mask


def annotate_perspective_views(perspective_wo_mask):
    """
    Annotate the panoramic views with the current view information.
    Input:
        perspective_wo_mask: Tensor of shape [4, C, H, W]
    Return:
        output: Tensor of shape [4, C, H, W]
    """
    output = torch.empty((perspective_wo_mask.shape), dtype=torch.float32)
    assert perspective_wo_mask.dim() == 4, "Input tensor must have 4 dimensions"

    for view_idx in range(perspective_wo_mask.shape[0]):
        pano_image = to_pil_image(
            perspective_wo_mask[view_idx]
        )
        anno_image = annotate_frame(
            pano_image,
            f"Current View: <{VIEW_ORDER[view_idx]}>",
        )
        pano_tensor = to_tensor(anno_image)
        output[view_idx] = pano_tensor

    # i = 1
    # save_image(output[i], f"pano_wo_mask_view{i}.png")
    return output


def get_perspective_views(per_hfov, pano_w_mask, h_pers, w_pers):
    """
    Convert a panorama image to perspective views.
    Args:
        per_hfov (float): Horizontal field of view for the perspective views.
        pano_w_mask (Tensor): Panorama image tensor of shape [14, 3, H, W].
    Returns:
        perspective_views (Tensor): Perspective views tensor of shape [4, 14, 3, H, W].
    """
    # w_per = int(per_hfov / 90 * h_per)
    perspective_views = []
    for k, view_name in enumerate(VIEW_ORDER):
        degree = -VIEW_ID_OFFSET[k] / np.pi * 180
        rot_tensors = rotate_by_degrees(pano_w_mask, degree)
        perspective_view = convert_equi2per(
            rot_tensors, w_pers=w_pers, h_pers=h_pers, fov_x=per_hfov
        )
        perspective_views.append(perspective_view)
    perspective_views = torch.stack(perspective_views, dim=0)

    return perspective_views


def compute_horizontal_rotation(masks: np.ndarray, img_hfov=360) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the horizontal rotation angle and pixel shift required to center the
    object (mask) in a panorama image for a batch of masks.
    For each mask in the input batch (shape: (B, C, H, W)), the function computes
    a bounding box from the nonzero pixels (using the first channel if multiple exist),
    then calculates:
        - bbox_center = (x_min + x_max) / 2.0
        - pixel_shift = image_center - bbox_center, where image_center = image_width / 2.0
        - computed_angle = (pixel_shift / image_width) * 360.0
    If a mask is empty (no nonzero pixels), the corresponding computed_angle and pixel_shift
    are set to np.nan.
    Args:
        masks (np.ndarray): Batch of binary masks with shape (B, C, H, W) of type uint8.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays each of shape (B,):
            - computed_angles: The rotation angles in degrees for each mask.
            - pixel_shifts: The pixel shifts required for each mask.
    """
    B, C, H, W = masks.shape
    computed_angles = np.empty(B, dtype=float)
    pixel_shifts = np.empty(B, dtype=float)

    # The horizontal center of the image (panorama)
    image_center = W / 2.0

    for i in range(B):
        # Use the first channel if more than one exists.
        mask = masks[i, 0, :, :]
        # Find the nonzero (foreground) pixel coordinates.
        ys, xs = np.nonzero(mask)

        if xs.size == 0:
            computed_angles[i] = np.nan
            pixel_shifts[i] = np.nan
        else:
            x_min = xs.min()
            x_max = xs.max()
            bbox_center = (x_min + x_max) / 2.0
            # Positive pixel_shift means the bbox is to the left of center.
            shift = image_center - bbox_center
            angle = (shift / W) * img_hfov

            computed_angles[i] = angle
            pixel_shifts[i] = shift

    return computed_angles, pixel_shifts


def mask_to_bbox(mask: np.ndarray, retain_range=(0.001, 0.50)) -> Optional[Dict[str, int]]:
    """
    Compute the bounding box from a binary mask.
    Args:
        mask (np.ndarray): Binary mask of shape (H, W), or (1, H, W).
        retain_range (tuple): Tuple (min_ratio, max_ratio) specifying the acceptable
                              range for the bounding box area relative to the image area.
                              If the computed ratio is outside this range, the function returns None.
    Returns:
        Optional[Dict[str, int]]: A dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'
                                  if a valid bounding box is found; otherwise, None.
    """
    result = None  # Default to None unless a valid bbox is computed
    # Ensure mask is 2D
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    # ys, xs = np.where(mask)
    # Find the nonzero (foreground) pixel coordinates.
    ys, xs = np.nonzero(mask)

    # Proceed only if the mask is not empty
    if xs.size != 0 and ys.size != 0:
        x_min, x_max = xs.min(), xs.max()       # y is height, x is width
        y_min, y_max = ys.min(), ys.max()

        # Calculate the area of the bounding box and the image
        bbox_area = (x_max - x_min) * (y_max - y_min)
        image_area = mask.shape[0] * mask.shape[1]
        area_ratio = bbox_area / image_area

        # Accept bounding box only if the area ratio falls within the specified range
        if retain_range[0] < area_ratio < retain_range[1]:
            result = {
                "x_min": int(x_min),
                "x_max": int(x_max),
                "y_min": int(y_min),
                "y_max": int(y_max),
            }

    return result


def bbox_to_mask(bbox: Dict[str, int], img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a bounding box to a binary mask.
    Args:
        bbox (Dict[str, int]): A dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'.
        img_shape (Tuple[int, int]): Shape of the image (height, width).
    Returns:
        np.ndarray: Binary mask of shape (H, W) with 1s inside the bounding box and 0s outside.
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask[bbox["y_min"]:bbox["y_max"], bbox["x_min"]:bbox["x_max"]] = 1
    return mask


def generate_bbox_imgs(
    rgb_imgs: list,  # or np.ndarray with shape (B, C, H, W)
    mask_imgs: list, # or np.ndarray with shape (B, 1, H, W)
    scale_factor: float = 1.0,
    bbox_color: tuple = (255, 0, 0),
    bbox_thickness: int = 2,
    bbox_retain_range: tuple = (0.000, 0.50),
    draw_bbox: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[Tensor]]:
    """
    For each (rgb_img, mask_img) pair, compute a bounding box from the mask,
    optionally rescale it according to 'scale_factor', and draw the bounding box
    on a copy of the original RGB image. Returns a list of images (BGR or RGB
    depending on how 'cv2' interprets your data).
    Args:
        rgb_imgs (list): A list of original RGB images as NumPy arrays of shape (H, W, 3).
        mask_imgs (list): A list of corresponding binary mask images as NumPy arrays of shape (H', W').
                          Typically, H'=H and W'=W if no downscaling was applied. Otherwise,
                          set 'scale_factor' accordingly.
        scale_factor (float, optional): If the mask was generated at a different scale
        bbox_color (tuple, optional): BGR color for the bounding box in OpenCV (default: (255,0,0)).
        bbox_thickness (int, optional): Thickness (in pixels) of the bounding box lines.
    Returns:
        list: A list of the same length as rgb_imgs, where each element is the
              original image with the bounding box drawn.
    """
    if len(rgb_imgs) != len(mask_imgs):
        raise ValueError("rgb_imgs and mask_imgs must be the same length.")
    if isinstance(rgb_imgs, torch.Tensor):
        rgb_imgs = rgb_imgs.cpu().numpy()
        mask_imgs = mask_imgs.cpu().numpy()

    output_imgs, img_idxs, bbox_coords = [], [], []

    for i, (img, mask) in enumerate(zip(rgb_imgs, mask_imgs)):
        # Ensure the image is in (H, W, C) format.
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)

        # Make a copy so we don't overwrite the original
        annotated_img = img.copy()

        # 1. Compute bounding box from mask
        bbox = mask_to_bbox(mask, retain_range=bbox_retain_range)
        if bbox is not None:
            x_min, y_min, x_max, y_max = (
                bbox["x_min"],
                bbox["y_min"],
                bbox["x_max"],
                bbox["y_max"],
            )

            # 2. Rescale if mask was smaller/larger than the original
            #    If scale_factor < 1.0, mask is smaller than original. So bounding box
            #    needs to be scaled up. If scale_factor > 1.0, bounding box is scaled down.
            if scale_factor != 1.0:
                x_min = int(x_min / scale_factor)
                x_max = int(x_max / scale_factor)
                y_min = int(y_min / scale_factor)
                y_max = int(y_max / scale_factor)

            # 3. Draw bounding box on annotated_img
            #    OpenCV uses (x, y) for coords
            if draw_bbox:
                cv2.rectangle(
                    annotated_img,
                    (x_min, y_min),
                    (x_max, y_max),
                    bbox_color,
                    bbox_thickness,
                )

            output_imgs.append(annotated_img)
            img_idxs.append(i)
        else:
            output_imgs.append(img)
        bbox_coords.append(bbox)

    # create retain_idxs which can be index the original rgb_imgs (IndexError: tensors used as indices must be long, int, byte or bool tensors)
    retain_idxs = torch.tensor(img_idxs, dtype=torch.long)
    output_imgs = np.stack(output_imgs)
    output_imgs = torch.from_numpy(output_imgs)
    output_imgs = torch.einsum("bhwc->bchw", output_imgs)
    return output_imgs, retain_idxs, bbox_coords


def generate_aligned_bbox_frames(
    rgb_frames: np.ndarray,
    mask_frames: np.ndarray,
    per_hfov: float,
    img_size: Tuple[int, int],
    draw_bbox: bool = True,
) -> Optional[torch.Tensor]:
    """
    1. Compute pixel_shifts (horizontal rotation).
    2. Discard frames with NaN shifts.
    3. Convert frames to torch tensors.
    4. Align them & draw bounding boxes.
    Input:
        rgb_frames (np.ndarray): Array of RGB frames with shape (N, C, H, W). N is the number of frames.
        mask_frames (np.ndarray): Array of binary masks with shape (N, 1, H, W).
        per_hfov (float): Horizontal field of view for cropped perspective views.
    Returns:
        rgbs_aligned_w_bbox (Tensor or None): The aligned frames (BCHW) with bounding boxes,
                                              or None if no frames remain after discard.
    """
    h_per, w_per = img_size            # e.g. (384, 512)

    # 1) horizontal rotation from masks
    angles, pixel_shifts = compute_horizontal_rotation(mask_frames)
    discard_idxs = np.where(np.isnan(pixel_shifts))[0]

    if len(discard_idxs) > 0:
        first_nan = discard_idxs[0]
        rgb_frames_filt = rgb_frames[:first_nan]
        mask_frames_filt = mask_frames[:first_nan]
        pixel_shifts = pixel_shifts[:first_nan]
    else:
        rgb_frames_filt = rgb_frames
        mask_frames_filt = mask_frames

    if len(rgb_frames_filt) == 0:
        return None, None

    rgb_frames_filt_t  = torch.from_numpy(rgb_frames_filt)
    mask_frames_filt_t = torch.from_numpy(mask_frames_filt).to(torch.uint8) * 255

    # 2) align + convert
    rgbs_front_aligned = _align_frames_and_convert(
        pixel_shifts, rgb_frames_filt_t,  per_hfov, h_per, w_per
    )
    masks_front_aligned = _align_frames_and_convert(
        pixel_shifts, mask_frames_filt_t, per_hfov, h_per, w_per
    )

    # 3) draw bboxes & keep retained frames
    rgbs_bbox, retain_idxs, bbox_coords = generate_bbox_imgs(
        rgbs_front_aligned, masks_front_aligned, draw_bbox=draw_bbox
    )
    rgbs_bbox = rgbs_bbox[retain_idxs]
    if len(rgbs_bbox) == 0:
        return None, None
    return rgbs_bbox, bbox_coords


def generate_bbox_cube_frames(
    rgb_frames: np.ndarray,
    mask_frames: np.ndarray,
    per_hfov: float,
    img_size,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    1. Convert the entire sequence of frames to perspective (cube) form.
    2. Draw bounding boxes, returning (rgbs_w_bbox, retain_idxs).
    NOTE: this is a no filtering version of the above function.
    """
    # Convert to torch
    rgb_frames_t = torch.from_numpy(rgb_frames)
    mask_frames_t = torch.from_numpy(mask_frames).to(torch.uint8) * 255

    # Convert equirect -> perspective for all frames
    h_per, w_per = img_size            # e.g. (384, 512)

    rgbs_front: UInt8[Tensor, "X C H W"] = convert_equi2per(
        rgb_frames_t, w_pers=w_per, h_pers=h_per, fov_x=per_hfov
    )
    masks_front: UInt8[Tensor, "X C H W"] = convert_equi2per(
        mask_frames_t, w_pers=w_per, h_pers=h_per, fov_x=per_hfov
    )

    # Generate bounding boxes & get retained frames
    rgbs_w_bbox, retain_idxs, bbox_coords = generate_bbox_imgs(
        rgbs_front, masks_front, bbox_retain_range=(0.0, 1.0)
    )

    return rgbs_w_bbox, retain_idxs


def _align_frames_and_convert(
    pixel_shifts: np.ndarray,
    frames: torch.Tensor,
    per_hfov: float,
    h_per: int,
    w_per: int,
) -> torch.Tensor:
    """
    Rotates each frame by its pixel shift and converts the equirectangular
    panorama to a perspective (cube-face) view.

    If w_per is None, it is computed as in the original code:
        w_per = int(per_hfov / 90 * h_per)
    """
    aligned = [rotate_by_shift(frames[j], pixel_shifts[j]) for j in range(len(frames))]
    aligned: UInt8[Tensor, "B C H W"] = torch.stack(aligned, dim=0)

    pers_frames: UInt8[Tensor, "B C H W"] = convert_equi2per(
        aligned, w_pers=w_per, h_pers=h_per, fov_x=per_hfov
    )
    return pers_frames

if __name__ == "__main__":

    output_dict = {
        "out_paths": [
            ["downstream/states/AR_05.01_Debug_genex_1/5ZKStnWn8Zo/E081/A000/igenex/PredA-0/gen_video.mp4", "downstream/states/AR_05.01_Debug_genex_1/5ZKStnWn8Zo/E081/A000/igenex/PredA-0/gen_video_mask_obj_1.mp4"],
        ]
    }
    out = post_process_output_ar(output_dict, 90, img_size=384)
    rgbs_aligned_w_bbox, rgbs_w_bbox, retain_idxs = out
    save_image(
        rgbs_aligned_w_bbox[0] / 255.0,
        # "PLOT/exported_frames/gen_video/igenex_aligned_bbox.png",
        "PLOT/exported_frames/gen_video/igenex_aligned_bbox-5ZKS.png",
        nrow=len(rgbs_aligned_w_bbox[0]),
        format="png",
    )

