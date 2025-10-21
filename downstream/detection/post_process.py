import logging
import numpy as np
import supervision as sv
from downstream.downstream_datasets import BACKGROUND_CLASS
from utils.util import is_empty


def mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def filter_detections(
    image_HW,
    detections: sv.Detections,
    all_text_labels,
    top_x_detections=None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.90,        # IoU similarity threshold
    proximity_threshold: float = 20.0,  # Default proximity threshold
    keep_larger: bool = True,           # Keep the larger bounding box by area if True, else keep the smaller
    min_mask_size_ratio = 0.003,
    max_mask_size_ratio = 0.5,
    exclude_obj_names: list[str] = ['door', 'stairs', 'stair rail', 'doorway'],
    verbose: bool = False,
) -> tuple[sv.Detections, list[str]]:
    """
    Filter detections based on confidence, top X detections, and proximity of bounding boxes.
    Args:
        proximity_threshold (float): The minimum distance between centers of bounding boxes to consider them non-overlapping.
        keep_larger (bool): If True, keeps the larger bounding box when overlaps occur; otherwise keeps the smaller.
    Returns:
        tuple[sv.Detections]: Filtered detections
    """
    # Sort by confidence initially
    detections_combined = sorted(
        zip(
            detections.confidence,
            detections.class_id,
            detections.xyxy,
            detections.mask,
            range(len(detections)),
        ),
        key=lambda x: x[0],
        reverse=True,
    )

    if top_x_detections is not None:
        detections_combined = detections_combined[:top_x_detections]

    # Calculate the total number of pixels as a threshold for small masks
    total_pixels = image_HW[0] * image_HW[1]
    small_mask_size = total_pixels * min_mask_size_ratio
    large_mask_size = total_pixels * max_mask_size_ratio

    # Further filter based on proximity
    filtered_detections = []
    for idx, current_det in enumerate(detections_combined):
        conf, curr_class_id, curr_xyxy, curr_mask, _ = current_det
        curr_center = (
            (curr_xyxy[0] + curr_xyxy[2]) / 2,
            (curr_xyxy[1] + curr_xyxy[3]) / 2,
        )
        curr_area = (curr_xyxy[2] - curr_xyxy[0]) * (curr_xyxy[3] - curr_xyxy[1])
        keep = True

        # check mask size and remove if too small
        mask_size = np.count_nonzero(current_det[3])
        det_obj_name = all_text_labels[curr_class_id]
        if mask_size < small_mask_size and det_obj_name not in exclude_obj_names:
            print(f"Removing {det_obj_name} because the mask size is too small.") if verbose else None
            keep = False
            continue
        if mask_size > large_mask_size and det_obj_name not in exclude_obj_names:
            print(f"Removing {det_obj_name} because the mask size is too large.") if verbose else None
            keep = False
            continue

        # Check confidence threshold
        if conf < confidence_threshold and det_obj_name not in exclude_obj_names:
            print(f"Removing {det_obj_name} because it has a confidence of\t\t{conf:.3} < {confidence_threshold}.") if verbose else None
            keep = False
            continue

        remove_idxs = []
        for i, other in enumerate(filtered_detections):
            _, other_class_id, other_xyxy, other_mask, _ = other

            if mask_iou(curr_mask, other_mask) > iou_threshold and det_obj_name not in exclude_obj_names:
                keep = False
                print(f"Removing {det_obj_name}"
                      f"because IoU: {mask_iou(curr_mask, other_mask):.3} with object {all_text_labels[other_class_id]}.")  if verbose else None
                break

            other_center = (
                (other_xyxy[0] + other_xyxy[2]) / 2,
                (other_xyxy[1] + other_xyxy[3]) / 2,
            )
            other_area = (other_xyxy[2] - other_xyxy[0]) * (
                other_xyxy[3] - other_xyxy[1]
            )

            # Calculate distance between centers
            dist = np.sqrt(
                (curr_center[0] - other_center[0]) ** 2
                + (curr_center[1] - other_center[1]) ** 2
            )
            if dist < proximity_threshold:
                if (keep_larger and curr_area > other_area) or (
                    not keep_larger and curr_area < other_area
                ):
                    remove_idxs.append(i)
                else:
                    keep = False
                    break

        filtered_detections = [
            filtered_detections[i] for i in range(len(filtered_detections))
            if i not in remove_idxs
        ]
        # print(given_labels[idx])
        if det_obj_name in BACKGROUND_CLASS:
            print(f"Removing {det_obj_name} "
                  f"because it is a background class, specifically {BACKGROUND_CLASS}.") if verbose else None
            keep = False
            continue

        if keep: filtered_detections.append(current_det)

    if is_empty(filtered_detections):
        # create a dummy empty detections object, with input: 2D np.ndarray shape (0, 4),
        empty_det = sv.Detections(np.array([], dtype=np.float32).reshape(0, 4))
        empty_det.text_labels = []
        return empty_det

    # Unzip the filtered results
    confidences, class_ids, xyxy, masks, indices = zip(*filtered_detections)

    # Create new detections object
    filtered_detections = sv.Detections(
        class_id=np.array(class_ids, dtype=np.int64),
        confidence=np.array(confidences, dtype=np.float32),
        xyxy=np.array(xyxy, dtype=np.float32),
        mask=np.array(masks, dtype=np.bool_),
    )
    filtered_detections.text_labels = [
        all_text_labels[class_id] for class_id in filtered_detections.class_id
    ]

    return filtered_detections



def mask_subtract_contained(xyxy: np.ndarray, mask: np.ndarray, th1=0.8, th2=0.7):
    """
    Compute the containing relationship between all pair of bounding boxes.
    For each mask, subtract the mask of bounding boxes that are contained by it.

    Args:
        xyxy: (N, 4), in (x1, y1, x2, y2) format
        mask: (N, H, W), binary mask
        th1: float, threshold for computing intersection over box1
        th2: float, threshold for computing intersection over box2

    Returns:
        mask_sub: (N, H, W), binary mask
    """
    N = xyxy.shape[0]  # number of boxes

    # Get areas of each xyxy
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])  # (N,)

    # Compute intersection boxes
    lt = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])  # left-top points (N, N, 2)
    rb = np.minimum(
        xyxy[:, None, 2:], xyxy[None, :, 2:]
    )  # right-bottom points (N, N, 2)

    inter = (rb - lt).clip(
        min=0
    )  # intersection sizes (dx, dy), if no overlap, clamp to zero (N, N, 2)

    # Compute areas of intersection boxes
    inter_areas = inter[:, :, 0] * inter[:, :, 1]  # (N, N)

    inter_over_box1 = inter_areas / areas[:, None]  # (N, N)
    # inter_over_box2 = inter_areas / areas[None, :] # (N, N)
    inter_over_box2 = inter_over_box1.T  # (N, N)

    # if the intersection area is smaller than th2 of the area of box1,
    # and the intersection area is larger than th1 of the area of box2,
    # then box2 is considered contained by box1
    contained = (inter_over_box1 < th2) & (inter_over_box2 > th1)  # (N, N)
    contained_idx = contained.nonzero()  # (num_contained, 2)

    mask_sub = mask.copy()  # (N, H, W)
    # mask_sub[contained_idx[0]] = mask_sub[contained_idx[0]] & (~mask_sub[contained_idx[1]])
    for i in range(len(contained_idx[0])):
        mask_sub[contained_idx[0][i]] = mask_sub[contained_idx[0][i]] & (
            ~mask_sub[contained_idx[1][i]]
        )

    return mask_sub
