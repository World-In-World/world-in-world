import glob
import json
import os

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def save_video_from_tensor(pixel_values_re, save_path, fps=3, action_ids=None):
    """
    Saves a video from a tensor of shape (T, C, H, W), optionally overlaying each frame with an action ID.
    Args:
        pixel_values_re (torch.Tensor): Tensor containing video frames (T, C, H, W).
        save_path (str): Path where the video will be saved.
        fps (int): Frames per second for the video.
        action_ids (torch.Tensor or None): Optional tensor of shape (T,) with integer action IDs to overlay on each frame.
    """
    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = imageio.get_writer(
        save_path, fps=fps,
        codec='libx264',
        ffmpeg_params=['-crf', '18']
    )

    # Check if there are any frames.
    if pixel_values_re.size(0) == 0:
        print('No Movement to Export.')
        writer.close()
        return

    # Move to CPU if necessary and convert to numpy array.
    if pixel_values_re.is_cuda:
        pixel_values_re = pixel_values_re.cpu()
    frames = pixel_values_re.detach().numpy()  # shape: (T, C, H, W)

    # If action_ids is provided, ensure it's a numpy array and matches the frame count.
    if action_ids is not None:
        if isinstance(action_ids, torch.Tensor):
            if action_ids.is_cuda:
                action_ids = action_ids.cpu()
            action_ids = action_ids.detach().numpy()
        if len(action_ids) != frames.shape[0]:
            raise ValueError("Length of action_ids does not match number of frames.")

    # Process each frame.
    for i in range(frames.shape[0]):
        frame = frames[i]  # shape: (C, H, W)
        # Rearrange dimensions to (H, W, C) for imageio.
        frame = np.transpose(frame, (1, 2, 0))

        # Convert to uint8 if necessary.
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Convert to a PIL image to allow text overlay.
        pil_img = Image.fromarray(frame)

        # If action_ids are provided, overlay the corresponding action ID.
        if action_ids is not None:
            draw = ImageDraw.Draw(pil_img)
            # Choose the text and its position (e.g., (10, 10)).
            text = str(action_ids[i])
            draw.text((10, 10), text, fill=(255, 255, 255))

        # Convert back to numpy array and append to the video.
        frame_with_text = np.array(pil_img)
        writer.append_data(frame_with_text)

    writer.close()
    print(f'Video saved to: {save_path}')


def save_img_stitch(imgs, save_path, axis=0):
    """
    Stitches a list of images together along the specified axis and saves the result.

    Args:
        imgs (List[numpy.ndarray]): A list of images to stitch.
            - Expected dtype: Typically np.uint8 (or any dtype supported by imageio.imwrite).
            - Expected shape: Each image should be in HWC format (Height, Width, Channels) for color images,
              or (H, W) for grayscale images. All images must have compatible dimensions along axes not being concatenated.
        save_path (str): The file path where the stitched image will be saved.
        axis (int, optional): The axis along which to stitch the images.
            - Use 0 to stack vertically (one above the other).
            - Use 1 to stack horizontally (side by side).
            Default is 0.
    """
    if not imgs:
        raise ValueError("The list of images is empty.")

    # Ensure the output directory exists.
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Concatenate the list of images along the specified axis.
    stitched_img = np.concatenate(imgs, axis=axis)

    # Save the stitched image.
    imageio.imwrite(save_path, stitched_img)
    print(f'Stitched image saved to: {save_path}')



def save_img(img, save_path, mode=None, verbose=True, resize_shape=None):
    """
    Save an image (RGB, RGBA, depth, or semantic) to the specified path.
    Args:
        img (torch.Tensor or np.ndarray): The image data. Accepted formats:
            - RGB: (3, H, W) or (H, W, 3)
            - RGBA: (4, H, W) or (H, W, 4)
            - Depth: (1, H, W), (H, W, 1), or (H, W), float values.
            - Semantic: (H, W) or (H, W, 1), int32 or Uint32 values representing classes.
        save_path (str or Path): File path to save the image.
        resize_shape (tuple, optional): If provided, the image will be resized to this shape before saving.
    Returns:
        None
    """
    # Convert torch.Tensor to numpy array if needed
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # Adjust shape if necessary:
    if img.ndim == 3:
        if img.shape[0] in (3, 4):
            img = img.transpose(1, 2, 0)
        elif img.shape[0] == 1:
            img = np.squeeze(img, axis=0)
        elif img.shape[-1] == 1:
            img = np.squeeze(img, axis=-1)

    # Determine image mode based on dtype and shape
    if mode is None:
        if img.ndim == 2:
            # if dtype is int32 or Uint32, treat as semantic image, and convert to Uint32
            if img.dtype in [np.uint16, np.int32, np.uint32]:
                mode = "I"
            else:
                mode = "L"
        elif img.ndim == 3:
            if img.shape[2] == 3:
                mode = "RGB"
            elif img.shape[2] == 4:
                mode = "RGBA"

    # Normalize and convert to uint8 for RGB, RGBA, or depth
    if mode in ["RGB", "RGBA", "L"]:
        if img.dtype != np.uint8:
            if np.issubdtype(img.dtype, np.floating):
                if mode == "L" and img.max() > 1.0:
                    min_val, max_val = img.min(), img.max()
                    img = (img - min_val) / (max_val - min_val + 1e-8)
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
    elif mode == "I":
        if save_path.endswith('.png'):
            # make sure the values are in the range [0, 65535]
            assert img.max() <= 65535, f"Semantic image values exceed 65535: {max_val}"
            img = img.astype(np.uint16)
            mode = "I;16"  # Use the 16-bit mode for saving
        elif save_path.endswith('.tiff'):
            assert img.dtype == np.uint32
        else:
            raise ValueError(f"Unsupported file format for semantic image postfix: {save_path}")

    # Create and save PIL image
    if resize_shape is not None:
        pil_img = Image.fromarray(img, mode=mode).resize(resize_shape, Image.BILINEAR)
    else:
        pil_img = Image.fromarray(img, mode=mode)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_img.save(save_path)

    if verbose:
        print(f"===> Saved image to {save_path}")


def annotate_canvas_with_keys(canvas, face_order, img_width, img_height, grid_cols=3):
    """
    Annotate each face of the canvas with its corresponding key.
    Args:
        canvas (PIL.Image): The canvas image with cube faces.
        face_order (list): List of keys corresponding to each face.
        img_width (int): Width of a single face image.
        img_height (int): Height of a single face image.
        grid_cols (int): Number of columns in the grid.
    Returns:
        PIL.Image: The annotated canvas.
    """
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, key in enumerate(face_order):
        # Compute the top-left coordinate for the current face
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * img_width
        y = row * img_height

        # Set text position with a margin (e.g., 10 pixels from top-left)
        text_position = (x + 10, y + 10)

        # Optionally draw a background rectangle for contrast
        text_bbox = draw.textbbox(text_position, key, font=font)
        bg_rect = [text_bbox[0] - 5, text_bbox[1] - 5, text_bbox[2] + 5, text_bbox[3] + 5]
        # draw.rectangle(bg_rect, fill="black")
        # Draw the key text in white
        draw.text(text_position, key, font=font, fill="grey", size=25)

    return canvas


def stitch_cube_imgs(cube_dict, face_order=None):
    """
    Stitch cube face images into a 3x2 grid image and annotate each face with its key.

    The desired grid order is:
        Row 1: Back, Down, Front
        Row 2: Right, Left, Up

    This corresponds to the dictionary keys:
        ['B', 'D', 'F', 'R', 'L', 'U']

    Args:
        cube_dict (dict): Dictionary containing cube face images.
                          Expected keys are 'F', 'R', 'B', 'L', 'U', 'D'.
                          Each value is assumed to be an image in (3, H, W)
                          format (either a torch.Tensor or a numpy array).
        face_order (list, optional): Custom order of keys to use. Defaults to ['B', 'D', 'F', 'R', 'L', 'U'].

    Returns:
        PIL.Image: The stitched and annotated image in a 3x2 grid.
    """
    if face_order is None:
        face_order = ['Back', 'Down', 'Front', 'Right', 'Left', 'Up']

    # Convert each face to a PIL Image
    images = []
    for key in face_order:
        face_img = cube_dict[key]
        # Convert torch tensor to numpy if needed
        if isinstance(face_img, torch.Tensor):
            face_img = face_img.detach().cpu().numpy()
        # If image is in CHW format, convert it to HWC
        if face_img.ndim == 3 and face_img.shape[0] == 3:
            face_img = face_img.transpose(1, 2, 0)
        # NOTE If image dtype is not uint8, assume it's in [0,1] and convert
        if face_img.dtype != np.uint8:
            face_img = np.clip(face_img, 0, 1)
            face_img = (face_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(face_img)
        images.append(pil_img)

    # Assume all faces have the same dimensions
    img_width, img_height = images[0].size

    # Create a canvas for a 3x2 grid
    grid_cols, grid_rows = 3, 2
    canvas_width = grid_cols * img_width
    canvas_height = grid_rows * img_height
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))

    # Paste the images into the canvas following the order
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * img_width
        y = row * img_height
        canvas.paste(img, (x, y))

    # Annotate the canvas with each face's corresponding key
    canvas = annotate_canvas_with_keys(canvas, face_order, img_width, img_height, grid_cols)

    return canvas


def get_all_trajs_voidratios(json_files):
    all_trajs_voidratios = {}; num_skipped, num_broken = 0, 0

    for jfile in json_files:
        # 2. Skip empty or invalid JSON
        if os.stat(jfile).st_size == 0:
            print(f"Skipping empty or broken JSON file: {jfile}")
            num_broken += 1
            continue

        with open(jfile, 'r') as f:
            data = json.load(f)
        if "VoidRatio" not in data or not data["VoidRatio"]:
            print(f"Skipping {jfile}, no 'VoidRatio' found.")
            num_skipped += 1
            # del the json file if it is empty
            # os.remove(jfile)
            continue

        # 3. For each "StartStep-X", data["VoidRatio"][f"StartStep-{start_step}"] is
        #    presumably a list of floats. We'll get the mean.
        file_ratios_dict = {}
        for startstep_key, ratio_list in data["VoidRatio"].items():
            # Convert to float array, compute mean
            ratio_vals = np.array(ratio_list, dtype=float)
            mean_val = np.mean(ratio_vals)
            file_ratios_dict[startstep_key] = mean_val

        # We'll store these means keyed by the JSON's path (or maybe just by the parent folder).
        traj_key = os.path.dirname(jfile)
        all_trajs_voidratios[traj_key] = file_ratios_dict

    print(f"Warning: Skipped {num_skipped} JSON files with no 'VoidRatio'.")
    print(f"Warning: Skipped {num_broken} empty or broken JSON files.")
    return all_trajs_voidratios


def assign_sample_weights(all_trajs_voidratios, method='linear', cutoff=None, **kwargs):
    """
    Given a dictionary of trajectories where each key is a trajectory identifier (e.g., a folder path)
    and each value is a dictionary mapping "StartStep-X" to a void ratio (float), this function computes
    a global sample weight for each StartStep entry. Lower void ratios (better quality) will receive larger
    weights. The function can use one of several weighting strategies:

      - 'linear': Compute weights by linear inverse mapping:
           weight = (max_v - v) / (max_v - min_v)
      - 'exponential': Compute weights using an exponential decay:
           weight = exp(-alpha * v), with parameter alpha (default 1.0)
      - 'cutoff': Use a cutoff threshold (provided via cutoff); weight = (cutoff - v) if v < cutoff, else 0.

    You can also combine methods. For example, if a cutoff is provided, then entries with v >= cutoff
    will be forced to zero weight, and the chosen base method (e.g. linear or exponential) is applied only
    to the remaining entries.

    The final weights are normalized (global normalization) so that the sum over all entries is 1.

    :param all_trajs_voidratios: dict, where keys are trajectory identifiers (e.g., folder paths)
         and values are dicts mapping "StartStep-X" to void ratio values.
    :param method: str, one of 'linear', 'exponential', or 'cutoff' (if used alone).
                   Default is 'linear'. (When combining, supply cutoff parameter and choose a base method.)
    :param cutoff: float or None. If provided, any entry with void ratio >= cutoff will receive a weight of 0.
    :param kwargs: additional parameters for the chosen method; for example, for 'exponential', use alpha.
    :return: A dict with the same keys as all_trajs_voidratios; each value is a dict mapping "StartStep-X"
             to its computed normalized sample weight.
    """
    # Flatten the data: create a list of tuples: (traj_key, startstep_key, void_ratio)
    entries = []
    for traj_key, void_dict in all_trajs_voidratios.items():
        for step_key, v in void_dict.items():
            entries.append((traj_key, step_key, v))

    # Extract the void ratio values as a numpy array.
    values = np.array([entry[2] for entry in entries], dtype=float)  # values is the void ratio for each StartStep

    # Create a boolean mask indicating which entries pass the cutoff (if any).
    if cutoff is not None:
        valid_mask = values < cutoff
    else:
        valid_mask = np.ones_like(values, dtype=bool)

    valid_num = valid_mask.sum()
    print(f'Filtering out {len(values) - valid_num}/{len(values)} entries with void ratio >= {cutoff}')
    # 1 If cutoff is provided, force weights of non-valid entries to zero.
    valid_values = values[valid_mask]

    # 2. Norm the unmasked values to 0-1
    min_v = valid_values.min(); max_v = valid_values.max()
    norm_valid_values = (valid_values - min_v) / (max_v - min_v) # range from 0 to 1

    # Initialize raw_weights for all entries to zero.
    raw_weights = np.zeros_like(values)

    # Compute raw weights according to the chosen base method.
    if method == 'linear':
        intercept = kwargs.get('intercept', 2.0)  # Fixed the typo.
        slope = kwargs.get('slope', -1.0)
        # For valid entries, use the normalized values.
        raw_weights[valid_mask] = slope * norm_valid_values + intercept     # range from 1 to 2
    elif method == 'exponential':
        alpha = kwargs.get('alpha', 1.0)
        raw_weights[valid_mask] = np.exp(-alpha * norm_valid_values)       # range from e^0 to e^(-alpha)
    elif method == 'uniform':
        raw_weights = np.ones_like(values)
    elif method == 'uniform2':
        raw_weights[valid_mask] = 1

    # Normalize the weights so that they sum to 1.
    # total = raw_weights.sum()
    # norm_weights = raw_weights / total

    # Reassemble the results into the entries_weight with the same structure as entries.
    traj_entries = []
    for i, (traj_key, step_key, _) in enumerate(entries):
        traj_entries.append((traj_key, step_key))       #raw_weights[i]

    return traj_entries, raw_weights


# Example usage:
def glob_all_overlap_json(base_folder, n_frame):
    pattern = os.path.join(str(base_folder), "*", "traj-*", "waypoint-*", f"overlap_Nframe-{n_frame}_1.json")
    json_files = glob.glob(pattern)
    if not json_files:
        raise ValueError(f"No overlap_Nframe-{n_frame}.json files found in {base_folder}.")
    # sort the files for consistency
    json_files = sorted(json_files)
    return json_files


if __name__ == '__main__':
    # Sample input structure: keys are traj paths, values are dict mapping StartStep keys to void ratio
    # base_path = "/home/jchen293/scratchayuille1/jzhan423/igenex_code/data/datasets__/01.26_indoor_dataset-wp"
    base_path = "/home/jchen293/scratchayuille1/jzhan423/igenex_code/data/datasets__/03.13_mp3d-train-new"
    json_files = glob_all_overlap_json(base_path, 14)

    all_trajs_voidratios = get_all_trajs_voidratios(json_files)

    # Choose a method (for example, exponential with alpha=2.0)
    sample_weights = assign_sample_weights(
        all_trajs_voidratios,
        method='linear',
        cutoff=0.45,
        # alpha=2.0,
    )
    print(f'len of Assigned sample weights for each StartStep: {len(sample_weights)}')
