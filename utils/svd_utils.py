import torch
import torch.nn.functional as F
import numpy as np

from typing import Union, Optional, List, Dict, Tuple, Any
from jaxtyping import Float
from torchvision import transforms
import PIL
from scipy.spatial.transform import Rotation


# =======Added implementation:=======
# Mapping from angle to denominator for computing shift (except for 0°).
# agent turn left <==> rotate 22.5 for pano
SCENE_BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
ANGLE_TO_DENOM = {
    22.5: 16,  -22.5: 16,
    45: 8,    -45: 8,
    90: 4,    -90: 4,
    180: 2,   -180: 2,
    -67.5: "16/3", 67.5: "16/3",
    -112.5: "16/5", 112.5: "16/5",
}


def images_to_tensor(video_frames: list[PIL.Image.Image]):
    transform = transforms.ToTensor()
    video_tensors = []
    for clip in video_frames:
        tensor_frames = [transform(frame) for frame in clip]
        video_tensor = torch.stack(tensor_frames)  # Shape: [N, C, H, W]
        video_tensors.append(video_tensor)

    # Verification
    assert video_tensor.max() <= 1.0 and video_tensor.min() >= 0.0
    return torch.stack(video_tensors)  # Shape: [b, N, C, H, W]


def sample_latent_noise(actions: torch.Tensor,
                        shape: torch.Size,
                        device,
                        dtype,
                        task_type,
                        generator=None) -> torch.Tensor:
    """
    Samples a latent noise tensor and applies rotations based on the given actions.
    Args:
        actions (torch.Tensor, Float[Tensor, 'B F']): Tensor of shape [B, F] containing action ids.
        shape (torch.Size): Expected shape of the noise tensor: [B, F, 4, 32, 64].
        device: The device on which to create the tensor.
        dtype: The data type for the noise tensor.
    Returns:
        torch.Tensor: A noise tensor of shape [B, F, 4, 32, 64] with modifications based on actions.
    """
    from diffusers.utils.torch_utils import randn_tensor
    noise_all = randn_tensor(shape, generator, device, dtype)
    if actions.dim() == 2:
        B, F = actions.shape  # B: batch size, F: number of frames
        a_dim = 1
    elif actions.dim() == 3:
        B, F, a_dim = actions.shape

    #  # only do the processing for nav_actions which are 1D
    if task_type == 'navigation':
        assert a_dim == 1, f"Expected actions to be 1D for navigation, got {a_dim}D"
        for b in range(B):
            # For each frame (starting at 1 because the first frame remains unchanged)
            for i in range(1, F):
                # Get the current action for sample b at frame i (scalar).
                curr_action = actions[b, i].item()

                # If action == 2: turn left.
                if curr_action == 2:
                    # Copy the previous frame's noise and apply a left rotation.
                    noise_tfm = noise_all[b, i-1].clone()
                    noise_all[b, i] = rotate_by_degrees(noise_tfm, 22.5)
                # If action == 3: turn right.
                elif curr_action == 3:
                    # Copy the previous frame's noise and apply a right rotation.
                    noise_tfm = noise_all[b, i-1].clone()
                    noise_all[b, i] = rotate_by_degrees(noise_tfm, -22.5)

    return noise_all


def rotate_by_degrees(input_z: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotates the input tensor by performing a cyclic horizontal shift along its width dimension.
      - 0 or 360°:        no shift.
      - ±22.5°:   shift = ±(width / 16)
      - ±45°:     shift = ±(width / 8)
      - ±90°:     shift = ±(width / 4)
      - ±180°:     shift = ±(width / 2)
    Args:
        input_z (torch.Tensor): Input tensor with shape [B, F, C, H, W] (or similar) where the last dimension is width.
        angle (float): Rotation angle in degrees.
    Returns:
        torch.Tensor: The rotated tensor.
    """
    width = input_z.shape[-1]

    if angle == 0 or angle == 360:
        return input_z  # no shift needed
    elif angle not in ANGLE_TO_DENOM:
        raise ValueError("Unsupported rotation angle. Supported values: 0, ±22.5, ±45, ±90, 180, 360.")
    else:
        shift = get_rotate_shift(angle, width)

    return torch.roll(input_z, shifts=shift, dims=-1)

def get_rotate_shift(angle, width):
    denom = ANGLE_TO_DENOM[angle]
    if isinstance(denom, str):
        # Assume the string is in the format "numerator/denom" (e.g., "16/3")
        num, den = map(int, denom.split("/"))
        if (width * den) % num != 0:
            raise AssertionError(f"Width must be divisible by {denom} for {angle}-degree rotation.")
        shift = (width * den) // num
    else:
        if width % denom != 0:
            raise AssertionError(f"Width must be divisible by {denom} for {angle}-degree rotation.")
        shift = width // denom

    if angle < 0:
        shift = -shift
    return shift


def rotate_by_shift(input_z: torch.Tensor, shift: int) -> torch.Tensor:
    """
    Rotates the input tensor by performing a cyclic horizontal shift along its width dimension.
    Args:
        input_z (torch.Tensor): Input tensor with shape [B, F, C, H, W] (or similar) where the last dimension is width.
        shift (int): The number of pixels to shift the tensor horizontally.
    Returns:
        torch.Tensor: The rotated tensor.
    """
    if not isinstance(shift, int):
        shift = int(shift)
    if not isinstance(input_z, torch.Tensor):
        input_z = torch.tensor(input_z)
    width = input_z.shape[-1]
    return torch.roll(input_z, shifts=shift, dims=-1)


def rotate_coord_by_degrees(bbox_coord: Dict[str, int], angle: float, img_width: int) -> Dict[str, int]:
    """
    Rotates the bounding box coordinates by applying a cyclic horizontal shift.
    Only the x coordinates ("x_min" and "x_max") are shifted; the y coordinates remain unchanged.
    Args:
        bbox_coord (Dict[str, int]): Dictionary with keys "x_min", "x_max", "y_min", "y_max".
        angle (float): Rotation angle in degrees.
        image_width (int): The width of the image (used for modulo arithmetic).
    Returns:
        Dict[str, int]: The rotated bounding box coordinates.
    """
    if angle == 0 or angle == 360:
        shift = 0
    elif angle not in ANGLE_TO_DENOM:
        raise ValueError("Unsupported rotation angle. Supported values: 0, ±22.5, ±45, ±90, 180, 360.")
    else:
        shift = get_rotate_shift(angle, img_width)

    # Apply cyclic shift on the x coordinates.
    new_x_min = (bbox_coord["x_min"] + shift) % img_width
    new_x_max = (bbox_coord["x_max"] + shift) % img_width

    return {
        "x_min": new_x_min,
        "x_max": new_x_max,
        "y_min": bbox_coord["y_min"],
        "y_max": bbox_coord["y_max"],
    }


def apply_conditioning_dropout(
    encoder_hidden_states: torch.Tensor,
    conditional_latents: torch.Tensor,
    action_conditioning: torch.Tensor,
    bsz: int,
    conditioning_dropout_prob: Optional[float],
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply classifier-free conditioning dropout for text (prompt), image, and action conditioning.
      1. **Text / Prompt Conditioning:**
         If a random value is less than 2 * conditioning_dropout_prob, the text conditioning is dropped.
      2. **Image Conditioning:**
         If the random value lies in the range [conditioning_dropout_prob, 3 * conditioning_dropout_prob),
         the image conditioning is dropped.
      3. **Action Conditioning:**
         If the random value lies in the range [3 * conditioning_dropout_prob, 4 * conditioning_dropout_prob),
         the action conditioning is dropped.
    Args:
        encoder_hidden_states (torch.Tensor): Text encoder outputs.
            (e.g., shape `[bsz, seq_len, hidden_dim]`)
        conditional_latents (torch.Tensor): Image conditioning latent representations.
            (e.g., shape `[bsz, channels, height, width]`)
        action_conditioning (torch.Tensor): Action conditioning tensor.
            (e.g., shape `[bsz, feature_dim]` or similar)
        bsz (int): Batch size.
        conditioning_dropout_prob (Optional[float]): The base probability for conditioning dropout.
            If None, no dropout is applied.
        generator (Optional[torch.Generator]): Optional random number generator for reproducibility.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple of (encoder_hidden_states, conditional_latents, action_conditioning) after dropout.
    """
    if conditioning_dropout_prob is not None:
        # Use the device from one of the conditioning tensors.
        device = encoder_hidden_states.device
        # Sample a random probability for each batch element.
        random_p = torch.rand(bsz, device=device, generator=generator)

        # --- 1) TEXT / PROMPT CONDITIONING ---
        # Create a mask where values < 2*dropout_prob will zero out the text conditioning.
        prompt_mask = random_p < 2 * conditioning_dropout_prob
        # Reshape to broadcast along sequence and feature dimensions.
        prompt_mask = prompt_mask.reshape(bsz, 1, 1)
        # Create a "null" tensor of the same shape as encoder_hidden_states.
        null_conditioning = torch.zeros_like(encoder_hidden_states)
        # For entries where prompt_mask is True, replace with null conditioning.
        encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

        # --- 2) IMAGE CONDITIONING ---
        #Creates a mask where elements are 1 if the corresponding element in random_p is outside the range [args.conditioning_dropout_prob, 3 * args.conditioning_dropout_prob), and 0 otherwise.
        # If random_p is in [dropout_prob, 3*dropout_prob), we drop the image conditioning.
        image_mask_dtype = conditional_latents.dtype
        image_mask = 1.0 - (
            (random_p >= conditioning_dropout_prob).to(image_mask_dtype)
            * (random_p < 3 * conditioning_dropout_prob).to(image_mask_dtype)
        )
        # Reshape to match conditional_latents, e.g., [bsz, 1, 1, 1] for broadcasting.
        image_mask = image_mask.reshape(bsz, 1, 1, 1)
        conditional_latents = image_mask * conditional_latents

        # --- 3) ACTION CONDITIONING (New Condition) ---
        # pass

    return encoder_hidden_states, conditional_latents, action_conditioning


def apply_discrete_conditioning_dropout(
    encoder_hidden_states: torch.Tensor,   # B
    conditional_latents: torch.Tensor,     # C
    action_conditioning: torch.Tensor,     # A
    bsz: int,
    generator: torch.Generator = None,
    only_action_dropout: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Explicitly controls the probability of dropping out subsets of A, B, C (action, text, image).
    Each item in the batch is mapped to exactly one scenario based on a random p in [0,1):
    Args:
        encoder_hidden_states: Text/prompt conditioning (B).
        conditional_latents:   Image conditioning (C).
        action_conditioning:   Action conditioning (A).
        bsz (int):             Batch size.
        generator (torch.Generator): Optional random number generator for reproducibility.
    Returns:
        A tuple of (encoder_hidden_states, conditional_latents, action_conditioning) after dropout.
    """
    if only_action_dropout:
        # action_conditioning[mask_dropABC] = -1
        action_conditioning[:] = -1
        return None, None, action_conditioning

    device = encoder_hidden_states.device
    # 1) Sample random probabilities in [0,1) for each example in the batch
    random_p = torch.rand(bsz, device=device, generator=generator)

    # 2) Define boolean masks for each scenario
    mask_dropA_only   = (random_p >= 0.0) & (random_p < 0.1)
    mask_dropB_only   = (random_p >= 0.1) & (random_p < 0.2)
    mask_dropC_only   = (random_p >= 0.2) & (random_p < 0.3)
    mask_dropBC       = (random_p >= 0.3) & (random_p < 0.4)
    mask_dropAB       = (random_p >= 0.4) & (random_p < 0.5)
    mask_dropAC       = (random_p >= 0.5) & (random_p < 0.6)
    mask_dropABC      = (random_p >= 0.6) & (random_p < 0.7)
    mask_keep_all     = (random_p >= 0.7) & (random_p < 1.0)

    # 3) For each scenario, zero out the relevant conditioning
    batch_indices = torch.arange(bsz, device=device)

    # Scenario 1: drop only A (action_conditioning)
    if mask_dropA_only.any():
        action_conditioning[mask_dropA_only] = -1

    # Scenario 2: drop only B (encoder_hidden_states)
    if mask_dropB_only.any():
        encoder_hidden_states[mask_dropB_only] = 0.0

    # Scenario 3: drop only C (conditional_latents)
    if mask_dropC_only.any():
        conditional_latents[mask_dropC_only] = 0.0

    # Scenario 4: drop B + C
    if mask_dropBC.any():
        encoder_hidden_states[mask_dropBC] = 0.0
        conditional_latents[mask_dropBC] = 0.0

    # Scenario 5: drop A + B
    if mask_dropAB.any():
        action_conditioning[mask_dropAB] = -1
        encoder_hidden_states[mask_dropAB] = 0.0

    # Scenario 6: drop A + C
    if mask_dropAC.any():
        action_conditioning[mask_dropAC] = -1
        conditional_latents[mask_dropAC] = 0.0

    # Scenario 7: drop A + B + C
    if mask_dropABC.any():
        action_conditioning[mask_dropABC] = -1
        encoder_hidden_states[mask_dropABC] = 0.0
        conditional_latents[mask_dropABC] = 0.0

    # Scenario 8: keep everything (mask_keep_all)
    # -> Do nothing, because we keep A, B, C intact.
    return encoder_hidden_states.unsqueeze(0), conditional_latents, action_conditioning


# ============== Added implementation For Manipulation ==============
def check_xyz_bounds(xyz_iter, scene_bounds=SCENE_BOUNDS) -> np.ndarray:
    """
    Verify that all xyz coordinates are inside scene_bounds.

    xyz_iter : array-like or iterable of shape (..., 3)
        A single (N, 3) NumPy array **or** any iterable that yields
        (3,) position vectors.
    scene_bounds : array-like (6,), default = SCENE_BOUNDS
        (xmin, ymin, zmin, xmax, ymax, zmax).

    bound_min : ndarray (3,), bound_max : ndarray (3,)
    """
    xyz_arr = np.asarray(xyz_iter, dtype=float).reshape(-1, 3)

    # mask of shape (N,) – True where any coordinate is OOB for that row
    oob_mask = (
        (xyz_arr < scene_bounds[:3]) |
        (xyz_arr > scene_bounds[3:])
    ).any(axis=1)

    bad_idx = np.array([], dtype=int)  # default empty array
    if oob_mask.any():
        bad_idx = np.where(oob_mask)[0]          # row indices of offenders
        # print(f"Out-of-bounds positions at indices: {bad_idx.tolist()}")
        # raise ValueError(
        #     f"XYZ coordinates out of bounds at indices {bad_idx.tolist()}; "
        #     f"min {bound_min}, max {bound_max}; "
        #     f"expected within {scene_bounds[:3]} .. {scene_bounds[3:]}"
        # )
    return bad_idx

def quaternion_to_rotmatrix(quat):
    """
    Convert a unit quaternion to its 6-D rotation-matrix representation (R6).

    Parameters
    quat : array-like (..., 4)
        Quaternion(s) in (x, y, z, w) order, assumed to be unit length.

    Returns
    r6 : ndarray (..., 6)
        Concatenation of the first two rows of the 3×3 rotation matrix:
        [R00, R01, R02, R10, R11, R12].
        Range of the r6 values is [-1, 1], with the first two rows
    """
    # quat = np.asarray(quat, dtype=float)
    rot  = Rotation.from_quat(quat)
    R    = rot.as_matrix()                      # (..., 3, 3)
    # r6 = R[..., :, :2].reshape(R.shape[:-2] + (6,))
    return R

def get_action_from_continuous(continuous_action: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact inverse of `get_continous_action_from_discrete` – now batched.

    * Parameters
    continuous_action : array-like, (B, 8)
        [..., x, y, z, qx, qy, qz, qw, gripper_open]

    * Returns
    xyz         : ndarray, shape (..., 3)
    rotmatrix   : ndarray, shape (..., 3, 3)
    grip        : ndarray, shape (...)         (float(s) 0 → closed, 1 → open)
    """
    continuous_action = continuous_action.cpu().numpy()

    # ---- 1. Accept both single and batched inputs ------------------------------------
    single = continuous_action.ndim == 1
    if single:
        continuous_action = continuous_action[None, :]     # add batch dim B=1

    if continuous_action.shape[-1] != 8:
        raise ValueError("continuous_action must have final dimension 8")

    # ---- 2. Split fields -------------------------------------------------------------
    xyz   = continuous_action[..., :3]          # (B, 3)
    quats = continuous_action[..., 3:7]         # (B, 4)
    grip  = continuous_action[..., 7]           # (B,)

    # ---- 3. Quaternion → rotation-matrix (batch loop) --------------------------------
    # If your quaternion_to_rotmatrix already supports (B,4) input you can remove the loop.
    rotmats = np.stack([quaternion_to_rotmatrix(q) for q in quats], axis=0)  # (B, 3, 3)

    return xyz, rotmats, grip

def get_norm_actions(
    xyz: np.ndarray,           # (T, 3) absolute Cartesian positions
    rotmats: np.ndarray,       # (T, 3, 3) absolute rotation matrices
    grip: np.ndarray,          # (T,) gripper state 0 (open) → 1 (closed)
) -> torch.Tensor:
    """
    Normalise *absolute* actions (pose-as-state) into the same 10-D embedding
    used elsewhere:  [norm_xyz(3), r6(6), norm_grip(1)].

    It differs from `get_relative_norm_actions` only in the xyz term:
    • here we embed the absolute end-effector pose,
    • there the embed is a frame-to-frame delta.

    All numerical mapping is delegated to `normalize_action` for consistency.
    """
    xyz     = np.asarray(xyz,     dtype=float)
    rotmats = np.asarray(rotmats, dtype=float)
    grip    = np.asarray(grip,    dtype=float).reshape(-1)

    if not (xyz.shape[0] == rotmats.shape[0] == grip.shape[0]):
        raise ValueError("xyz, rotmats and grip must all share length T")

    T                = xyz.shape[0]
    actions_final    = np.zeros((T, 10), dtype=float)

    # Workspace statistics (already defined globally)
    scene_min        = SCENE_BOUNDS[:3]
    scene_max        = SCENE_BOUNDS[3:]
    scene_center     = 0.5 * (scene_min + scene_max)

    rows = []
    for i in range(T):
        # -- 1. Convert absolute xyz → synthetic “relative” vector -------------
        #     (maps scene_min→-span, scene_max→+span so `normalize_action`
        #      reaches its full range)
        rel_xyz = 2.0 * (xyz[i] - scene_center)        # ε ∈ [-span, +span]

        # -- 2. Extract r6 from absolute rotation ------------------------------
        r6 = rotmats[i, :, :2].reshape(6)              # first 2 cols → (6,)

        # -- 3. Normalise with the shared helper -------------------------------
        norm_xyz, norm_r6, norm_g = normalize_action(rel_xyz, r6, grip[i])

        rows.append(np.concatenate([norm_xyz, norm_r6, [norm_g]]))

    actions_final[:] = np.asarray(rows, dtype=float)
    return torch.tensor(actions_final, dtype=torch.float32)

def get_relative_norm_actions(
    xyz: np.ndarray,           # (T, 3)
    rotmats: np.ndarray,       # (T, 3, 3)
    grip: np.ndarray,          # (T,)
) -> torch.Tensor:
    """
    Compute normalised *relative* actions between consecutive poses.

    * Returns
    actions_final : ndarray, shape (T, 10)
        The first row is all zeros (no previous frame to compare with).
        Each subsequent row: [rel_xyz(3), r6(6), norm_grip(1)].
    """
    xyz     = np.asarray(xyz,     dtype=float)
    rotmats = np.asarray(rotmats, dtype=float)
    grip    = np.asarray(grip,    dtype=float)

    T = xyz.shape[0]
    actions_final = np.zeros((T, 10), dtype=float)          # row 0 stays zeros
    if T == 1:
        return actions_final                                # nothing more to compute

    # ---- 1. Relative translation ------------------------------------------------------
    delta_xyz      = xyz[1:] - xyz[:-1]                     # (T-1, 3)
    prev_rotm_T    = rotmats[:-1].transpose(0, 2, 1)        # (T-1, 3, 3)
    rel_xyz        = np.einsum('nij,nj->ni', prev_rotm_T, delta_xyz)  # (T-1, 3)

    # ---- 2. Relative rotation ---------------------------------------------------------
    rel_rotm       = np.einsum('nij,njk->nik', prev_rotm_T, rotmats[1:])  # (T-1, 3, 3)
    r6             = rel_rotm[:, :, :2].reshape(-1, 6)                    # (T-1, 6)

    # ---- 3. Normalise each relative action -------------------------------------------
    rows = []
    for i in range(T - 1):
        rel_xyz_, r6_, grip_ = normalize_action(rel_xyz[i], r6[i], grip[i + 1])
        rows.append(np.concatenate([rel_xyz_, r6_, [grip_]]))

    actions_final[1:] = np.asarray(rows, dtype=float)
    return torch.tensor(actions_final, dtype=torch.float32)

def normalize_action(rel_xyz, rel_r6, gripper, scale_range=(-2*np.pi, 2*np.pi)):
    """
    Map (Δxyz, 6-D rotation, gripper) into `scale_range`.

    rel_xyz : array-like (..., 3)
        Relative translation expressed in the previous end-effector frame.
    rel_r6  : array-like (..., 6)
        First two rows of the relative rotation matrix, each entry ∈ [-1, 1].
    gripper : float or array-like (...)
        0 → fully open, 1 → fully closed.
    scale_range : tuple(float, float), optional
        Target interval [low, high] for *all* normalised components
        (default is (-50, 50)).

    norm_xyz : ndarray (..., 3)
    norm_r6  : ndarray (..., 6)
    norm_g   : ndarray (...)
        All values lie inside `scale_range`.
    """
    low, high = scale_range

    # utility: linear map from [0,1] → [low, high]
    def _to_range(z, high=high, low=low, rescale=False):
        if rescale:
            high = high / 2.0
            low  = low / 2.0
        return z * (high - low) + low

    # 1. Δxyz  →  [-1, 1]  →  [low, high]
    span = SCENE_BOUNDS[3:] - SCENE_BOUNDS[:3]          # full workspace span
    xyz_norm01 = (np.clip(rel_xyz / np.maximum(span, 1e-8), -1.0, 1.0) + 1.0) * 0.5
    norm_xyz   = _to_range(xyz_norm01)

    # 2. r6  (already in [-1,1])  →  [low/100, high/10]
    r6_norm01  = (np.clip(rel_r6, -1.0, 1.0) + 1.0) * 0.5
    norm_r6    = _to_range(r6_norm01, rescale=True)

    # 3. gripper  ([0,1])  →  [low/100, high/100]
    g_norm01   = np.clip(gripper, 0.0, 1.0)
    norm_g     = _to_range(g_norm01, rescale=True)

    return norm_xyz, norm_r6, norm_g
# ============== Added implementation For Manipulation ==============


def get_action_ids(
    batch_size, actions, strategy, weight_type, use_absolute_pose=True, #'micro_cond' or 'action_block'
):
    if strategy == 'action_block' or strategy == 'action_block_nocfg':
        action_ids = action_ids_onehot_encode(batch_size, actions)
    elif strategy == 'micro_cond':
        if actions.dim() == 2:
            action_ids = action_ids_idx_encode(batch_size, actions)
        elif actions.dim() == 3:
            actions_rel_batch = []
            for i in range(batch_size):
                xyz, rotmats, grip = get_action_from_continuous(actions[i])
                if use_absolute_pose:
                    actions_rel = get_norm_actions(xyz, rotmats, grip)
                else:
                    actions_rel = get_relative_norm_actions(xyz, rotmats, grip)
                actions_rel_batch.append(actions_rel)
            actions_rel_batch = torch.stack(actions_rel_batch)
            # action_ids = action_encode_positional(batch_size, actions_rel_batch)
            action_ids = actions_rel_batch
    else:
        action_ids = torch.empty(0)

    return action_ids.to(dtype=weight_type, device=actions.device)


def action_encode_positional(batch_size, actions):
    """
    Encodes action ids into a positional format.
    Args:
        batch_size (int): Number of sequences in the batch.
        actions (torch.Tensor): Tensor of shape (batch_size, sequence_length, action_len).
    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length, action_max_len).
    """
    assert actions.dim() == 3, f"Expected actions to be 3D (batch_size, sequence_length, action_len), got {actions.shape}"
    action_len = actions.size(2)  # action_len
    sequence_len = actions.size(1)  # sequence_length
    action_max_len = sequence_len + action_len - 1
    action_seq_frames = []
    for b in range(batch_size):
        seq_frames = []
        for i in range(sequence_len):
            frame = torch.zeros(action_max_len, dtype=torch.float, device=actions.device)
            action_id = actions[b, i]
            frame[i: i+action_len] = action_id
            seq_frames.append(frame.unsqueeze(0).clone())
        action_seq_frames.append(torch.cat(seq_frames, dim=0).unsqueeze(0))
    return torch.cat(action_seq_frames, dim=0)

def action_ids_idx_encode(batch_size, actions):
    """
    Converts a batch of action id sequences into a sequence of frames.
    Args:
        batch_size (int): Number of sequences in the batch.
        actions (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing action ids.
                                The expected sequence length is assumed to be action_max_len.
    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length, action_max_len).
    """
    assert actions.dim() == 2, f"Expected actions to be 2D (batch_size, sequence_length), got {actions.shape}"
    # You may use the length from actions; here we assume the max length is 25
    action_max_len = actions.size(1)
    action_mapping = {'forward': 1, 'turn_left': 2, 'turn_right': 3, 'stop': 4, 'placeholder': 0}

    batch_frames = []  # To store the output for each sample in the batch.
    for b in range(batch_size):
        seq_frames = []  # To store the sequence of frames for sample b.
        # Loop over each time step in the sequence.
        frame = torch.zeros(action_max_len, dtype=torch.long, device=actions.device)
        for i in range(action_max_len):
            # Create a frame vector of length action_max_len, initially all zeros.
            # The shape here is (action_max_len,)
            # For the first time step, force the action to be 'stop'
            if i == 0:
                action_id = action_mapping['stop']
            else:
                # Otherwise, take the provided action id from the input.
                action_id = actions[b, i].item()  # Use .item() to get a Python number (optional)

            frame[i] = action_id
            seq_frames.append(frame.unsqueeze(0).clone())  # Now frame shape is (1, action_max_len)

        sample_seq_frames = torch.cat(seq_frames, dim=0)
        batch_frames.append(sample_seq_frames.unsqueeze(0))  # Now shape is (1, sequence_length, action_max_len)

    # Concatenate along the batch dimension.
    action_seq_frames = torch.cat(batch_frames, dim=0)  # Shape: (batch_size, sequence_length, action_max_len)
    return action_seq_frames


def set_first_action_as_stop(one_hot, stop_index=3):
    """
    Sets the first action in each sequence to 'stop' (one-hot [0, 0, 0, 1]).
    Returns:
        torch.Tensor: The one_hot tensor with the first action in each sequence set to 'stop'.
    """
    batch_size, sequence_length, num_classes = one_hot.shape
    # Create a tensor of shape (batch_size,) with all entries equal to stop_index.
    stop_indices = torch.full((batch_size,), stop_index, dtype=torch.long, device=one_hot.device)
    # Convert to one-hot representation: shape (batch_size, num_classes)
    stop_one_hot = F.one_hot(stop_indices, num_classes=num_classes)
    # Set the first action in each sequence to stop:
    one_hot = torch.cat([stop_one_hot.unsqueeze(1), one_hot[:, 1:, :]], dim=1)
    return one_hot

def action_ids_onehot_encode(batch_size, actions):
    """
    Converts a batch of action id sequences into their one-hot representations,
    and sets the first action in each sequence to 'stop' (one-hot [0, 0, 0, 1]).
    Args:
        batch_size (int): Number of sequences in the batch.
        actions (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing action ids.
                                Action ids are assumed to be one-indexed.
    Returns:
        torch.Tensor: One-hot shape is (batch_size, sequence_length, num_classes).
    """
    # For this example, we expect actions of shape [1, 25]. Adjust as needed.
    assert actions.dim() == 2 and actions.size(0) == batch_size, \
        f"Expected actions tensor of shape ({batch_size}, sequence_length), got {actions.shape}"
    # action_mapping = {'forward': 1, 'turn_left': 2, 'turn_right': 3, 'stop': 4, 'placeholder': 0}

    action_zero_indexed = actions - 1
    num_classes = 4  # we assume 4 classes: forward, turn_left, turn_right, stop

    # Output shape will be (batch_size, sequence_length, num_classes)
    # temporarily set the first action in each actions sequence to 1:
    action_zero_indexed[:, 0] = 1
    one_hot = F.one_hot(action_zero_indexed, num_classes=num_classes)
    # Set the first action in each sequence to 'stop' ([0, 0, 0, 1]).
    one_hot = set_first_action_as_stop(one_hot, stop_index=3)

    return one_hot


# =======from the original implementation:=======

def norm_image(pixel_values):
    # assert input pixel_values: [-1, 1]
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(0)

    pixel_values_list = []
    for i in range(pixel_values.shape[1]):
        pixel_values_ = pixel_values[:, i, :, :, :]
        pixel_values_ = _resize_with_antialiasing(pixel_values_, (224, 224))
        pixel_values_list.append(pixel_values_)
    pixel_values = torch.cat(pixel_values_list, dim=0)

    # We unnormalize it after resizing.
    pixel_values = (pixel_values + 1.0) / 2.0   # range [0, 1]
    return pixel_values


# resizing utils
def _resize_with_antialiasing(input, size, interpolation="bicubic", align_corners=True):
    h, w = input.shape[-2:]
    factors = (h / size[0], w / size[1])

    # First, we have to determine sigma
    # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
    sigmas = (
        max((factors[0] - 1.0) / 2.0, 0.001),
        max((factors[1] - 1.0) / 2.0, 0.001),
    )

    # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
    # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
    # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
    ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

    # Make sure it is odd
    if (ks[0] % 2) == 0:
        ks = ks[0] + 1, ks[1]

    if (ks[1] % 2) == 0:
        ks = ks[0], ks[1] + 1

    input = _gaussian_blur2d(input, ks, sigmas)

    output = torch.nn.functional.interpolate(
        input, size=size, mode=interpolation, align_corners=align_corners)
    return output


def _gaussian_blur2d(input, kernel_size, sigma):
    if isinstance(sigma, tuple):
        sigma = torch.tensor([sigma], dtype=input.dtype)
    else:
        sigma = sigma.to(dtype=input.dtype)

    ky, kx = int(kernel_size[0]), int(kernel_size[1])
    bs = sigma.shape[0]
    kernel_x = _gaussian(kx, sigma[:, 1].view(bs, 1))
    kernel_y = _gaussian(ky, sigma[:, 0].view(bs, 1))
    out_x = _filter2d(input, kernel_x[..., None, :])
    out = _filter2d(out_x, kernel_y[..., None])

    return out



def _compute_padding(kernel_size):
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding


def _filter2d(input, kernel):
    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...].to(
        device=input.device, dtype=input.dtype)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    padding_shape: list[int] = _compute_padding([height, width])
    input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = torch.nn.functional.conv2d(
        input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    out = output.view(b, c, h, w)
    return out


def _gaussian(window_size: int, sigma):
    if isinstance(sigma, float):
        sigma = torch.tensor([[sigma]])

    batch_size = sigma.shape[0]

    x = (torch.arange(window_size, device=sigma.device,
         dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

    if window_size % 2 == 0:
        x = x + 0.5

    gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

    return gauss / gauss.sum(-1, keepdim=True)
