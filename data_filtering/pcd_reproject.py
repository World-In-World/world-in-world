import quaternion
import numpy as np
import torch
from typing import Tuple
import math
try:
    from torch_scatter import scatter_min
    USE_TORCH_SCATTER = True
except ImportError:
    USE_TORCH_SCATTER = False
    print("Warning: torch_scatter not installed, falling back to slow scatter_min implementation.")
from collections import defaultdict


def habitat_camera_intrinsic(width, height, hfov=90):
    width = width; height = height
    xc = (width - 1.) / 2.  #x-coordinate of the center of an image
    zc = (height - 1.) / 2. #y-coordinate of the center of an image
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))    #This line calculates the focal length f of a camera. A larger focal length results in a narrower field of view (more zoomed in), while a smaller focal length results in a wider field of view (less zoomed in).
    intrinsic_matrix = np.array([[f,0,xc],
                                 [0,f,zc],
                                 [0,0,1]],np.float32)
    return intrinsic_matrix

def habitat_translation(position):
    return np.array([position[0],position[2],position[1]])

def pos_normal_to_habitat(pts):
    # +90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))

def pos_normal_to_habitat_ori(pts):
    # +90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))


def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def pose_habitat_to_normal(pose):
    # T_normal_cam = T_normal_habitat * T_habitat_cam
    return np.dot(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), pose
    )

def habitat_rotation(rotation):
    """Convert a rotation from Habitat's quaternion to a rotation matrix"""
    if isinstance(rotation, list):
        rotation = quaternion.from_float_array(rotation)
    rotation_matrix = quaternion.as_rotation_matrix(rotation)   #converts the input quaternion to a 3x3 rotation matrix
    transform_matrix = np.array([[1,0,0],
                                 [0,0,1],
                                 [0,1,0]])
    rotation_matrix = np.matmul(transform_matrix, rotation_matrix)   #used to reorient the rotation matrix. It appears to swap the y and z axes,
    return rotation_matrix


def quaternion_from_axis_angle(axis_angle):
    """
    Convert an axis (x,y,z) and a scalar angle (radians) to a quaternion (w, x, y, z).
    Axis must be a 3D vector in Python list or NumPy array form. We'll assume it's already normalized.
    """
    axis, angle = axis_angle
    axis = np.asarray(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)  # normalize
    half_angle = angle * 0.5
    w = math.cos(half_angle)
    xyz = axis * math.sin(half_angle)
    return quaternion.quaternion(w, xyz[0], xyz[1], xyz[2])


def euler_to_quaternion(euler, convention="XYZ"):
    """
    Convert Euler angles [rx, ry, rz] to a quaternion.
    The 'convention' param can define the rotation order, e.g. 'XYZ', 'ZYX', etc.
    Here we'll assume extrinsic rotations about X, Y, Z in that order.
    """
    rx, ry, rz = euler
    # For example, we can build it step by step:
    qx = quaternion.from_euler_angles(rx, 0, 0)
    qy = quaternion.from_euler_angles(0, ry, 0)
    qz = quaternion.from_euler_angles(0, 0, rz)

    # Compose them in the correct order. Commonly (rx, ry, rz) means Rz * Ry * Rx, but
    # it depends on your coordinate system. Adjust as needed.
    # For a simple approach, do:
    q = qz * qy * qx
    return q


def preprocess_depth_v0(depth:np.ndarray, lower_bound:float=0.025, depth_scale=10.0): #The preprocess_depth function processes a depth image by setting all depth values outside a specified range (between lower_bound and upper_bound) to zero.
    depth = depth / 255.0 * depth_scale      #unnormlize the depth image
    depth[np.where((depth<lower_bound)|(depth>depth_scale))] = 0
    return depth

def preprocess_depth(depth:np.ndarray,lower_bound:float=0.025, depth_scale=20.0): #The preprocess_depth function processes a depth image by setting all depth values outside a specified range (between lower_bound and upper_bound) to zero.
    assert depth.max() <= 1.0 and depth.min() >= 0.0, "Depth image should be normalized to [0, 1]"
    depth = depth * depth_scale      #unnormlize the depth image
    depth[np.where((depth<lower_bound)|(depth>depth_scale))] = 0
    return depth


def get_pointcloud_from_depth(
    rgb: np.ndarray, depth: np.ndarray, intrinsic: np.ndarray
):  # transform depth image to point cloud with camera coordinate system
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    filter_z, filter_x = np.where(depth > 0)  # row indices, column indices
    depth_values = depth[filter_z, filter_x]
    pixel_z = (
        (depth.shape[0] - 1 - filter_z - intrinsic[1][2])
        * depth_values
        / intrinsic[1][1]
    )  # Upward direction in the image.
    pixel_x = (
        (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
    )  # Rightward direction in the image
    pixel_y = depth_values  # Forward direction
    color_values = rgb[filter_z, filter_x]
    point_values = np.stack(
        [pixel_x, pixel_z, -pixel_y], axis=-1
    )  # shape is (307200, 3)
    return point_values, color_values


def camera_to_world(pointcloud, position, rotation):
    """
    This function transforms a point cloud from the camera coordinate system
    to the world coordinate system using the given position and rotation.
    """
    extrinsic = np.eye(4)
    extrinsic[0:3, 0:3] = rotation
    extrinsic[0:3, 3] = position
    world_points = np.matmul(
        extrinsic,
        np.concatenate((pointcloud, np.ones((pointcloud.shape[0], 1))), axis=-1).T,
    ).T  # P_w = T_wc P_c
    return world_points[:, 0:3]


def world_to_camera_(xyz_world, extrinsics):
    '''
    :param xyz_world: (..., 3) array of float.
    :param extrinsics: (4, 4) array of float.
    :return xyz_camera: (..., 3) array of float.
    '''
    xyz_camera = xyz_world - extrinsics[0:3, 3]
    xyz_camera = xyz_camera @ extrinsics[0:3, 0:3]
    return xyz_camera


def world_to_camera(
    points,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Projects a point cloud to the camera coordinate system.
    Args:
        pcd: Point cloud object with attribute 'points' of shape (N, 3).
        intrinsic (torch.Tensor): Intrinsic camera matrix of shape (3, 3).
        extrinsic (torch.Tensor): Extrinsic camera matrix of shape (4, 4).
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - filter_x (torch.Tensor): x pixel coordinates as integers of shape (N,).
            - filter_z (torch.Tensor): z pixel coordinates as integers of shape (N,).
            - depth_values (torch.Tensor): Depth values as floats of shape (N,).
    """
    # Construct the extrinsic matrix (world to camera)
    extrinsic = torch.inverse(extrinsic)  # Invert to get camera to world

    # Extract points and convert to homogeneous coordinates
    points_hom = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)], dim=1)  # (N, 4)

    # Transform points to camera coordinates
    camera_points = (extrinsic @ points_hom.T).T[:, :3]  # (N, 3)

    # Calculate depth values, Prevent division by zero
    depth_values = -camera_points[:, 2]  # (N,)

    # Extract intrinsic parameters
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    # Project points onto the image plane
    filter_x = (((camera_points[:, 0] * fx / depth_values) + cx) + 0.5).to(torch.int32)  # (N,)
    filter_z = (((-camera_points[:, 1] * fy / depth_values) - cy + (cy * 2) - 1) + 0.5).to(torch.int32)  # (N,)

    uv = torch.stack([filter_x, filter_z], dim=1)
    return uv, depth_values.unsqueeze(1)


def spreaded_index_add_(tensor, indices, values, H, W, radius):
    '''
    :param tensor: (N, C) tensor of any type. ==> pixel_weights_flat
    :param indices: (M) tensor of int with values in [0, N - 1]. ==> inds_flat
    :param values: (M, C) tensor of any type. ==> point_weights
    :return tensor: (N, C) tensor of any type.
    '''
    # Accumulate values at indices in-place within tensor.
    tensor.index_add_(0, indices, values)   #

    # NOTE: Only the above line would be the default / vanilla operation, but we wish to
    # avoid random pixel holes inbetween points, which requires a more advanced algorithm.
    # left = radius // 2
    # right = (radius + 1) // 2
    # offset_list = []
    # for dx in range(-left, right + 1):
    #     for dy in range(-left, right + 1):
    #         if dx == 0 and dy == 0:
    #             continue
    #         offset_list.append((dx, dy))
    # # when radius = 1: [(1, 0), (0, 1), (1, 1)].
    # # when radius = 2: [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)].
    # # when radius = 3: x and y go from -1 to 2 inclusive, etc.. (length = 15)

    # for (dx, dy) in offset_list:
    #     # Spread values to neighboring pixels.
    #     inds_x = indices % W + dx
    #     inds_y = indices // W + dy
    #     shift_inds = inds_y * W + inds_x

    #     # Also avoid leaking across image borders.
    #     # shift_inds = torch.clamp(shift_inds, min=0, max=H * W - 1)
    #     mask = (inds_x >= 0) & (inds_x < W) & (inds_y >= 0) & (inds_y < H)
    #     mask_inds = shift_inds[mask]
    #     mask_values = values[mask] * 0.02  # Weaken as original pixels should have priority.

    #     tensor.index_add_(0, mask_inds, mask_values)

    return tensor


def spreaded_index_add(tensor, indices, values, H, W, radius):
    """
    Perform a two-pass radius-based spread:
      1) Accumulate 'values' into 'tensor' at 'indices', then spread to neighbors for *all* indices.
      2) Calculate overlap counts after the first spread. For pixels with overlap_count > 1,
         spread again (a second pass) using the pixel's current value in 'tensor'.
    Args:
        tensor (Tensor): (N, C) float tensor updated in-place (N = H*W).
        indices (Tensor): (M,) integer tensor in [0, N-1].
        values (Tensor): (M, C) float tensor, same channel dimension as 'tensor'.
        H (int): Image height.
        W (int): Image width.
        radius (int): Neighborhood size for spreading.
    Returns:
        Tensor: The updated 'tensor' (also updated in-place).
    """
    if radius <= 2:
        num_neighbor = 1
    else:
        num_neighbor = 2
    device = tensor.device
    N, C = tensor.shape  # N should be H*W

    # Accumulate the provided values directly into 'tensor'.
    tensor.index_add_(0, indices, values)

    # Also track how many times each pixel is "hit"  We use an integer count_tensor of shape (N,).
    count_tensor = torch.zeros(N, dtype=torch.long, device=device)
    one_counts = torch.ones_like(indices, dtype=torch.long, device=device)
    count_tensor.index_add_(0, indices, one_counts)

    # 2) FIRST PASS RADIUS SPREAD (FOR ALL PIXELS)
    # We also do a parallel spread for 'count_tensor' to track how many times neighbors got touched.
    left = radius // 2
    right = (radius + 1) // 2
    offset_list = []
    for dx in range(-left, right + 1):
        for dy in range(-left, right + 1):
            if dx == 0 and dy == 0:
                continue
            offset_list.append((dx, dy))

    # Spread for first pass
    # We'll do the exact same approach for 'count_tensor' (with an increment of 1 to each neighbor),
    for dx, dy in offset_list:
        # Convert flat indices to (x, y)
        inds_x = indices % W + dx
        inds_y = indices // W + dy
        shifted_inds = inds_y * W + inds_x

        # Mask out-of-bounds
        in_bounds = (inds_x >= 0) & (inds_x < W) & (inds_y >= 0) & (inds_y < H)
        neighbor_inds = shifted_inds[in_bounds]

        # Also update the count_tensor by +1 for each neighbor
        one_counts_neighbor = torch.ones_like(neighbor_inds, dtype=torch.long, device=device)
        count_tensor.index_add_(0, neighbor_inds, one_counts_neighbor)

    # 3) DETERMINE OVERLAP AFTER FIRST PASS
    # Now that we've done the first spread, 'count_tensor[p]' tells us how many times pixel p was touched.
    overlap_mask = (count_tensor > num_neighbor)                   # True if a pixel was touched by more than 1 point.
    overlap_indices = overlap_mask.nonzero().flatten()  # All pixel indices with overlap>1

    if overlap_indices.numel() == 0:
        # If there are no overlapping pixels, we're done.
        return tensor

    # 4) SECOND PASS RADIUS SPREAD (ONLY FOR OVERLAPPING PIXELS)
    overlap_vals = tensor[overlap_indices]  # shape (K, C), these are the current accumulated values
    for dx, dy in offset_list:
        inds_x = overlap_indices % W + dx
        inds_y = overlap_indices // W + dy
        shifted_inds = inds_y * W + inds_x

        in_bounds = (inds_x >= 0) & (inds_x < W) & (inds_y >= 0) & (inds_y < H)
        neighbor_inds = shifted_inds[in_bounds]

        neighbor_vals = overlap_vals[in_bounds] * 0.02
        tensor.index_add_(0, neighbor_inds, neighbor_vals)

    return tensor


def project_points_to_pixels(xyzrgb, cur_indexs, K, RTs, H, W, device,
                             start_step, spread_radius=3):
    '''
    :param xyzrgb: (N, 6) tensor of float, where N can be any value.
    :param cur_indexs: (N) tensor of int, representing the index to distinguish source frame of the point cloud.
    :param K: (3, 3) tensor of float.
    :param RTs: (*6,*  4, 4) tensor of float.
    :return img_norm: (H, W, 3) tensor of float.
    :return pixel_weights: (H, W, 1) tensor of float.
    :return uv: (N, 2) tensor of float.
    :return depth: (N, 1) tensor of float.
    '''
    xyzrgb, K, RTs = convert_to_tensor(xyzrgb, K, RTs, device)
    cur_indexs = cur_indexs.to(device)

    # 1) Project from world -> camera -> pixel coords
    xyz_world, rgb = xyzrgb[:, 0:3], xyzrgb[:, 3:6]

    # for step_id in range(start_step, current_step):
    result_cube = defaultdict(list)
    for cube_id in range(6):
        RT = RTs[cube_id]
        uv_int, depth = world_to_camera(xyz_world, K, RT)
        # del uv_int, depth

        # 2) Filter out-of-image or invalid-depth
        mask_in_image = (
            (uv_int[:, 0] >= 0) & (uv_int[:, 0] < W) &
            (uv_int[:, 1] >= 0) & (uv_int[:, 1] < H) &
            (depth[:, 0] > 0.02)
        )
        uv_int_filter   = uv_int[mask_in_image]
        depth_filter    = depth[mask_in_image]
        rgb_filter      = rgb[mask_in_image]
        cur_indexs_filt = cur_indexs[mask_in_image]
        # del mask_in_image, uv_int, depth, rgb, cur_indexs

        # 4) Find minimal depth per pixel (fast approach using scatter) Then keep points within depth_threshold of that minimal depth.
        pixel_coords_unique, pixel_ids = torch.unique(      # pixel_ids is a reverse index
            uv_int_filter, dim=0, return_inverse=True
        )
        depth_threshold = 0.08

        min_depth_mask = get_min_depth_mask(
            depth_filter, pixel_coords_unique, pixel_ids, depth_threshold
        )
        # del pixel_coords_unique, pixel_ids

        # 3) Keep only points from 'target_step' (fallback=0)
        # for step_id in range(start_step, current_step): #TODO
        step_mask = (cur_indexs_filt == start_step)

        # Optionally keep only those final points
        final_keep_mask  = min_depth_mask & step_mask
        uv_int_filter_   = uv_int_filter[final_keep_mask]
        depth_filter_    = depth_filter[final_keep_mask]
        rgb_filter_      = rgb_filter[final_keep_mask]
        # del step_mask, final_keep_mask

        # 5) Build an output image according to depth
        if rgb_filter_.shape[0] > 0:
            target_step_img_norm, void_mask = build_output_image(
                uv_int_filter_, depth_filter_, rgb_filter_, H, W, spread_radius,
            )
        else:
            print("Warrning:No pcd points in the target step can be reprojected to the current frame.")
            # target_step_img_norm should be all black pixels and void_mask should be all True
            target_step_img_norm, void_mask = torch.zeros(H, W, 3, device=device), torch.ones(H, W, 1, device=device)

        num_void = void_mask.sum().item()
        void_ratio = num_void / (H * W)
        result_cube['img_norm'].append(target_step_img_norm.cpu().numpy())
        result_cube['void_mask'].append(void_mask.cpu().numpy())
        result_cube['void_ratio'].append(void_ratio)

    return result_cube

def convert_to_tensor(xyzrgb, K, RTs, device):
    """
    Convert numpy arrays to PyTorch tensors and move them to the specified device.
    """
    data_list = [xyzrgb, K, RTs]; tensor_list = []
    for tensor in data_list:
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.device.type == 'cpu' or tensor.dtype != torch.float64:
            tensor = tensor.to(device, dtype=torch.float64)
        tensor_list.append(tensor)
    xyzrgb, K, RTs = tensor_list
    return xyzrgb, K, RTs


def get_min_depth_mask(depth_filter, pixel_coords_unique, pixel_ids, depth_threshold):
    if USE_TORCH_SCATTER:
        # a) scatter_min to get the minimal depth (and index) per pixel
        min_depth_per_pixel, min_inds_per_pixel = scatter_min(
            depth_filter[:, 0],  # Values to compare
            pixel_ids,           # Group/bucket IDs
            dim=0                # We scatter along dimension 0
        )
        # b) points within threshold of that minimal depth
        depth_diff = depth_filter[:, 0] - min_depth_per_pixel[pixel_ids]    #check min_depth_per_pixel.shape
        keep_mask = (depth_diff <= depth_threshold)

        # If you want to keep *only* the single "true minimal" point:
        # final_mask = exact_min_mask
        # If you want to keep all points within 0.01:
        min_depth_mask = keep_mask
    else:
        num_pixels = pixel_coords_unique.size(0)
        # a) no scatter => do a slow Python loop or your prior approach
        # Create a big array to hold minimal depth
        min_depth_per_pixel = torch.full(
            (num_pixels,),
            float('inf'),
            dtype=depth_filter.dtype,
            device=depth_filter.device
        )
        # We'll also track the "index of the minimal depth"
        min_inds_per_pixel = torch.full(
            (num_pixels,),
            -1,
            dtype=torch.long,
            device=depth_filter.device
        )
        # Accumulate minimal depths
        for i in range(depth_filter.shape[0]):
            p_idx = pixel_ids[i].item()
            zval = depth_filter[i, 0].item()
            if zval < min_depth_per_pixel[p_idx]:
                min_depth_per_pixel[p_idx] = zval
                min_inds_per_pixel[p_idx] = i
        # b) keep points within threshold
        depth_diff = depth_filter[:, 0] - min_depth_per_pixel[pixel_ids]
        keep_mask = (depth_diff <= depth_threshold)

        # If you want all within threshold:
        min_depth_mask = keep_mask
    return min_depth_mask


def build_output_image(uv_int_filter, depth_filter, rgb_filter, H, W, spread_radius):
    """
    Build an output image by accumulating weighted RGB values per pixel,
    where closer points (lower depth) have higher weight.
    Args:
        uv_int_filter: (M, 2) tensor of int, filtered pixel coordinates.
        depth_filter: (M, 1) tensor of float64, corresponding depth values.
        rgb_filter: (M, 3) tensor of float64, corresponding color values.
        H: int, image height.
        W: int, image width.
    Returns:
        target_step_img_norm: (H, W, 3) tensor of float32 with values in [0, 1].
        void_mask: (H, W, 1) tensor of bool, True where no point contributed.
    """
    # 1) Flatten 2D pixel coordinates into a 1D index
    inds_flat = uv_int_filter[:, 1] * W + uv_int_filter[:, 0]  # shape (M,)

    # 2) Compute weighting by depth so that closer points contribute more.
    strength = 512.0
    depth_norm = depth_filter / depth_filter.max() * 2.0 - 1.0  # shape (M, 1)
    point_weights = torch.exp(-depth_norm * strength)           # shape (M, 1)
    weighted_rgb = rgb_filter * point_weights                   # shape (M, 3)

    # 3) Allocate accumulators for per-pixel weights and weighted color sums.
    pixel_weights_flat = torch.zeros(H * W, 1, dtype=torch.float64, device=rgb_filter.device)
    img_flat = torch.zeros(H * W, 3, dtype=torch.float64, device=rgb_filter.device)

    # 4) Scatter-add contributions to the accumulators.
    spreaded_index_add(pixel_weights_flat, inds_flat, point_weights, H, W, radius=spread_radius)
    spreaded_index_add(img_flat, inds_flat, weighted_rgb, H, W, radius=spread_radius)

    # 5) Reshape accumulators into image form.
    pixel_weights = pixel_weights_flat.view(H, W, 1)  # (H, W, 1)
    img_agg = img_flat.view(H, W, 3)                  # (H, W, 3)

    # 6) Build a void mask (no contribution) and set weights to -1.0 to avoid division by zero.
    void_mask = (pixel_weights <= 0.0)
    pixel_weights[void_mask] = -1.0

    # 7) Normalize the final image by dividing color sums by the accumulated weights,
    target_step_img_norm = (img_agg / pixel_weights).clamp(0.0, 1.0).float()

    return target_step_img_norm, void_mask
