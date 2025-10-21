import math
import quaternion
from habitat.config.default_structured_configs import (
    HeadDepthSensorConfig,
    HeadRGBSensorConfig,
    HabitatSimSemanticSensorConfig,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.config.default import get_agent_config
from habitat.config.read_write import read_write
from copy import deepcopy
import numpy as np
import torch
import os
from collections import defaultdict
from jaxtyping import Float, Int32, UInt8
from typing import Any, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import Tensor
# from sklearn.cluster import DBSCAN


def find_leaf_candidates(dist_matrix, alpha=1.7):
    """
    Identify leaf candidates based on eccentricity and an average-distance proxy
    (similar in spirit to closeness centrality).
    We assume dist_matrix[i, j] is the minimal distance between points i and j.
    Args:
        dist_matrix (np.ndarray): Shape (N, N), fully connected (no infinities),
                                  with dist_matrix[i, j] = dist_matrix[j, i].
        alpha (float): Weighting factor to combine eccentricity with average distance.
    Returns:
        leaf_candidates (list): A list of (node_index, score) sorted by descending score.
    """
    N = dist_matrix.shape[0]
    ecc_dict = {}
    closeness_dict = {}

    for i in range(N):
        # Eccentricity = max distance from i to any other node
        ecc_i = np.max(dist_matrix[i, :])
        ecc_dict[i] = ecc_i

        # Sum of distances to all other nodes
        sum_dist = np.sum(dist_matrix[i, :])

        # Closeness: typically (N - 1) / (sum of distances)
        # If sum_dist = 0 (degenerate case), we set closeness to 0.
        if sum_dist > 0:
            c_i = (N - 1) / sum_dist
        else:
            raise ValueError(f"Node {i} has zero sum of distances.")
        closeness_dict[i] = c_i

    leaf_candidates = []
    for i in range(N):
        ecc = ecc_dict[i]
        c = closeness_dict[i]
        if c > 0:
            # 1/c is proportional to average distance
            score = ecc + alpha * (1.0 / c)
        else:
            # If closeness is zero, we can consider this extremely "leafy" or handle differently
            score = float('inf')
            print(f"WARNNING: Node {i} has zero closeness, setting score to infinity.")
        leaf_candidates.append((i, score))

    # Sort by descending score (bigger = more leaf-like)
    leaf_candidates.sort(key=lambda x: x[1], reverse=True)

    return leaf_candidates


def compute_cluster_centers(points: np.ndarray, labels: np.ndarray):
    """
    For each cluster label >= 0, compute the mean of points in that cluster.
    Returns:
        cluster_centers: List of np.array([x, y, z]) for each cluster.
        and a dict:
        {cluster_label: cluster_points}, where cluster_points is np.ndarray of shape (N, 3).
    """
    cluster_centers = []
    cluster_split = {}
    unique_labels = set(labels.tolist())
    for lbl in unique_labels:
        if lbl == -1:
            continue  # ignore noise cluster
        cluster_indices = np.where(labels == lbl)[0]
        split_points = points[cluster_indices]
        center = split_points.mean(axis=0)
        cluster_centers.append(center)
        cluster_split[center] = [p for p in split_points]
    return cluster_centers, cluster_split



def save_episodes(uni_eps, save_path):
    vals = []; keys = []
    for path, v in uni_eps.items():
        path = os.path.basename(path)
        keys.append(path)
        vals.append(v)
    uni_ = dict(zip(keys, vals))
    # filter out the scene, if necessary:
    scene_episodes = {k: v for k, v in uni_.items() if 'qvNra81N8BU.basis.glb' not in k}
    torch.save(scene_episodes, save_path)
    print(f'Successfully saved episodes as <{save_path}>')
    return True


def cal_img_near_black_ratio(image: torch.Tensor, black_threshold: int = 5, device='cuda') -> float:
    """
    Calculate the ratio of pixels in an image that are near black.
    Args:
        image (torch.Tensor): Input RGB image of shape (1, 3, H, W).
        black_threshold (int): Maximum RGB value considered "near black".
    Returns:
        float: Ratio of near-black pixels in the image.
    """
    # Ensure the input is in the correct shape (1, 3, H, W)
    assert image.ndim == 4 and image.shape[1] == 3, "Input must be of shape (1, 3, H, W)"

    # Convert the RGB image to grayscale using a weighted sum
    # Grayscale formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray_image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]

    # Count near-black pixels (values <= black_threshold)
    near_black_pixels = (gray_image < black_threshold).sum()

    # Calculate the total number of pixels
    total_pixels = gray_image.numel()

    # Compute the ratio of near-black pixels
    ratio = near_black_pixels.item() / total_pixels

    return ratio


def nearest_neighbor_tsp(points, dist_matrix):
    """
    Perform a simple Nearest Neighbor TSP on the set of points.
    points: List of np.array([x, y, z]) (length N).
    dist_matrix: NxN numpy array of pairwise geodesic distances.

    Returns:
        route: a list of points in the visiting order.
        total_dist: the total distance of that route.
    """
    N = len(points)
    visited = [False] * N
    route_indices = []

    # We'll pick the first point (index=0) as the start
    current_idx = 0
    route_indices.append(current_idx)
    visited[current_idx] = True

    # Track total distance traveled
    total_dist = 0.0

    # For each of the remaining points to visit
    for _ in range(N - 1):
        # Find the closest unvisited neighbor
        best_dist = float('inf')
        best_j = None
        for j in range(N):
            if not visited[j]:
                d = dist_matrix[current_idx][j]
                if d < best_dist:
                    best_dist = d
                    best_j = j
        # Mark that neighbor as visited
        visited[best_j] = True
        route_indices.append(best_j)
        total_dist += best_dist
        current_idx = best_j

    # Convert indices to actual 3D positions
    route = [points[i] for i in route_indices]

    return route, total_dist


def best_middle_neighbor(points, dist_row, dist_thr=2):
    """
    Select the best single point based on the "middle" distance ordering criteria.
    The point closest to dist_thr is selected, regardless of being above or below it.
    Args:
        points: List of np.array([x, y, z]) (length N).
        dist_row: 1D numpy array of distances from a reference point to each point (length N).
        dist_thr: Threshold distance to influence selection.
    Returns:
        best_point: The best point (np.array) based on the criteria.
    """
    if len(points) != len(dist_row):
        raise ValueError("Length of points and dist_row must be the same.")

    # Compute absolute differences from dist_thr
    differences = np.abs(dist_row - dist_thr)

    # Find the index of the smallest difference
    best_idx = np.argmin(differences)

    # Return the best point
    return points[best_idx]



def habitat_rotation(rotation):
    rotation_matrix = quaternion.as_rotation_matrix(rotation)   #converts the input quaternion to a 3x3 rotation matrix
    transform_matrix = np.array([[1,0,0],
                                 [0,0,1],
                                 [0,1,0]])
    rotation_matrix = np.matmul(transform_matrix, rotation_matrix)   #used to reorient the rotation matrix. It appears to swap the y and z axes,
    return rotation_matrix


def configure_cubemap_sensors(sim_cfg, enable_depth, depth_scale=20.0, sensor_height=1.5, enable_semantic=False):
    """
    Configures cubemap cameras and sets up equirectangular transformers.
    Args:
        meta_config (omegaconf.dictconfig.DictConfig): The Habitat configuration.
        enable_depth (bool): Whether to enable depth sensors.
    Returns:
        dict: Dictionary containing equirect transformers for RGB and depth.
    """
    with read_write(sim_cfg):
        # config = sim_cfg.habitat
        agent_config = get_agent_config(sim_cfg)
        CAMERA_NUM = 6
        uuid_str = ['back', 'down', 'front', 'right', 'left', 'up']
        orient = [
            [0, math.pi, 0],                # Back
            [-math.pi / 2, 0, 0],           # Down
            [0, 0, 0],                      # Front
            [0, math.pi / 2, 0],            # Right  #TODO check order and retest this func for Renderer
            [0, 3 / 2 * math.pi, 0],        # Left
            [math.pi / 2, 0, 0],            # Up
        ]
        sensor_uuids_dict = {}
        position = [v.position for k, v in agent_config.sim_sensors.items()][0]
        position[1] = sensor_height  # Set the height of the sensors
        cube_cameras = {"rgb_sensor": HeadRGBSensorConfig(width=768, height=768,
                                                            position=position)}
        if enable_depth:
            cube_cameras.update({
                "depth_sensor": HeadDepthSensorConfig(
                    width=768, height=768,
                    position=position,
                    normalize_depth=False,
                    max_depth=depth_scale,
                )
            })
        if enable_semantic:
            cube_cameras.update({
                "semantic_sensor": HabitatSimSemanticSensorConfig(
                    width=768, height=768, position=position
                )
            })

        sensor_types = ["rgb"]
        if enable_depth: sensor_types.append("depth")
        if enable_semantic: sensor_types.append("semantic")
        for sensor_type in sensor_types:
            sensor_uuids = []
            sensor_temp = cube_cameras[f"{sensor_type}_sensor"]
            for camera_id in range(CAMERA_NUM):
                camera_name = f"{sensor_type}_{uuid_str[camera_id]}"
                camera_config = deepcopy(sensor_temp)
                camera_config.orientation = orient[camera_id]
                camera_config.uuid = camera_name
                agent_config.sim_sensors[camera_name] = camera_config
                sensor_uuids.append(camera_config.uuid)
            sensor_uuids_dict[sensor_type] = sensor_uuids

        # sim_cfg.habitat = config

    return sensor_uuids_dict


def configure_equirect_tfm(height, width, enable_depth, sensor_uuids_dict, enable_semantic=False):
    # Initialize equirect transformers
    obs_trans_to_eq = baseline_registry.get_obs_transformer("CubeMap2Equirect")
    cube2equirect_rgb = obs_trans_to_eq(sensor_uuids_dict['rgb'], (height, width))
    transformers = {'rgb': cube2equirect_rgb}
    if enable_depth:
        cube2equirect_dep = obs_trans_to_eq(sensor_uuids_dict['depth'], (height, width))
        transformers['depth'] = cube2equirect_dep
    if enable_semantic:
        cube2equirect_dep_sem = obs_trans_to_eq(sensor_uuids_dict['semantic'], (height, width))
        transformers['semantic'] = cube2equirect_dep_sem

    return transformers


def convert_panos_to_cube(v_frames, width=512, height=512):
    """Batched version of converting equirectangular images to cubemap images"""
    B, C, H, W = v_frames.shape
    pano_images = {
        i: {"rgb": torch.einsum("chw->hwc", v_frames[i])}
        for i in range(B)
    }
    cube_images = tfm_equi_to_cube(
        sensor_uuids=["rgb"],
        pano_images=pano_images,
        width=width, height=height,
    )
    # Now shape is [6, H, W, 3] for the "rgb" key at step 1:
    # face_images = cube_images[1]["rgb"]
    cube_frames_all = []
    for i in range(B):
        cube_frames: Float[Tensor, "6 H W 3"] = cube_images[i]["rgb"]
        cube_frames = torch.einsum("bhwc->bchw", cube_frames)
        cube_frames_all.append(cube_frames)
    cube_frames_all = torch.stack(cube_frames_all, dim=0)
    return cube_frames_all


def tfm_equi_to_cube(pano_images, sensor_uuids=['rgb', 'depth'], width=512, height=512):
    """
    Convert equirectangular panoramic images to cubemap images for specified sensors.
    Args:
        pano_images (dict): A dictionary where keys are step indices and values are dictionaries
                            mapping sensor UUIDs to their corresponding equirectangular images
                            (numpy arrays). Each image is expected to have shape (H, W, 3) for RGB
                            or (H, W) for depth images.
        sensor_uuids (list, optional): List of sensor identifiers to process (default is ['rgb', 'depth']).
        width (int, optional): Desired width for the cubemap images.
        height (int, optional): Desired height for the cubemap images.
    Returns:
        defaultdict: A nested dictionary containing cubemap images for each step and sensor, structured as:
                     {step_idx: {sensor_name: cubemap_tensor, ...}, ...}.
    """
    cube_images = defaultdict(lambda: defaultdict(dict))

    obs_trans_to_cube = baseline_registry.get_obs_transformer("Equirect2CubeMap")
    eq2cube_tfms = {}
    for uuid in sensor_uuids:
        eq2cube_tfms[uuid] = obs_trans_to_cube([f"{uuid}_eq"], (width, height))

    # Go through each step that has loaded pano images
    for step_idx, img_dict in pano_images.items():
        # Build the batch dict for each sensor
        for uuid in sensor_uuids:
            eq_img = img_dict[uuid]  # np.array, shape (H, W, 3) or (H, W)
            if eq_img.ndim == 2:
                # Depth case: shape (H, W) -> (H, W, 1)
                eq_img = eq_img[..., np.newaxis]
            eq_img = np.expand_dims(eq_img, axis=0)  # (1, H, W, C)
            assert eq_img.shape[-1] < 4, "Only RGB or single-channel depth images supported."
            if eq_img.shape[-1] == 1:
                batch_eq = {f"{uuid}_eq": torch.from_numpy(eq_img / 65535.0)}
            else:
                batch_eq = {f"{uuid}_eq": torch.from_numpy(eq_img)}

            # Convert to torch Tensors
            batch_cube = eq2cube_tfms[uuid](batch_eq)
            for k, v in batch_cube.items():
                sensor_name = k.replace("_eq", "")
                # v is shape (6, H, W, C)
                cube_images[step_idx][sensor_name] = v

    return cube_images


def set_agent_heading(
    # old_wxyz,  # existing rotation (w, x, y, z)
    x1, z1, x2, z2,
):
    """
    Returns a new rotation (w, x, y, z) so that +Z faces from (x1,z1)->(x2,z2),
    ignoring the old heading in old_wxyz.
    """
    # 1) Compute the new heading quaternion
    dx = x2 - x1
    dz = z2 - z1
    yaw = math.atan2(-dx, -dz)
    # w = cos(yaw/2)
    # (x, y, z) = (0, sin(yaw/2), 0)
    half_yaw = 0.5 * yaw
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    # python 'quaternion' library uses 'quaternion(w, x, y, z)' in that order.
    new_q = quaternion.quaternion(cy, 0.0, sy, 0.0)

    # 2) If we just want to *replace* the orientation:
    return new_q
