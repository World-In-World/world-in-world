# https://raw.githubusercontent.com/facebookresearch/open-eqa/refs/heads/main/data/hm3d/config.py
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"
from copy import deepcopy
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from habitat_sim import (
    ActionSpec,
    ActuationSpec,
    AgentState,
    CameraSensorSpec,
    Configuration,
    SensorType,
    Simulator,
    SimulatorConfiguration,
)
from habitat_sim.agent import AgentConfiguration
from PIL import Image

from downstream.prompts import UNIT_DEGREE, unit_degree_look, UNIT_DISTANCE
from habitat_data.habitat_util import configure_equirect_tfm
from habitat_data.config_utils import hm3d_config, hssd_config, mp3d_config
from downstream.utils.util import rgba2rgb


def _create_sensor_spec(
    uuid: str, type, hfov, height, width, sensor_position, sensor_pitch, sensor_roll=0.0
):
    spec = CameraSensorSpec()
    spec.uuid = uuid
    spec.hfov = hfov
    spec.sensor_type = type
    spec.resolution = [height, width]
    spec.position = [0.0, sensor_position, 0.0]
    spec.orientation = [sensor_pitch, sensor_roll, 0.0]
    return spec


def _add_move_actions(action_space, amount):
    for key in [
        # "move_backward",
        "move_forward",
        # "move_left",
        # "move_right",
        # "move_down",
        # "move_up",
    ]:
        action_space[key] = ActionSpec(key, ActuationSpec(amount=amount))


def _add_turn_actions(action_space, amount):
    for key in ["turn_left", "turn_right"]:
        action_space[key] = ActionSpec(key, ActuationSpec(amount=amount))


def _add_look_actions(action_space, amount):
    for key in ["look_up", "look_down"]:
        action_space[key] = ActionSpec(key, ActuationSpec(amount=amount))


default_settings = {
    "random_seed": 42,
    "scene_id": None,
    "sensor_hfov": 90.0,
    "enable_surround_sensor": False,
    "sensor_hfov_cubemap": 90.0,
    "sensor_width": 512,
    "sensor_height": 512,
    "sensor_width_cubemap": 512,
    "sensor_height_cubemap": 512,
    "sensor_position": 1.0,  # height only
    "sensor_pitch": np.deg2rad(0),
    "agent_height": 1.0,
    "agent_radius": 0.1,
    "enable_depth": True,
    "enable_semantic": False,
}


def get_sensors(s, prefix) -> List[CameraSensorSpec]:
    sensors = []
    surround_order = ["front", "left", "back", "right"]
    if "rgb" in prefix:
        type = SensorType.COLOR
    elif "depth" in prefix:
        type = SensorType.DEPTH
    elif "semantic" in prefix:
        type = SensorType.SEMANTIC

    if "surround" in prefix:
        for i in range(4):
            sensors.append(
                _create_sensor_spec(
                    f"{prefix}_{surround_order[i]}",
                    type,
                    s["sensor_hfov"],
                    s["sensor_height"],
                    s["sensor_width"],
                    s["sensor_position"],
                    s["sensor_pitch"],
                    sensor_roll=np.deg2rad(90 * i),
                )
            )
    else:
        sensors.append(
            _create_sensor_spec(
                f"{prefix}",
                type,
                s["sensor_hfov"],
                s["sensor_height"],
                s["sensor_width"],
                s["sensor_position"],
                s["sensor_pitch"],
            )
        )
    return sensors

def make_cfg(settings: Dict) -> SimulatorConfiguration:
    s = default_settings | settings     #merge dicts
    print(f"Simulator settings: {s}")
    assert s["scene_id"] is not None

    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = s["scene_id"]
    if "scene_dataset_config_file" in s:  # * for MP3D & AR Dataset
        sim_cfg.scene_dataset_config_file = s["scene_dataset_config_file"]
    sim_cfg.random_seed = s["random_seed"]

    agent_cfg = AgentConfiguration()
    agent_cfg.height = s["agent_height"]
    agent_cfg.radius = s["agent_radius"]

    # sensors
    agent_cfg.sensor_specifications = []
    if s["enable_surround_sensor"]:
        rgb_sensors = get_sensors(s, prefix="rgb_surround")
        depth_sensors = get_sensors(s, prefix="depth_surround")
        semantic_sensors = get_sensors(s, prefix="semantic_surround")
    else:
        rgb_sensors = get_sensors(s, prefix="rgb_sensor")
        depth_sensors = get_sensors(s, prefix="depth_sensor")
        semantic_sensors = get_sensors(s, prefix="semantic_sensor")

    agent_cfg.sensor_specifications.extend(rgb_sensors)
    if s["enable_depth"]:
        agent_cfg.sensor_specifications.extend(depth_sensors)
    if s["enable_semantic"]:
        agent_cfg.sensor_specifications.extend(semantic_sensors)

    # actions
    agent_cfg.action_space = {}
    _add_move_actions(agent_cfg.action_space, amount=UNIT_DISTANCE)
    _add_turn_actions(agent_cfg.action_space, amount=UNIT_DEGREE)
    # _add_look_actions(agent_cfg.action_space, amount=unit_degree_look)

    return Configuration(sim_cfg, [agent_cfg])


# https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/nb_python/ECCV_2020_Navigation.py#L90-L116
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]), depth_scale=10.0):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / depth_scale * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)


def get_simulator(scene_id, device_id, **kwargs):
    args = {"scene_id": scene_id}
    args.update(kwargs)
    if "mp3d" in scene_id:
        args["scene_dataset_config_file"] = (
            "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
        )
    if "hm3d" in scene_id:
        args["scene_dataset_config_file"] = (
            "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        )
    sim_cfg = make_cfg(args)
    cube2equirect_tfms = configure_pano_sensors(sim_cfg, args)
    sim_cfg.sim_cfg.gpu_device_id = device_id
    sim = Simulator(sim_cfg)

    # assert len(sim.semantic_scene.objects) < 65535      #if we use Uint16 for semantic img
    return sim, cube2equirect_tfms


def configure_pano_sensors(sim_cfg, args):
    # Setup cubemap cameras and equirect transformers
    sensor_uuids_dict = configure_cubemap_sensors_sim(
        sim_cfg,
        enable_depth=args["enable_depth"],
        enable_semantic=args["enable_semantic"],
    )
    cube2equirect_tfms = configure_equirect_tfm(
        576,
        1024,  # genex model resolution
        sensor_uuids_dict=sensor_uuids_dict,
        enable_depth=args["enable_depth"],
        enable_semantic=args["enable_semantic"],
    )
    return cube2equirect_tfms


def search_sensor(sensor_specifications: List, key_words: List):
    for sensor in sensor_specifications:
        if all(word in sensor.uuid for word in key_words):
            return sensor

def configure_cubemap_sensors_sim(sim_cfg, enable_depth, enable_semantic=False):
    """
    Configures cubemap cameras and sets up equirectangular transformers.
    Args:
        meta_config (omegaconf.dictconfig.DictConfig): The Habitat configuration.
        enable_depth (bool): Whether to enable depth sensors.
    Returns:
        dict: Dictionary containing equirect transformers for RGB and depth.
    """
    s = default_settings
    agent_config = sim_cfg.agents[0]
    CAMERA_NUM = 6
    uuid_str = ["back", "down", "front", "right", "left", "up"]
    orient = [
        [0, math.pi, 0],  # Back
        [-math.pi / 2, 0, 0],  # Down
        [0, 0, 0],  # Front
        [0, math.pi / 2, 0],  # Right
        [0, 3 / 2 * math.pi, 0],  # Left
        [math.pi / 2, 0, 0],  # Up
    ]
    sensor_uuids_dict = {}
    spec = agent_config.sensor_specifications
    camera_rgb_temp = search_sensor(spec, ["rgb", "front"])
    camera_temp = {"rgb": camera_rgb_temp}
    if enable_depth:
        camera_depth_temp = search_sensor(spec, ["depth", "front"])
        camera_temp["depth"] = camera_depth_temp
    if enable_semantic:
        camera_semantic_temp = search_sensor(spec, ["semantic", "front"])
        camera_temp["semantic"] = camera_semantic_temp

    sensor_types = ["rgb"]
    if enable_depth:
        sensor_types.append("depth")
    if enable_semantic:
        sensor_types.append("semantic")
    for sensor_type in sensor_types:
        sensor_uuids = []
        for camera_id in range(CAMERA_NUM):
            camera_name = f"{sensor_type}_{uuid_str[camera_id]}"
            if sensor_type == "rgb":
                SensorType_ = SensorType.COLOR
            elif sensor_type == "depth":
                SensorType_ = SensorType.DEPTH
            elif sensor_type == "semantic":
                SensorType_ = SensorType.SEMANTIC
            camera_config = _create_sensor_spec(
                camera_name,
                SensorType_,
                s["sensor_hfov_cubemap"],
                s["sensor_height_cubemap"],
                s["sensor_width_cubemap"],
                s["sensor_position"],
                s["sensor_pitch"],
            )
            camera_config.orientation = orient[camera_id]
            # camera_config.uuid = camera_name
            # agent_config.sim_sensors[camera_name] = camera_config
            agent_config.sensor_specifications.append(camera_config)
            sensor_uuids.append(camera_config.uuid)
        sensor_uuids_dict[sensor_type] = sensor_uuids

    return sensor_uuids_dict


def get_observations(sim, position, rotation):
    set_agent_coordinates(sim, position, rotation)
    observations = sim.get_sensor_observations()

    for k, v in observations.items():
        if 'rgb' in k:
            observations[k] = rgba2rgb(v)
        elif 'semantic' in k:
            observations[k] = v.astype(np.int32)

    return observations


def set_agent_coordinates(sim, position, rotation):
    # 1) normalize inputs
    pos = np.array(position, dtype=np.float32)

    # 2) navmesh check
    if not sim.pathfinder.is_navigable(pos):
        return False

    # 3) build & apply new state
    agent_state = AgentState()
    agent_state.position = pos
    agent_state.rotation = np.quaternion(*rotation) # quaternion format: (w, x, y, z)

    agent = sim.get_agent(0)
    agent.set_state(agent_state, reset_sensors=False)
    return True


def display_from_observation(observation):
    display_sample(
        observation["rgb"],
        observation["semantic"],
        observation["depth"],
    )

def draw_target_bbox(
    sem_obs,
    rgb_obs,
    target_id,
    bbox_color,
    thickness=2,
    scale_factor=1.0
):
    """
    Draws a bounding box around the largest connected component of the target in sem_obs,
    then scales the box by the provided scale factor. If no region found, returns the original
    image and False.
    Args:
        sem_obs: 2D semantic observation (numpy array or torch.Tensor), shape [H, W].
        rgb_obs: The corresponding 3D RGB image (numpy array or torch.Tensor), shape [H, W, 3].
        target_id: The id of the target to draw the bounding box around.
        bbox_color: Color for the bounding box, e.g. (255, 0, 0).
        thickness: Thickness of the bounding box lines.
        scale_factor: Scale factor to adjust the bounding box size relative to its original.
    Returns:
        Tuple: (annotated_rgb, found_flag, box_coord)
            - annotated_rgb: The RGB image with the drawn bounding box (numpy array).
            - found_flag (bool): True if the target is found, False otherwise.
            - box_coord (dict): {"x_min", "x_max", "y_min", "y_max"} for the adjusted
                                bounding box, or None if not found.
    """
    # 1) Convert torch.Tensor -> numpy if needed
    if isinstance(sem_obs, torch.Tensor):
        sem_obs = sem_obs.cpu().numpy()
    if isinstance(rgb_obs, torch.Tensor):
        rgb_obs = rgb_obs.cpu().numpy()

    assert sem_obs.shape[:2] == rgb_obs.shape[:2], "sem_obs and rgb_obs must match in [H, W]"

    # 2) Generate a binary mask of the target_id
    mask = (sem_obs == target_id).astype(np.uint8) #mask.shape is [H, W, 1]

    # 3) Label connected components in the mask
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # If num_labels == 1 => only background found => no target region
    if num_labels <= 1:
        return rgb_obs, False, None

    # 4) Identify the largest connected component by area (exclude label 0 = background)
    largest_area = 0
    largest_label = -1
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = label_id

    # If still -1, something’s off => no valid region
    if largest_label < 1:
        return rgb_obs, False, None

    # 5) Extract bounding box for the largest component
    # stats row = [left, top, width, height, area]
    x_min = stats[largest_label, cv2.CC_STAT_LEFT]
    y_min = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    x_max = x_min + w - 1
    y_max = y_min + h - 1

    # 6) Scale the bounding box around its center
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    orig_w = x_max - x_min
    orig_h = y_max - y_min

    new_w = orig_w * scale_factor
    new_h = orig_h * scale_factor

    new_x_min = int(center_x - new_w / 2)
    new_x_max = int(center_x + new_w / 2)
    new_y_min = int(center_y - new_h / 2)
    new_y_max = int(center_y + new_h / 2)

    # 7) Draw bounding box on a copy of the original
    rgb_obs_with_bbox = rgb_obs.copy()
    cv2.rectangle(
        rgb_obs_with_bbox,
        (new_x_min, new_y_min),
        (new_x_max, new_y_max),
        bbox_color,
        thickness
    )
    box_coord = {
        "x_min": new_x_min,
        "x_max": new_x_max,
        "y_min": new_y_min,
        "y_max": new_y_max
    }

    return rgb_obs_with_bbox, True, box_coord

