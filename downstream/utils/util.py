import json
from datetime import datetime
import numpy as np
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_angle_axis
from data_filtering.pcd_reproject import habitat_rotation, habitat_translation

import quaternion
import math
from habitat_sim.utils.common import angle_between_quats
from typing import List, Tuple

# ================================For pathfinder======================

class ActionFinder:
    def __init__(self, sim, planner):
        self.sim = sim
        self.planner = planner
        self.curr_nav_pt = None

    def get_next_action_seq(self):
        # return the action to take
        # action = self.planner.get_next_action(self.curr_nav_pt)
        actions = self.planner.find_path(self.curr_nav_pt)
        self.clear_nav_pt()
        return actions

    def set_new_nav_pt(self, object_pos, object_radius):
        # set the new navigation point
        assert self.curr_nav_pt is None, "Navigation point already set."
        curr_pos = self.sim.get_agent(0).state.position
        max_search_radius = object_radius + 2.0
        nav_pts = get_nav_pts_with_pathfinder(
            object_pos, self.sim.pathfinder,
            height=curr_pos[1],
            max_search_radius=max_search_radius,
        )
        # find the closest navigation point
        nav_pt, path_len = self.find_best_nav_pt(curr_pos, nav_pts)
        if nav_pt is None:
            print("No valid navigation point found.")
            return False
        else:
            # set the new navigation point
            self.curr_nav_pt = nav_pt
            print(f"New navigation point set, length: {path_len:.2f} m")
            return True

    def clear_nav_pt(self):
        # clean the navigation point
        self.curr_nav_pt = None

    def find_best_nav_pt(self, curr_pos, nav_pts):
        pts_length = []

        for p2 in nav_pts:
            length, _ = get_distance(curr_pos, p2, self.sim.pathfinder)
            pts_length.append(length)

        # find the best navigation point (dist >0.3 and min dist):
        best_length = np.inf
        best_pt = None
        for i, length in enumerate(pts_length):
            if length > 0.3 and length < best_length:
                best_length = length
                best_pt = nav_pts[i]

        return best_pt, best_length


def get_nav_pts_with_pathfinder(target_point_habitat, pathfinder, height, max_search_radius=2.5, max_tries=500):
    """
    Samples navigable observation points near `target_point_habitat` within a fixed radius,
    and returns all unique points that satisfy the height constraint.
    Parameters:
        target_point_habitat: array-like of shape (3,)
            The target point in Habitat coordinates.
        pathfinder:
            The pathfinder instance used to locate navigable points.
        height: float
            The expected height for the navigable points.
        max_search_radius: float, optional
            The fixed search radius (in meters) within which points are sampled. Default is 2.5 m.
        max_total_tries: int, optional
            The total number of sampling attempts. Default is 200.
    Returns:
        List of valid navigable points (each a NumPy array in Habitat coordinates).
        Returns an empty list if no valid points are found.
    """
    max_valid_points_len = 100
    valid_points = []
    for try_count in range(1, max_tries + 1):
        try:
            sample_pt = pathfinder.get_random_navigable_point_near(
                circle_center=target_point_habitat,
                radius=max_search_radius
            )
        except Exception as e:
            if try_count % 10 == 0:
                print(f"Error on try {try_count}: {e}")
            continue

        if np.isnan(sample_pt).any():
            continue

        if abs(sample_pt[1] - height) < 0.2:
            # Avoid duplicates with a small tolerance.
            if not any(np.allclose(sample_pt, pt, atol=1e-3) for pt in valid_points):
                valid_points.append(sample_pt)
                if len(valid_points) >= max_valid_points_len:
                    break

    if not valid_points:
        print("Warning: No valid navigable points found within the maximum search radius.")
    return valid_points


def get_distance(p1, p2, pathfinder):
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(p1, dtype=np.float32)
    path.requested_end = np.array(p2, dtype=np.float32)
    found_path = pathfinder.find_path(path)

    if found_path:
        return path.geodesic_distance, path.points
    else:
        # if still not found, then return np.inf
        return np.inf, None

def calc_traj_distance(traj, pathfinder):
    total_distance = 0
    for i in range(1, len(traj)):
        distance, _ = get_distance(traj[i - 1], traj[i], pathfinder)
        total_distance += distance
    return total_distance


def compute_rot_difference(q_curr, q_goal):
    def to_q(q_xyzw):
        x, y, z, w = q_xyzw
        q = quaternion .quaternion(w, x, y, z)
        return q
    #     return q / np.linalg.norm(q)        # ensure unit length
    q_curr = to_q(q_curr)                # your current rotation
    q_goal = to_q(q_goal)   # the goal rotation

    azimuth_error = angle_between_quats(q_curr, q_goal)      # |Δψ| ∈ [0, π]
    return azimuth_error

# ================================For projection======================

def get_cam_extrinsic_from_rotate(agent_state, theta_dict):
    """
    Get the camera extrinsic matrix from the agent_state and rotation angles.
    Return:
        RTs: List of camera extrinsic matrices for each rotation angle.
    """
    RTs = []
    agent_position, agent_rotation = agent_state.position, agent_state.rotation
    for i, (_, theta) in enumerate(theta_dict.items()):
        quat_new = _rotate_yaw(agent_rotation, theta / 180 * np.pi)

        rotate = habitat_rotation(quat_new)
        position = habitat_translation(agent_position)
        RT = np.eye(4)
        RT[:3, :3] = rotate
        RT[:3, 3] = position
        RT[3, 3] = 1
        RTs.append(RT)

    return RTs


def action_number_to_polar_angle(
    chosen_action_id: str, action_choices: dict, polar_actions: dict[int, List]
) -> float:
    """Converts the chosen action number to its Polar instance"""
    for view_id, view_action_ids in action_choices.items():
        for i, action_id in enumerate(view_action_ids):
            if action_id == chosen_action_id:
                labeled_dist, theta = polar_actions[view_id][i]

    return -theta


def _rotate_yaw(curr_rotation, magnitude):
    """
    Rotate the agent_state by a specified angle.

    :param agent_state: The state of the agent to be updated.
    :param curr_rotation: Current rotation of the agent.
    :param magnitude: Angle in radians to rotate counterclockwise.
    """
    theta, axis = quat_to_angle_axis(curr_rotation)
    if axis[1] < 0:  # Ensure consistent rotation direction
        theta = 2 * np.pi - theta
    new_theta = theta + magnitude
    quat_new = quat_from_angle_axis(new_theta, np.array([0, 1, 0]))
    return quat_new


def rotate_and_forward_agent(curr_state, theta, magnitude):
    """
    Move the agent based on the specified rotation and forward action.
    """
    new_state = habitat_sim.AgentState()
    new_state.position = np.copy(curr_state.position)
    new_state.rotation = _rotate_yaw(curr_state.rotation, theta)

    new_state.position = forward_agent(new_state, magnitude)

    return new_state.position, new_state.rotation


def rotate_agent(agent, theta, set_state=True):
    """
    Move the agent based on the specified action and magnitude.

    :param action: The action to perform.
    """
    curr_state = agent.get_state()

    new_state = habitat_sim.AgentState()
    new_state.position = np.copy(curr_state.position)
    new_state.rotation = curr_state.rotation
    new_state.rotation = _rotate_yaw(new_state.rotation, theta)
    if set_state:
        agent.set_state(new_state)

    return new_state.rotation

def forward_agent(curr_state, magnitude):
    """
    Move the agent_state forward by a specified magnitude.

    :param agent_state: The state of the agent to be updated.
    :param curr_position: Current position of the agent.
    :param curr_rotation: Current rotation of the agent.
    :param magnitude: Distance to move forward.
    """
    curr_pos, curr_rot = curr_state.position, curr_state.rotation

    local_point = np.array([0, 0, -magnitude])
    global_point = local_to_global(curr_pos, curr_rot, local_point)

    return global_point


def generate_nav_path(polar_actions, agent_state, sensor_state, H, W, hfov):
    """
    Generate a navigation path (represent as start and end pixval in the image) for the agent in the environment.
    """
    paths = []
    start_px = agent_frame_to_image_coords(
        [0, 0, 0], agent_state, sensor_state,
        resolution=(H, W),
        hfov_degree=hfov
    )
    # end_pxs = []
    for _, (r_i, theta_i) in enumerate(polar_actions):
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]

        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state,
            resolution=(H, W),
            hfov_degree=hfov
        )
        paths.append((start_px, end_px))
    return paths


def calculate_focal_length(fov_degrees, image_width):
    """
    Calculates the focal length in pixels based on the field of view and image width.

    Args:
        fov_degrees (float): Field of view in degrees.
        image_width (int): The width of the image in pixels.

    Returns:
        float: The focal length in pixels.
    """
    fov_radians = np.deg2rad(fov_degrees)
    focal_length = (image_width / 2) / np.tan(fov_radians / 2)
    return focal_length

def agent_frame_to_image_coords(point, agent_state, sensor_state, resolution, hfov_degree):
    """
    Converts a point from agent frame to image coordinates.

    Args:
        point (np.ndarray): The point in agent frame coordinates.
        agent_state (6dof): The agent's state containing position and rotation.
        sensor_state (6dof): The sensor's state containing position and rotation.
        resolution (tuple): The image resolution as (height, width).
        hfov_degree (float): The horizontal field of view in degrees.

    Returns:
        tuple or None: The image coordinates (x_pixel, y_pixel), or None if the point is behind the camera.
    """
    focal_length = calculate_focal_length(hfov_degree, resolution[1])
    global_p = local_to_global(agent_state.position, agent_state.rotation, point)
    camera_pt = global_to_local(sensor_state.position, sensor_state.rotation, global_p)
    if camera_pt[2] > 0:
        return None
    return local_to_image(camera_pt, resolution, focal_length)


def local_to_image(local_point, resolution, focal_length):
    """
    Converts a local 3D point to image pixel coordinates.

    Args:
        local_point (np.ndarray): The point in local coordinates.
        resolution (tuple): The image resolution as (height, width).
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        tuple: The pixel coordinates (x_pixel, y_pixel).
    """
    point_3d = [local_point[0], -local_point[1], -local_point[2]]  # Inconsistency between Habitat camera frame and classical convention
    if point_3d[2] == 0:
        point_3d[2] = 0.0001
    x = focal_length * point_3d[0] / point_3d[2]
    x_pixel = int(resolution[1] / 2 + x)

    y = focal_length * point_3d[1] / point_3d[2]
    y_pixel = int(resolution[0] / 2 + y)
    return x_pixel, y_pixel


def local_to_global(position, orientation, local_point):
    """
    Transforms a local coordinate point to global coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        local_point (np.ndarray): The point in local coordinates.

    Returns:
        np.ndarray: Transformed global coordinates.
    """
    rotated_point = quaternion.rotate_vectors(orientation, local_point)
    global_point = rotated_point + position
    return global_point


def global_to_local(position, orientation, global_point):
    """
    Transforms a global coordinate point to local coordinates based on position and quaternion orientation.

    Args:
        position (np.ndarray): The global position.
        orientation (quaternion.quaternion): The quaternion representing the rotation.
        global_point (np.ndarray): The point in global coordinates.

    Returns:
        np.ndarray: Transformed local coordinates.
    """
    translated_point = global_point - position
    inverse_orientation = np.quaternion.conj(orientation)
    local_point = quaternion.rotate_vectors(inverse_orientation, translated_point)
    return local_point


# ======================================================
def output_is_none(var):
    if var is None:
        return True
    if isinstance(var, str):
        return var == "" or var.lower() == 'none'
    return False

def output_is_True(var):
    if var is True:
        return True
    if isinstance(var, str):
        return var.lower() == 'true'
    return False

def mask_semantic_by_target(target_id, obs):
    for key in obs.keys():
        if not key.startswith("semantic"):
            continue
        if key in (
            "semantic_front",
            "semantic_sensor",
            "semantic_right",
            "semantic_left",
            "semantic_up",
        ):
            obs[key][obs[key] != target_id] = 0
        else:
            obs[key] = np.zeros_like(obs[key])

    return obs


def visualize_semantic_img(sem_obs_raw):
    if sem_obs_raw.ndim == 3:
        sem_obs_raw = sem_obs_raw.unsqueeze(-1)
    # assert sem_obs_raw.ndim == 2, "semantic observation should be 2D"
    sem_obs = Image.new("P", (sem_obs_raw.shape[1], sem_obs_raw.shape[0]))
    sem_obs.putpalette(d3_40_colors_rgb.flatten())
    sem_obs.putdata((sem_obs_raw.flatten() % 40).tolist())
    sem_obs = sem_obs.convert("RGB")
    return np.array(sem_obs)


def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, "RGBA image has 4 channels."

    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype="float32") / 255.0
    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype="uint8")


def format_time(time):
    """ format eta and time elapsed """
    hours = int(time // 3600)
    minutes = int((time % 3600) // 60)
    seconds = int(time % 60)
    return f"{hours}h {minutes}m {seconds}s"

def log_metric(metrics_path, content):
    if content in ("start", "end"):
        with open(metrics_path, "a") as f:
            content = f"{'=' * 10} {content}: {datetime.now()} {'=' * 10}\n"
            f.write(content)
            f.flush()
    else:
        assert isinstance(content, dict)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(content, ensure_ascii=False) + "\n")
            f.flush()
