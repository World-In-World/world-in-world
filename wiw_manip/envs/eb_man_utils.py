import os
from typing import List
import numpy as np
from pyrep.objects import VisionSensor
from ultralytics import YOLO
import cv2
from scipy.spatial.transform import Rotation
from amsolver.backend.observation import Observation
from typing import Dict
from scipy.spatial.transform import Rotation
from typing import Optional, Sequence, Tuple

SCENE_BOUNDS = np.array([-0.3, -0.5, 0.6, 0.7, 0.5, 1.6])
ROTATION_RESOLUTION = 3
VOXEL_SIZE = 100
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
USE_GENERAL_OBJECT_NAMES = True
object_detection_model = YOLO("yolo11n.pt")

VALID_TASKS = [
    "slide_block_to_color_target",
    "insert_onto_square_peg",
    "push_buttons",
    "stack_cups",
]
DIFF_VALID_TASKS = [
    "slide_block_to_color_target",
    "insert_onto_square_peg",
    "push_buttons",
]

def get_interactive_objects_name(obs: Observation):
    waypoints = [s for s in obs.object_informations.keys() if 'waypoint' in s]
    interact_object_name = [obs.object_informations[waypoint]['target_obj_name'] for waypoint in waypoints]
    return list(dict.fromkeys(interact_object_name)) # remove duplicates and keep order.

# copied from wiw_manip/envs/eb_manipulation/EBManDP3Env.py
def extract_obs(obs: Observation, target_obj=None, objects_name=None, use_wrist=True) -> Dict[str, np.ndarray]:
    if objects_name is None:
        objects_name = get_interactive_objects_name(obs)
    if not use_wrist:
        rgb = obs.front_rgb # [H, W, 3]
        mask = obs.front_mask  # [H, W]
        pc = obs.front_point_cloud # [H, W, 3]
    else:
        rgb = np.concatenate([obs.front_rgb, obs.wrist_rgb, obs.overhead_rgb], axis=0) # [H, W, 3]
        mask = np.concatenate([obs.front_mask, obs.wrist_mask, obs.overhead_mask], axis=0)  # [H, W]
        pc = np.concatenate([obs.front_point_cloud, obs.wrist_point_cloud, obs.overhead_point_cloud], axis=0) # [H, W, 3]
    H, W, C = pc.shape
    mask_flat = mask.reshape(H*W)
    pc_flat = pc.reshape(H*W, C)  # [T, H*W, 3]
    rgb_flat = rgb.reshape(H*W, 3)  # [T, H*W, 3]
    pc_rgb_flat = np.concatenate([pc_flat, rgb_flat / 255.0], axis=-1)  # [T, H*W, 6]
    pc_sampled = np.zeros((2048, 6), dtype=np.float32)

    cond = (mask_flat > 80) | ((mask_flat > 35) & (mask_flat < 50))
    cond = cond & (SCENE_BOUNDS[0] < pc_flat[..., 0]) & (pc_flat[..., 0] < SCENE_BOUNDS[3]) \
                & (SCENE_BOUNDS[1] < pc_flat[..., 1]) & (pc_flat[..., 1] < SCENE_BOUNDS[4]) \
                & (SCENE_BOUNDS[2] < pc_flat[..., 2]) & (pc_flat[..., 2] < SCENE_BOUNDS[5])
    idx = np.where(cond)[0]
    if len(idx) >= 2048:
        sampled = np.random.choice(idx, 2048, replace=False)
    elif len(idx) > 0:
        sampled = np.random.choice(idx, 2048, replace=True)
    else:
        sampled = np.zeros(2048, dtype=int)
    pc_sampled = pc_rgb_flat[sampled]
    # agent_pos = obs.get_euler_gripper_pose()
    # agent_pos = np.concatenate([obs.gripper_pose, [obs.gripper_open]], axis=-1)
    agent_pos = np.concatenate([obs.gripper_pose, [obs.gripper_open]], axis=-1)
    action = np.concatenate([obs.gripper_pose, [obs.gripper_open]], axis=-1)
    if target_obj is not None:
        return {
            "agent_pos": agent_pos,
            "point_cloud": pc_sampled,
            "action": action,
            "target_obj": target_obj,
        }
    else:
        return {
            "agent_pos": agent_pos,
            "point_cloud": pc_sampled,
            "action": action,
        }

# From https://github.com/stepjam/RLBench/blob/master/rlbench/backend/utils.py
def point_to_voxel_index(
        point: np.ndarray):
    bb_mins = np.array(SCENE_BOUNDS[0:3])[None]
    bb_maxs = np.array(SCENE_BOUNDS[3:])[None]
    dims_m_one = np.array([VOXEL_SIZE] * 3)[None] - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([VOXEL_SIZE] * 3) + 1e-12)
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)

    return voxel_indicy.reshape(point.shape)

def discrete_euler_to_quaternion(discrete_euler):
    euluer = (discrete_euler * ROTATION_RESOLUTION) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()


def get_continous_action_from_discrete_batch(discrete_actions: List[List[int]]):
    return [get_continous_action_from_discrete(action) for action in discrete_actions]

def get_continous_action_from_discrete(discrete_action):
    """
    Converts a discrete action representation into a continuous action representation.

    :param discrete_action: A list or array containing discrete action values.
        - The first 3 values represent the translational indices in the voxel grid.
        - The next 3 values represent the rotational and gripper indices.
        - The last value represents whether the gripper is open (1) or closed (0).
    :return: A numpy array representing the continuous action, which includes:
        - Attention coordinates in continuous space (x, y, z).
        - Quaternion representing rotation (qx, qy, qz, qw).
        - Gripper state (open or closed).
    """
    # Assert that all elements in the discrete action are integers
    assert all(isinstance(x, (int, np.integer)) for x in discrete_action), "All elements in discrete_action must be integers"
    # Extract translational indices (x, y, z) from the discrete action
    trans_indicies = np.array(discrete_action[:3])

    # Calculate the resolution of each voxel in the scene bounds
    bounds = SCENE_BOUNDS
    res = (bounds[3:] - bounds[:3]) / VOXEL_SIZE

    # Convert translational indices to continuous attention coordinates
    attention_coordinate = bounds[:3] + res * trans_indicies + res / 2

    # Extract the gripper state (open or closed)
    is_gripper_open = discrete_action[-1]

    if len(discrete_action) == 7:
        # Extract rotational indices (roll, pitch, yaw) and gripper index
        rot_and_grip_indicies = np.array(discrete_action[3:6])

        # Convert discrete rotational indices to a quaternion
        quat = discrete_euler_to_quaternion(rot_and_grip_indicies)

        # Combine the continuous attention coordinates, quaternion, and gripper state
        continuous_action = np.concatenate([
            attention_coordinate,  # Continuous (x, y, z) position
            quat,                  # Quaternion (qx, qy, qz, qw)
            [is_gripper_open]      # Gripper state (1 for open, 0 for closed)
        ])
    elif len(discrete_action) == 4:
        # Combine the continuous attention coordinates, quaternion, and gripper state
        continuous_action = np.concatenate([
            attention_coordinate,  # Continuous (x, y, z) position
            [is_gripper_open]      # Gripper state (1 for open, 0 for closed)
        ])
    elif len(discrete_action) == 8: # for debug purpose,
        continuous_action = np.array(discrete_action)
    else:
        raise ValueError("Wrong length of discrete action")

    return continuous_action

def draw_xyz_coordinate(image_path, resolution):
    image = cv2.imread(image_path)
    postfix = os.path.splitext(image_path)[1]
    save_path = image_path.replace(postfix, f"_wCoord{postfix}")
    # origin = (45, 172)  # Adjust based on the table's position in the image
    if resolution == 500:
        origin = (62, 239)  # Adjust based on the table's position in the image

        # Define axis lengths
        axis_length = 30  # Length of each axis

        # Define colors for the axes
        color_x = (0, 0, 255)  # Red for X-axis
        color_y = (0, 255, 0)  # Green for Y-axis
        color_z = (255, 0, 0)  # Blue for Z-axis

        cv2.circle(image, (int(origin[0]), int(origin[1])), 3, (0, 0, 255), -1)

        cv2.putText(
                image,
                f"(0, 0)",  # Convert number to string
                (62, 255),        # Position for the text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                fontScale=0.3,        # Font scale
                color=(0, 0, 255),      # Black color for the text
                thickness=1,          # Thickness of the text
                lineType=cv2.LINE_AA  # Anti-aliased line
            )

        # Draw the Y-axis
        cv2.arrowedLine(image, origin, (origin[0] + axis_length, origin[1]), color_y, 2, tipLength=0.2)
        cv2.putText(image, "y", (origin[0] + axis_length, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_y, 2)

        # Draw the Z-axis
        cv2.arrowedLine(image, origin, (origin[0], origin[1] - axis_length), color_z, 2, tipLength=0.2)
        cv2.putText(image, "z", (origin[0] - 20, origin[1] - axis_length), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z, 2)

        # Draw the X-axis (diagonal for 3D representation)
        cv2.arrowedLine(image, origin, (origin[0] - axis_length + 12, origin[1] + axis_length), color_x, 2, tipLength=0.2)
        cv2.putText(image, "x", (origin[0] - axis_length, origin[1] + axis_length - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_x, 2)

        # Save the image with the axes
        cv2.imwrite(save_path, image)
    elif resolution == 300:
        origin = (38, 142)  # Adjust based on the table's position in the image

        # Define axis lengths
        axis_length = 30  # Length of each axis

        # Define colors for the axes
        color_x = (0, 0, 255)  # Red for X-axis
        color_y = (0, 255, 0)  # Green for Y-axis
        color_z = (255, 0, 0)  # Blue for Z-axis

        cv2.circle(image, (int(origin[0]), int(origin[1])), 3, (0, 0, 255), -1)

        cv2.putText(
                image,
                f"(0, 0)",  # Convert number to string
                (38, 158),        # Position for the text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                fontScale=0.3,        # Font scale
                color=(0, 0, 255),      # Black color for the text
                thickness=1,          # Thickness of the text
                lineType=cv2.LINE_AA  # Anti-aliased line
            )

        # Draw the Y-axis
        cv2.arrowedLine(image, origin, (origin[0] + axis_length, origin[1]), color_y, 2, tipLength=0.2)
        cv2.putText(image, "y", (origin[0] + axis_length, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_y, 2)

        # Draw the Z-axis
        cv2.arrowedLine(image, origin, (origin[0], origin[1] - axis_length), color_z, 2, tipLength=0.2)
        cv2.putText(image, "z", (origin[0] - 20, origin[1] - axis_length), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z, 2)

        # Draw the X-axis (diagonal for 3D representation)
        cv2.arrowedLine(image, origin, (origin[0] - axis_length + 12, origin[1] + axis_length), color_x, 2, tipLength=0.2)
        cv2.putText(image, "x", (origin[0] - axis_length, origin[1] + axis_length - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_x, 2)

        # Save the image with the axes (need to get the postfix and replace it with _wCoord.{postfix}
        postfix = os.path.splitext(image_path)[1]
        cv2.imwrite(save_path.replace(postfix, f"_wCoord{postfix}"), image)
    elif resolution == 700:
        origin = (88, 335)  # Adjust based on the table's position in the image

        # Define axis lengths
        axis_length = 50  # Length of each axis

        # Define colors for the axes
        color_x = (0, 0, 255)  # Red for X-axis
        color_y = (0, 255, 0)  # Green for Y-axis
        color_z = (255, 0, 0)  # Blue for Z-axis

        cv2.circle(image, (int(origin[0]), int(origin[1])), 3, (0, 0, 255), -1)

        cv2.putText(
                image,
                f"(0, 0)",  # Convert number to string
                (88, 355),        # Position for the text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                fontScale=0.3,        # Font scale
                color=(0, 0, 255),      # Black color for the text
                thickness=1,          # Thickness of the text
                lineType=cv2.LINE_AA  # Anti-aliased line
            )

        # Draw the Y-axis
        cv2.arrowedLine(image, origin, (origin[0] + axis_length, origin[1]), color_y, 2, tipLength=0.2)
        cv2.putText(image, "y", (origin[0] + axis_length, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_y, 2)

        # Draw the Z-axis
        cv2.arrowedLine(image, origin, (origin[0], origin[1] - axis_length), color_z, 2, tipLength=0.2)
        cv2.putText(image, "z", (origin[0] - 20, origin[1] - axis_length), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z, 2)

        # Draw the X-axis (diagonal for 3D representation)
        cv2.arrowedLine(image, origin, (origin[0] - axis_length + 20, origin[1] + axis_length), color_x, 2, tipLength=0.2)
        cv2.putText(image, "x", (origin[0] - axis_length, origin[1] + axis_length - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_x, 2)

        # Save the image with the axes
        cv2.imwrite(save_path, image)
    elif resolution == 360:
        origin = (45, 172)  # Adjust based on the table's position in the image

        # Define axis lengths
        axis_length = 30  # Length of each axis

        # Define colors for the axes
        color_x = (0, 0, 255)  # Red for X-axis
        color_y = (0, 255, 0)  # Green for Y-axis
        color_z = (255, 0, 0)  # Blue for Z-axis

        cv2.circle(image, (int(origin[0]), int(origin[1])), 3, (0, 0, 255), -1)

        cv2.putText(
                image,
                f"(0, 0)",  # Convert number to string
                (45, 188),        # Position for the text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                fontScale=0.3,        # Font scale
                color=(0, 0, 255),      # Black color for the text
                thickness=1,          # Thickness of the text
                lineType=cv2.LINE_AA  # Anti-aliased line
            )

        # Draw the Y-axis
        cv2.arrowedLine(image, origin, (origin[0] + axis_length, origin[1]), color_y, 2, tipLength=0.2)
        cv2.putText(image, "y", (origin[0] + axis_length, origin[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_y, 2)

        # Draw the Z-axis
        cv2.arrowedLine(image, origin, (origin[0], origin[1] - axis_length), color_z, 2, tipLength=0.2)
        cv2.putText(image, "z", (origin[0] - 20, origin[1] - axis_length), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z, 2)

        # Draw the X-axis (diagonal for 3D representation)
        cv2.arrowedLine(image, origin, (origin[0] - axis_length + 12, origin[1] + axis_length), color_x, 2, tipLength=0.2)
        cv2.putText(image, "x", (origin[0] - axis_length, origin[1] + axis_length - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_x, 2)

        # Save the image with the axes
        cv2.imwrite(save_path, image)
    else:
        ValueError("Detection boxes are not supported for this resolution. Please disable detection boxes or use a valid resolution.")

    return save_path

def increase_bbox(bbox, scale_factor=1):
    """
    Increase the bounding box size by a scale factor.

    :param bbox: A list of coordinates [x1, y1, x2, y2] for the bounding box
    :param scale_factor: Factor by which to increase the bounding box size
    :return: A list of coordinates [x1, y1, x2, y2] for the scaled bounding box
    """
    x1, y1, x2, y2 = bbox

    # Calculate the original width and height
    original_width = x2 - x1
    original_height = y2 - y1

    # Compute the center of the bounding box
    center_x = x1 + original_width / 2
    center_y = y1 + original_height / 2

    # Calculate the new width and height
    new_width = original_width * scale_factor
    new_height = original_height * scale_factor

    # Compute the new bounding box coordinates
    new_x1 = int(center_x - new_width / 2)
    new_y1 = int(center_y - new_height / 2)
    new_x2 = int(center_x + new_width / 2)
    new_y2 = int(center_y + new_height / 2)

    return [new_x1, new_y1, new_x2, new_y2]

def project_world_points_to_image(world_points, camera_extrinsics, camera_intrinsics):
    """
    将世界坐标点投影到图像平面上。
    """
    T_inv = np.linalg.inv(camera_extrinsics)
    rvec = T_inv[:3, :3]
    tvec = T_inv[:3, 3]
    pixel_points_2D, _ = cv2.projectPoints(np.array(world_points), rvec, tvec, camera_intrinsics, np.zeros(4))
    return pixel_points_2D

def annotate_image_with_boxes(input_image_path, pixel_points_2D):
    """
    在图像上根据 projected pixel points 和目标检测框进行绘制与编号，并保存图像。
    """
    # 读取图像并检测目标框
    image_bgr = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    results = object_detection_model.predict(source=input_image_path, conf=0.0001, line_width=1, verbose=False)
    predicted_boxes = results[0].boxes.xyxy
    # out_file = draw_points_on_image(input_image_path, pixel_points_2D)

    box_id = 0
    text_positions = []

    for point in pixel_points_2D:
        x, y = point[0]
        min_dist = float('inf')
        min_idx = -1
        for i, box in enumerate(predicted_boxes):
            center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]
            dist = (center[0] - x) ** 2 + (center[1] - y) ** 2
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_dist > 400:
            continue

        increased_box = increase_bbox(predicted_boxes[min_idx], 1.2)
        center_pixel = (0, 0, 255)

        cv2.rectangle(
            image_bgr,
            (int(increased_box[0]), int(increased_box[1])),
            (int(increased_box[2]), int(increased_box[3])),
            center_pixel,
            1
        )

        text_position = (int(increased_box[0]) + 20, int(increased_box[1]) - 10)
        for pos in text_positions:
            if abs(pos[0] - text_position[0]) < 10 and abs(pos[1] - text_position[1]) < 10:
                text_position = (int(increased_box[0]) + 10, int(increased_box[1]) - 10)
        text_positions.append(text_position)

        cv2.putText(
            image_bgr,
            str(box_id + 1),
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            center_pixel,
            1,
            cv2.LINE_AA
        )
        box_id += 1

    # 保存图像
    base, ext = os.path.splitext(input_image_path)
    image_save_path = f"{base}_annotated{ext}"
    cv2.imwrite(image_save_path, image_bgr)

    return image_save_path

def draw_bounding_boxes(image_path_list, world_points, camera_extrinsics_list, camera_intrinsics_list):
    """
    主函数：遍历每张图，先投影，再绘制，再保存。
    """
    image_save_path_list = []

    for input_image_path, camera_extrinsics, camera_intrinsics in zip(image_path_list, camera_extrinsics_list, camera_intrinsics_list):
        pixel_points_2D = project_world_points_to_image(world_points, camera_extrinsics, camera_intrinsics)
        # image_save_path = annotate_image_with_boxes(input_image_path, pixel_points_2D)
        image_save_path = draw_points_on_image(input_image_path, pixel_points_2D)
        image_save_path_list.append(image_save_path)

    return image_save_path_list

####### Generate object information for the initial observation
def _get_mask_id_to_name_dict_for_input(object_info):
    mask_id_to_name_dict = {}
    for obj in object_info:
        if 'id' in object_info[obj]:
            mask_id_to_name_dict[object_info[obj]['id']] = obj
    return mask_id_to_name_dict

def _get_point_cloud_dict_for_input(obs, camera_types):
    # This function gets the point cloud using the same operations as PerAct Colab Tutorial
    point_cloud_dict = {}
    camera_extrinsics_list, camera_intrinsics_list = [], []
    for camera_type in CAMERAS:
        cam_extrinsics = obs['misc'][f"{camera_type}_camera_extrinsics"]
        cam_intrinsics = obs['misc'][f"{camera_type}_camera_intrinsics"]
        if camera_type + "_rgb" in camera_types:
            camera_extrinsics_list.append(cam_extrinsics)
            camera_intrinsics_list.append(cam_intrinsics)
        cam_depth = obs[f"{camera_type}_depth"]
        near = obs['misc'][f"{camera_type}_camera_near"]
        far = obs['misc'][f"{camera_type}_camera_far"]
        cam_depth = (far - near) * cam_depth + near
        point_cloud_dict[camera_type] = VisionSensor.pointcloud_from_depth_and_camera_params(cam_depth, cam_extrinsics, cam_intrinsics) # reconstructed 3D point cloud in world coordinate frame

    return point_cloud_dict, camera_extrinsics_list, camera_intrinsics_list

def _get_mask_dict_for_input(obs):
    mask_dict = {}
    for camera in CAMERAS:
        rgb_mask = np.array(obs[f"{camera}_mask"], dtype=int)
        mask_dict[camera] = rgb_mask
    return mask_dict

def form_obs_for_input(
    mask_dict,
    mask_id_to_real_name,
    point_cloud_dict):

    # convert object id to char and average and discretize point cloud per object
    uniques = np.unique(np.concatenate(list(mask_dict.values()), axis=0))
    real_name_to_avg_coord = {}
    all_avg_point_list = []
    for _, mask_id in enumerate(uniques):
        if mask_id not in mask_id_to_real_name:
            continue
        avg_point_list = []
        for camera in CAMERAS:
            mask = mask_dict[camera]
            point_cloud = point_cloud_dict[camera]
            if not np.any(mask == mask_id):
                continue
            avg_point_list.append(np.mean(point_cloud[mask == mask_id].reshape(-1, 3), axis = 0))

        avg_point = sum(avg_point_list) / len(avg_point_list)
        all_avg_point_list.append(avg_point)
        real_name = mask_id_to_real_name[mask_id]
        real_name_to_avg_coord[real_name] = list(point_to_voxel_index(avg_point))
    if USE_GENERAL_OBJECT_NAMES:
        implicit_name_to_avg_coord = {}
        i = 1
        for key, value in real_name_to_avg_coord.items():
            implicit_name_to_avg_coord[f"object {i}"] = value
            # implicit_name_to_avg_coord[f"{i}"] = value
            i += 1
        real_name_to_avg_coord = implicit_name_to_avg_coord

    # Sort the objects based on the y-coordinate
    sorted_indices = sorted(range(len(all_avg_point_list)), key=lambda i: all_avg_point_list[i][1])
    all_avg_point_list = [all_avg_point_list[i] for i in sorted_indices]

    # Sort the objects in the general name based on the same order
    real_name_to_avg_coord = sorted(real_name_to_avg_coord.items(), key=lambda item: item[1][1])
    real_name_to_avg_coord = {f'object {i+1}': value for i, (_, value) in enumerate(real_name_to_avg_coord)}

    return real_name_to_avg_coord, all_avg_point_list

def form_object_coord_for_input(obs, task_class, camera_types):
    mask_id_to_sim_name = _get_mask_id_to_name_dict_for_input(obs['object_informations'])
    point_cloud_dict, camera_extrinsics_list, camera_intrinsics_list = _get_point_cloud_dict_for_input(obs, camera_types)
    mask_dict = _get_mask_dict_for_input(obs)

    task_handler = TASK_HANDLERS[task_class]()
    sim_name_to_real_name = task_handler.sim_name_to_real_name
    mask_id_to_real_name = {mask_id: sim_name_to_real_name[name] for mask_id, name in mask_id_to_sim_name.items()
                        if name in sim_name_to_real_name}
    avg_coord, all_avg_point_list = form_obs_for_input(mask_dict, mask_id_to_real_name, point_cloud_dict)
    return avg_coord, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list

def draw_points_on_image(
    input_image_path: str,
    pixel_points_2D: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    point_radius: int = 4,
    label_color: Tuple[int, int, int] = (255, 255, 255),
    label_bg_color: Tuple[int, int, int] = (0, 0, 0),
    out_path: Optional[str] = None,
    # New explicit styling controls for points:
    point_fill_color: Tuple[int, int, int] = (255, 255, 255),  # white center (BGR)
    point_edge_color: Tuple[int, int, int] = (40, 40, 40),        # black ring (BGR)
    edge_thickness: int = 2,                                   # ring width in px
) -> str:
    """
    Load an image and overlay 2D pixel points (x, y) onto it as white disks with a black ring.
    Supports input shapes (N, 2) or (N, 1, 2). Coordinates are (x, y) in pixels.

    Args:
        input_image_path: Path to the input image.
        pixel_points_2D: Array of shape (N, 2) or (N, 1, 2) with float/int (x, y).
        labels: Optional list/sequence of labels for each point. Defaults to 1-based indices.
        point_radius: Outer radius of the point marker in pixels.
        thickness: Kept for backward-compatibility (not used when drawing ring + fill).
        color: Kept for backward-compatibility (not used when drawing ring + fill).
        label_color: BGR color for text (foreground).
        label_bg_color: BGR color for text outline.
        out_path: Optional output path. If None, appends "_points" before extension.
        point_fill_color: BGR color for the marker center (default white).
        point_edge_color: BGR color for the marker ring (default black).
        edge_thickness: Ring thickness in pixels.

    Returns:
        Path to the saved annotated image.
    """
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_image_path}")
    h, w = img.shape[:2]

    pts = np.asarray(pixel_points_2D, dtype=np.float64)

    # Normalize shape to (N, 2)
    if pts.ndim == 3 and pts.shape[-2:] == (1, 2):
        pts = np.squeeze(pts, axis=-2)
    if pts.ndim != 2 or pts.shape[-1] != 2:
        raise ValueError("pixel_points_2D must have shape (N, 2) or (N, 1, 2).")

    # Keep only finite points
    finite_mask = np.isfinite(pts).all(axis=1)
    idx_all = np.arange(len(pts))
    pts = pts[finite_mask]
    idx_all = idx_all[finite_mask]

    # Round to nearest integer pixel coordinates
    pts_int = np.rint(pts).astype(int)

    # Keep points inside image bounds
    inside = (pts_int[:, 0] >= 0) & (pts_int[:, 0] < w) & (pts_int[:, 1] >= 0) & (pts_int[:, 1] < h)
    pts_int = pts_int[inside]
    kept_idx = idx_all[inside]

    # Draw points (white center + black ring) and labels (white text + black outline)
    for k, (x, y) in enumerate(pts_int):
        x_i, y_i = int(x), int(y)

        # Draw outer black ring
        cv2.circle(
            img, (x_i, y_i), point_radius, point_edge_color, thickness=edge_thickness, lineType=cv2.LINE_AA
        )
        # Draw inner white fill
        inner_r = max(1, point_radius - max(1, edge_thickness // 2 + 1))
        cv2.circle(
            img, (x_i, y_i), inner_r, point_fill_color, thickness=-1, lineType=cv2.LINE_AA
        )

        # Label with outline for contrast
        label = str(labels[kept_idx[k]]) if labels is not None else str(kept_idx[k] + 1)
        org = (x_i + 6, y_i - 6)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_bg_color, 3, cv2.LINE_AA)
        cv2.putText(img, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA)

    if out_path is None:
        base, ext = os.path.splitext(input_image_path)
        out_path = f"{base}_annotated{ext}"
    cv2.imwrite(out_path, img)
    print("==> Saved:", out_path)
    return out_path


class base_task_handler:
    def __init__(self, sim_name_to_real_name):
        self.sim_name_to_real_name = sim_name_to_real_name

class pick_cube_shape(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "small_container0": "first container",
            "small_container1": "second container",
            "star_normal_visual0": "first star",
            "star_normal_visual1": "second star",
            "cylinder_normal0": "first cylinder",
            "cylinder_normal1": "second cylinder",
            "triangular_normal0": "first triangular",
            "triangular_normal1": "second triangular",
            "cube_basic0": "first cube",
            "cube_basic1": "second cube",
            "moon_normal_visual0": "first moon",
            "moon_normal_visual1": "second moon",
        }
        super().__init__(sim_name_to_real_name)

class stack_cubes_color(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "star_normal_visual0": "first star",
            "star_normal_visual1": "second star",
            "star_normal_visual2": "third star",
            "star_normal_visual3": "fourth star",
            "cylinder_normal0": "first cylinder",
            "cylinder_normal1": "second cylinder",
            "cylinder_normal2": "third cylinder",
            "cylinder_normal3": "fourth cylinder",
            "triangular_normal0": "first triangular",
            "triangular_normal1": "second triangular",
            "triangular_normal2": "third triangular",
            "triangular_normal3": "fourth triangular",
            "cube_basic0": "first cube",
            "cube_basic1": "second cube",
            "cube_basic2": "third cube",
            "cube_basic3": "fourth cube",
            "moon_normal_visual0": "first moon",
            "moon_normal_visual1": "second moon",
            "moon_normal_visual2": "third moon",
            "moon_normal_visual3": "fourth moon",
        }
        super().__init__(sim_name_to_real_name)

class place_into_shape_sorter_color(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "star_normal_visual0": "first star",
            "star_normal_visual1": "second star",
            "star_normal_visual2": "third star",
            "star_normal_visual3": "fourth star",
            "cylinder_normal0": "first cylinder",
            "cylinder_normal1": "second cylinder",
            "cylinder_normal2": "third cylinder",
            "cylinder_normal3": "fourth cylinder",
            "triangular_normal0": "first triangular",
            "triangular_normal1": "second triangular",
            "triangular_normal2": "third triangular",
            "triangular_normal3": "fourth triangular",
            "cube_basic0": "first cube",
            "cube_basic1": "second cube",
            "cube_basic2": "third cube",
            "cube_basic3": "fourth cube",
            "moon_normal_visual0": "first moon",
            "moon_normal_visual1": "second moon",
            "moon_normal_visual2": "third moon",
            "moon_normal_visual3": "fourth moon",
            "shape_sorter_visual": "shape sorter"
        }
        super().__init__(sim_name_to_real_name)

class wipe_table_shape(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "rectangle": "first rectangle area",
            "rectangle0": "second rectangle area",
            "round": "first round area",
            "round0": "second round area",
            "triangle": "first triangle area",
            "triangle0": "second triangle area",
            "star": "first star area",
            "star0": "second star area",
            "sponge_visual0": "sponge",
        }
        super().__init__(sim_name_to_real_name)

class slide_block_to_color_target(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "target1": "first plane",
            "target2": "second plane",
            "target3": "third plane",
            "target4": "fourth plane",
            "block": "block",
        }
        super().__init__(sim_name_to_real_name)

class place_shape_in_shape_sorter(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "shape_sorter": "shape sorter",
            "star": "star",
            "moon": "moon",
            "triangular_prism": "triangular",
            "cube": "cube",
            "cylinder": "cylinder",

            # "shape_sorter_visual": "shape sorter visual",
            "star_visual": "star visual",
            "moon_visual": "moon visual",
            "triangular_prism_visual": "triangular visual",
            "cube_visual": "cube visual",
            "cylinder_visual": "cylinder visual",
        }
        super().__init__(sim_name_to_real_name)

class push_buttons(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "push_buttons_target0": "first button",
            "push_buttons_target1": "second button",
            "push_buttons_target2": "third button",
        }
        super().__init__(sim_name_to_real_name)

class stack_cups(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "cup1_visual": "first cup visual",
            "cup2_visual": "second cup visual",
            "cup3_visual": "third cup visual",

            "cup1": "first cup",
            "cup2": "second cup",
            "cup3": "third cup",
        }
        super().__init__(sim_name_to_real_name)

class insert_onto_square_peg(base_task_handler):
    def __init__(self):
        sim_name_to_real_name = {
            "square_ring": "square ring",
            "pillar0": "first square peg",
            "pillar1": "second square peg",
            "pillar2": "third square peg",
        }
        super().__init__(sim_name_to_real_name)

TASK_HANDLERS = {
    'pick': pick_cube_shape,
    'stack': stack_cubes_color,
    'place': place_into_shape_sorter_color,
    'wipe': wipe_table_shape,
    'slide_block_to_color_target': slide_block_to_color_target,
    'place_shape_in_shape_sorter': place_shape_in_shape_sorter,
    'push_buttons': push_buttons,
    'stack_cups': stack_cups,
    'insert_onto_square_peg': insert_onto_square_peg,
    # 'place_into_shape_sorter_color': place_into_shape_sorter_color,
    # 'push_buttons': place_into_shape_sorter_color,
}
