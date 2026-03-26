import io
import logging
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from PIL import Image
from scipy.spatial.transform import Rotation
import cv2

logger = logging.getLogger("libero_env")

LIBERO_SCENE_BOUNDS = np.array([-0.35, -0.35, -0.05, 0.35, 0.35, 0.85])
LIBERO_VALID_TASKS = ["libero_object", "libero_spatial"]
OSC_POS_OUTPUT_MAX = np.array([0.05, 0.05, 0.05], dtype=np.float64)
OSC_ORI_OUTPUT_MAX = np.array([0.5, 0.5, 0.5], dtype=np.float64)
SUBSTEP_POS_CLIP = np.array([0.02, 0.02, 0.02], dtype=np.float64)
SUBSTEP_ORI_CLIP = np.array([0.2, 0.2, 0.2], dtype=np.float64)
ACTION_REPEAT_MAX = 100
ACTION_PROGRESS_INTERVAL = 10
POSE_CONVERGE_POS_TOL_M = 0.005
POSE_CONVERGE_ORI_TOL_RAD = 0.10
POSE_STALL_CHECK_START_REPEAT = 30
POSE_STALL_POS_DELTA_TOL_M = 0.001
POSE_STALL_ORI_DELTA_TOL_RAD = 0.02


REPO_ROOT = Path(__file__).resolve().parents[4]
LIBERO_THIRD_PARTY_ROOT = REPO_ROOT / "third_party" / "libero"
LIBERO_PACKAGE_ROOT = LIBERO_THIRD_PARTY_ROOT / "libero" / "libero"
LIBERO_CONFIG_ROOT = REPO_ROOT / ".cache" / "libero"


def _ensure_libero_local_config():
    LIBERO_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    config_file = LIBERO_CONFIG_ROOT / "config.yaml"
    if config_file.exists():
        return
    config = {
        "benchmark_root": str(LIBERO_PACKAGE_ROOT),
        "bddl_files": str(LIBERO_PACKAGE_ROOT / "bddl_files"),
        "init_states": str(LIBERO_PACKAGE_ROOT / "init_files"),
        "datasets": str(LIBERO_THIRD_PARTY_ROOT / "datasets"),
        "assets": str(LIBERO_PACKAGE_ROOT / "assets"),
    }
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


os.environ.setdefault("LIBERO_CONFIG_PATH", str(LIBERO_CONFIG_ROOT))
_ensure_libero_local_config()
if str(LIBERO_THIRD_PARTY_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_THIRD_PARTY_ROOT))

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark_dict  # noqa: E402
from libero.libero.envs.env_wrapper import SegmentationRenderEnv  # noqa: E402


def _mujoco_camera_pose_to_extrinsic(camera_pos, camera_xmat):
    rotation_world_from_mujoco = np.asarray(camera_xmat, dtype=np.float64).reshape(3, 3)
    mujoco_to_cv = np.diag([1.0, -1.0, -1.0])
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation_world_from_mujoco @ mujoco_to_cv
    extrinsic[:3, 3] = np.asarray(camera_pos, dtype=np.float64)
    return extrinsic


def _camera_intrinsics_from_fovy(fovy_deg, width, height):
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _project_world_points_to_image(world_points, camera_extrinsics, camera_intrinsics):
    t_inv = np.linalg.inv(camera_extrinsics)
    rvec, _ = cv2.Rodrigues(t_inv[:3, :3])
    tvec = t_inv[:3, 3]
    pixel_points_2d, _ = cv2.projectPoints(np.array(world_points), rvec, tvec, camera_intrinsics, np.zeros(4))
    return pixel_points_2d


def _flip_projected_points_vertical(pixel_points_2d, height):
    flipped = np.asarray(pixel_points_2d, dtype=np.float64).copy()
    flipped[..., 1] = (height - 1) - flipped[..., 1]
    return flipped


def _segmentation_key_for_camera(obs, camera_name):
    candidates = [key for key in obs.keys() if camera_name in key and "segmentation" in key]
    preferred_order = [
        f"{camera_name}_segmentation_instance",
        f"{camera_name}_segmentation",
        f"{camera_name}_image_segmentation_instance",
        f"{camera_name}_image_segmentation",
    ]
    for preferred in preferred_order:
        if preferred in obs:
            return preferred
    if candidates:
        return sorted(candidates)[0]
    raise KeyError(f"Could not find a segmentation observation for camera '{camera_name}'.")


def _normalize_segmentation_image(segmentation_image):
    seg = np.asarray(segmentation_image)
    if seg.ndim == 3:
        if seg.shape[-1] == 1:
            seg = seg[..., 0]
        else:
            seg = seg[..., -1]
    return seg.astype(np.int32)


def _largest_component_centroid(mask: np.ndarray):
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return None
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(component_areas)) + 1
    cx, cy = centroids[largest_idx]
    return np.array([cx, cy], dtype=np.float64)


class LocalLiberoEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(
        self,
        eval_set,
        render_mode="human",
        img_size=(500, 500),
        down_sample_ratio=1.0,
        log_path=None,
        selected_indexes=None,
        enable_path_obs=False,
        exp_name=None,
        max_step=15,
        dataset_root=None,
        action_repeat=50,
    ):
        del selected_indexes, enable_path_obs, exp_name, dataset_root
        self._render_mode = render_mode
        self.camera_height = img_size[0]
        self.camera_width = img_size[1]
        self._max_episode_steps = max_step
        self.action_repeat = max(1, int(action_repeat))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,))
        self.camera_name_map = {
            "front_rgb": "agentview",
            "wrist_rgb": "robot0_eye_in_hand",
        }
        self.env = None
        self.benchmark = None
        self.task_suite_name = eval_set
        self.scene_bounds = LIBERO_SCENE_BOUNDS.copy()
        self._current_task_id = None
        self._current_init_state_idx = None
        self._current_task = None
        self._current_init_state = None
        self._current_bddl_path = None
        self._current_object_entries = []
        self.dataset = []
        self.current_task_variation = None
        self.log_path = log_path
        self.number_of_episodes = 0
        self._reset = False
        self._current_episode_num = 0
        self._current_step = 0
        self._episode_start_time = 0
        self.episode_log = []
        self.episode_language_instruction = ""
        self.last_frame_obs = None
        self.task_class = eval_set
        self.init_dataset_and_tasks(eval_set, down_sample_ratio, log_path)

    def _create_env_for_task(self, task):
        bddl_file = os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        )
        if self.env is not None:
            self.env.close()
        self.env = SegmentationRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=self.camera_height,
            camera_widths=self.camera_width,
            camera_segmentations="instance",
        )
        self._current_bddl_path = bddl_file

    def init_dataset_and_tasks(self, eval_task, down_sample_ratio, log_path, selected_indexes=None):
        del selected_indexes
        assert eval_task in LIBERO_VALID_TASKS
        self.task_class = eval_task
        self.task_suite_name = eval_task
        self.benchmark = get_benchmark_dict()[eval_task]()
        task_init_states = [
            self.benchmark.get_task_init_states(task_id)
            for task_id in range(self.benchmark.n_tasks)
        ]
        # Interleave episodes across task ids so early sampled episodes are
        # diverse (avoids repeatedly seeing the same instruction when downsampling).
        flattened_episodes = []
        max_init_len = max((len(states) for states in task_init_states), default=0)
        for init_state_idx in range(max_init_len):
            for task_id, init_states in enumerate(task_init_states):
                if init_state_idx < len(init_states):
                    flattened_episodes.append((task_id, init_state_idx))
        if down_sample_ratio < 1.0:
            num_ep = max(int(len(flattened_episodes) * down_sample_ratio), 1)
            flattened_episodes = flattened_episodes[:num_ep]
        self.dataset = flattened_episodes
        self.log_path = log_path
        self.number_of_episodes = len(self.dataset)
        self.current_task_variation = None
        self._reset = False
        self._current_episode_num = 0
        self._current_step = 0
        self._episode_start_time = 0
        self.episode_log = []
        self.episode_language_instruction = ""
        self.last_frame_obs = None

    def _gripper_open_from_obs(self, obs):
        qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
        return float(np.mean(np.abs(qpos)) > 0.01)

    def _format_observation(self, obs):
        formatted = dict(obs)
        formatted["gripper_pose"] = np.concatenate(
            [formatted["robot0_eef_pos"], formatted["robot0_eef_quat"]]
        )
        formatted["gripper_open"] = self._gripper_open_from_obs(formatted)
        return formatted

    def _compute_scene_bounds(self):
        object_positions = [entry["position"] for entry in self.get_prompt_object_entries()]
        if self.last_frame_obs is not None:
            object_positions.append(np.asarray(self.last_frame_obs["robot0_eef_pos"], dtype=np.float64))
        if not object_positions:
            self.scene_bounds = LIBERO_SCENE_BOUNDS.copy()
            return
        stacked = np.stack(object_positions, axis=0)
        mins = stacked.min(axis=0) - np.array([0.12, 0.12, 0.02], dtype=np.float64)
        maxs = stacked.max(axis=0) + np.array([0.12, 0.12, 0.30], dtype=np.float64)
        mins[2] = min(mins[2], -0.05)
        maxs[2] = max(maxs[2], 0.55)
        self.scene_bounds = np.concatenate([mins, maxs], axis=0)

    def get_prompt_object_entries(self):
        if self.env is None:
            return []
        entries = []
        objects_dict = self.env.env.objects_dict
        for object_name in objects_dict.keys():
            if object_name not in self.env.env.obj_body_id:
                continue
            position = np.asarray(
                self.env.env.sim.data.body_xpos[self.env.env.obj_body_id[object_name]],
                dtype=np.float64,
            )
            entries.append({"name": object_name, "position": position})
        entries.sort(key=lambda item: (item["position"][1], item["position"][0], item["position"][2]))
        self._current_object_entries = entries
        return entries

    def get_camera_params(self, camera_type):
        if self.env is None:
            return None
        camera_name = self.camera_name_map[camera_type]
        camera_id = self.env.sim.model.camera_name2id(camera_name)
        camera_pos = self.env.sim.data.cam_xpos[camera_id]
        camera_xmat = self.env.sim.data.cam_xmat[camera_id]
        camera_fovy = self.env.sim.model.cam_fovy[camera_id]
        extrinsic = _mujoco_camera_pose_to_extrinsic(camera_pos, camera_xmat)
        intrinsic = _camera_intrinsics_from_fovy(camera_fovy, self.camera_width, self.camera_height)
        return {"extrinsic": extrinsic, "intrinsic": intrinsic}

    def reset(self):
        assert self._current_episode_num <= self.number_of_episodes, "All episodes have been completed."
        self._current_step = 0
        self._current_episode_num += 1
        self._reset = True
        self.episode_log = []
        self._episode_start_time = time.time()
        logger.info("%d episodes in dataset", len(self.dataset))
        logger.info("Resetting episode %d ...", self._current_episode_num)

        task_id, init_state_idx = self.dataset[self._current_episode_num - 1]
        task = self.benchmark.get_task(task_id)
        if self.env is None or task_id != self._current_task_id:
            self._create_env_for_task(task)
        self._current_task_id = task_id
        self._current_init_state_idx = init_state_idx
        self._current_task = task
        init_states = self.benchmark.get_task_init_states(task_id)
        self._current_init_state = init_states[init_state_idx]
        self.current_task_variation = f"{self.task_class}_variation{task_id}"
        self.episode_language_instruction = task.language

        _ = self.env.reset()
        obs = self.env.set_init_state(self._current_init_state)
        for _ in range(10):
            obs, _, _, _ = self.env.step(np.zeros(7, dtype=np.float32))

        self.last_frame_obs = self._format_observation(obs)
        self.get_prompt_object_entries()
        self._compute_scene_bounds()
        return self.episode_language_instruction, self.last_frame_obs

    def _change_action_format(self, action):
        action = np.asarray(action, dtype=np.float64)
        current_pos = np.asarray(self.last_frame_obs["robot0_eef_pos"], dtype=np.float64)
        current_quat = np.asarray(self.last_frame_obs["robot0_eef_quat"], dtype=np.float64)
        target_pos = action[:3]
        target_quat = np.asarray(action[3:7], dtype=np.float64)

        if np.linalg.norm(current_quat) < 1e-8:
            current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            current_quat = current_quat / np.linalg.norm(current_quat)

        if np.linalg.norm(target_quat) < 1e-8:
            target_quat = current_quat.copy()
        else:
            target_quat = target_quat / np.linalg.norm(target_quat)

        # VLM predicts an absolute target pose in visual-voxel coordinates. The
        # OSC_POSE controller underneath LIBERO expects normalized delta actions
        # in [-1, 1], which are then internally scaled to meter / radian ranges
        # defined in robosuite's osc_pose config.
        position_delta = np.clip(target_pos - current_pos, -SUBSTEP_POS_CLIP, SUBSTEP_POS_CLIP)
        current_rot = Rotation.from_quat(current_quat)
        target_rot = Rotation.from_quat(target_quat)
        delta_axis_angle = (target_rot * current_rot.inv()).as_rotvec()
        delta_axis_angle = np.clip(delta_axis_angle, -SUBSTEP_ORI_CLIP, SUBSTEP_ORI_CLIP)

        normalized_pos_delta = np.divide(
            position_delta,
            OSC_POS_OUTPUT_MAX,
            out=np.zeros_like(position_delta),
            where=OSC_POS_OUTPUT_MAX > 1e-12,
        )
        normalized_ori_delta = np.divide(
            delta_axis_angle,
            OSC_ORI_OUTPUT_MAX,
            out=np.zeros_like(delta_axis_angle),
            where=OSC_ORI_OUTPUT_MAX > 1e-12,
        )
        normalized_pos_delta = np.clip(normalized_pos_delta, -1.0, 1.0)
        normalized_ori_delta = np.clip(normalized_ori_delta, -1.0, 1.0)

        desired_gripper_open = bool(action[-1] >= 0.5)
        current_gripper_open = bool(self.last_frame_obs.get("gripper_open", 1.0) >= 0.5)
        # Always command toward the desired gripper state.
        # The old toggle logic stopped commanding once qpos crossed a low threshold,
        # which could stall at a half-open aperture (~0.02) instead of fully open.
        gripper_action = -1.0 if desired_gripper_open else 1.0

        controller_action = np.concatenate(
            [normalized_pos_delta, normalized_ori_delta, [gripper_action]], axis=0
        ).astype(np.float32)
        debug = {
            "current_eef_pos_world": current_pos,
            "current_eef_quat_xyzw": current_quat,
            "target_eef_pos_world": target_pos,
            "target_eef_quat_xyzw": target_quat,
            "position_delta_world_m": position_delta,
            "delta_axis_angle_world_rad": delta_axis_angle,
            "normalized_pos_delta": normalized_pos_delta,
            "normalized_ori_delta": normalized_ori_delta,
            "desired_gripper_open": desired_gripper_open,
            "current_gripper_open": current_gripper_open,
            "controller_gripper_action": gripper_action,
            "controller_action_sent_to_libero": controller_action,
            "controller_logic": (
                "absolute world target pose -> world-frame delta pose (per-substep clipped to "
                "[0.02,0.02,0.02] m and [0.2,0.2,0.2] rad) -> normalize by robosuite "
                "OSC output_max ([0.05,0.05,0.05] m, [0.5,0.5,0.5] rad) -> append gripper action "
                "with PandaGripper convention -1=open, +1=close."
            ),
        }
        return controller_action, debug

    def _pose_error_to_target(self, action):
        action = np.asarray(action, dtype=np.float64)
        current_pos = np.asarray(self.last_frame_obs["robot0_eef_pos"], dtype=np.float64)
        current_quat = np.asarray(self.last_frame_obs["robot0_eef_quat"], dtype=np.float64)
        target_pos = np.asarray(action[:3], dtype=np.float64)
        target_quat = np.asarray(action[3:7], dtype=np.float64)

        if np.linalg.norm(current_quat) < 1e-8:
            current_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            current_quat = current_quat / np.linalg.norm(current_quat)

        if np.linalg.norm(target_quat) < 1e-8:
            target_quat = current_quat.copy()
        else:
            target_quat = target_quat / np.linalg.norm(target_quat)

        pos_error_world = target_pos - current_pos
        ori_error_axis_angle = (Rotation.from_quat(target_quat) * Rotation.from_quat(current_quat).inv()).as_rotvec()
        pos_error_norm_m = float(np.linalg.norm(pos_error_world))
        ori_error_norm_rad = float(np.linalg.norm(ori_error_axis_angle))
        return {
            "current_eef_pos_world": current_pos,
            "current_eef_quat_xyzw": current_quat,
            "target_eef_pos_world": target_pos,
            "target_eef_quat_xyzw": target_quat,
            "pos_error_world_m": pos_error_world,
            "ori_error_axis_angle_world_rad": ori_error_axis_angle,
            "pos_error_norm_m": pos_error_norm_m,
            "ori_error_norm_rad": ori_error_norm_rad,
        }

    def step(self, action, debug_payload=None):
        assert self._reset, "Reset the environment before stepping."
        del debug_payload
        info = {}
        absolute_target_action = np.asarray(action, dtype=np.float64)
        abs_mode_action = absolute_target_action.copy()
        abs_mode_action[-1] = 1.0 if abs_mode_action[-1] >= 0.5 else -1.0
        manual_abs_command = (
            f"abs {abs_mode_action[0]} {abs_mode_action[1]} {abs_mode_action[2]} "
            f"{abs_mode_action[3]} {abs_mode_action[4]} {abs_mode_action[5]} "
            f"{abs_mode_action[6]} {int(abs_mode_action[7])}"
        )
        self._current_step += 1
        action_success = False
        configured_repeat_count = max(1, int(self.action_repeat))
        repeat_count = ACTION_REPEAT_MAX
        controller_action_last = None
        executed_repeat_count = 0
        pose_progress_every_10 = []
        last_progress_pose = None
        stop_reason = "max_repeat_reached"
        pose_converged = False
        pose_stalled = False
        terminate = False
        reward = 0.0
        try:
            # One high-level VLM action is applied as repeated absolute-target
            # updates. This matches the verified manual-control behavior where the
            # same `abs` target is sent multiple times to smoothly move the arm.
            # We check pose convergence every ACTION_PROGRESS_INTERVAL sub-steps,
            # and stop early once close enough to the target pose.
            for repeat_idx in range(repeat_count):
                controller_action, _ = self._change_action_format(action)
                obs, reward, terminate, _ = self.env.step(controller_action)
                self.last_frame_obs = self._format_observation(obs)
                self.get_prompt_object_entries()
                controller_action_last = np.asarray(controller_action, dtype=np.float32)
                executed_repeat_count = repeat_idx + 1
                if terminate:
                    stop_reason = "env_terminated"
                    break
                if executed_repeat_count % ACTION_PROGRESS_INTERVAL == 0:
                    pose_error = self._pose_error_to_target(absolute_target_action)
                    converged = (
                        pose_error["pos_error_norm_m"] <= POSE_CONVERGE_POS_TOL_M
                        and pose_error["ori_error_norm_rad"] <= POSE_CONVERGE_ORI_TOL_RAD
                    )
                    checkpoint_pos_delta_m = None
                    checkpoint_ori_delta_rad = None
                    stalled = False
                    if last_progress_pose is not None:
                        checkpoint_pos_delta_m = float(
                            np.linalg.norm(
                                pose_error["current_eef_pos_world"] - last_progress_pose["pos"]
                            )
                        )
                        prev_rot = Rotation.from_quat(last_progress_pose["quat"])
                        curr_rot = Rotation.from_quat(pose_error["current_eef_quat_xyzw"])
                        checkpoint_ori_delta_rad = float(np.linalg.norm((curr_rot * prev_rot.inv()).as_rotvec()))
                        if (
                            executed_repeat_count > POSE_STALL_CHECK_START_REPEAT
                            and checkpoint_pos_delta_m <= POSE_STALL_POS_DELTA_TOL_M
                            and checkpoint_ori_delta_rad <= POSE_STALL_ORI_DELTA_TOL_RAD
                        ):
                            stalled = True
                    progress_item = {
                        "action_repeat_executed": executed_repeat_count,
                        "pos_error_norm_m": pose_error["pos_error_norm_m"],
                        "ori_error_norm_rad": pose_error["ori_error_norm_rad"],
                        "target_eef_pos_world": pose_error["target_eef_pos_world"],
                        "target_eef_quat_xyzw": pose_error["target_eef_quat_xyzw"],
                        "current_eef_pos_world": pose_error["current_eef_pos_world"],
                        "current_eef_quat_xyzw": pose_error["current_eef_quat_xyzw"],
                        "delta_from_prev_checkpoint_pos_m": checkpoint_pos_delta_m,
                        "delta_from_prev_checkpoint_ori_rad": checkpoint_ori_delta_rad,
                        "converged": converged,
                        "stalled_after_30": stalled,
                    }
                    pose_progress_every_10.append(progress_item)
                    logger.info(
                        "action progress repeat=%d/%d pos_err=%.6f ori_err=%.6f converged=%s",
                        executed_repeat_count,
                        repeat_count,
                        pose_error["pos_error_norm_m"],
                        pose_error["ori_error_norm_rad"],
                        converged,
                    )
                    last_progress_pose = {
                        "pos": np.asarray(pose_error["current_eef_pos_world"], dtype=np.float64),
                        "quat": np.asarray(pose_error["current_eef_quat_xyzw"], dtype=np.float64),
                    }
                    if converged:
                        pose_converged = True
                        stop_reason = "pose_converged"
                        break
                    if stalled:
                        pose_stalled = True
                        stop_reason = "pose_stalled_after_30"
                        break
            action_success = True
        except Exception as e:
            logger.info("*** An unexpected error occurred when env.step: %s", e)
            if executed_repeat_count == 0:
                reward = -1.0
            terminate = False
            action_success = str(e)
            stop_reason = "controller_exception"
        env_feedback = self.get_env_feedback(action_success, reward)
        info["env_feedback"] = env_feedback
        info["instruction"] = self.episode_language_instruction
        info["env_step"] = self._current_step
        info["episode_elapsed_seconds"] = time.time() - self._episode_start_time
        info["episode_num"] = self._current_episode_num
        info["action"] = absolute_target_action.tolist()
        info["absolute_target_action"] = absolute_target_action.tolist()
        info["abs_mode_action"] = abs_mode_action.tolist()
        info["manual_abs_command"] = manual_abs_command
        info["controller_action"] = (
            controller_action_last.tolist()
            if action_success is True and controller_action_last is not None
            else None
        )
        info["action_repeat_configured"] = configured_repeat_count
        info["action_repeat_target"] = repeat_count
        info["action_repeat_executed"] = executed_repeat_count
        info["action_stop_reason"] = stop_reason
        info["pose_converged"] = pose_converged
        info["pose_stalled"] = pose_stalled
        info["pose_convergence_check_interval"] = ACTION_PROGRESS_INTERVAL
        info["pose_convergence_thresholds"] = {
            "position_m": POSE_CONVERGE_POS_TOL_M,
            "orientation_rad": POSE_CONVERGE_ORI_TOL_RAD,
        }
        info["pose_stall_thresholds"] = {
            "start_repeat_exclusive": POSE_STALL_CHECK_START_REPEAT,
            "delta_position_m": POSE_STALL_POS_DELTA_TOL_M,
            "delta_orientation_rad": POSE_STALL_ORI_DELTA_TOL_RAD,
        }
        info["pose_progress_every_10"] = pose_progress_every_10
        info["action_success"] = 1.0 if action_success is True else 0.0
        info["task_success"] = 1.0 if terminate and reward == 1.0 else 0.0
        if self._current_step >= self._max_episode_steps:
            terminate = True
        self.episode_log.append(info)
        return self.last_frame_obs, reward, terminate, info

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def get_datum_states_path(self, curr_episode_num=None) -> str:
        if curr_episode_num is None:
            curr_episode_num = self._current_episode_num
        return os.path.join(self.log_path, "images", f"E{curr_episode_num:03d}")

    def get_datum_result_path(self, curr_episode_num=None) -> str:
        if curr_episode_num is None:
            curr_episode_num = self._current_episode_num
        filename = f"episode_{curr_episode_num}_res.json"
        return os.path.join(self.log_path, "results", filename)

    def _prepare_image(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            max_value = float(np.max(image)) if image.size > 0 else 0.0
            min_value = float(np.min(image)) if image.size > 0 else 0.0
            if max_value <= 1.0 and min_value >= 0.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.flipud(image))

    def render_images(self, camera_views=None):
        if camera_views is None:
            camera_views = ["front_rgb"]
        image_key_map = {
            "front_rgb": "agentview_image",
            "wrist_rgb": "robot0_eye_in_hand_image",
        }
        images = {}
        for cam_view in camera_views:
            source_key = image_key_map[cam_view]
            image = self._prepare_image(self.last_frame_obs[source_key])
            images[cam_view] = image
        return images

    def get_annotation_points(self, camera_views=None):
        if camera_views is None:
            camera_views = ["front_rgb"]
        object_entries = self.get_prompt_object_entries()
        all_avg_point_list = [np.asarray(entry["position"], dtype=np.float64) for entry in object_entries]
        instance_id_lookup = getattr(self.env, "instance_to_id", {})
        output = {}
        for camera_view in camera_views:
            camera_name = self.camera_name_map[camera_view]
            seg_key = _segmentation_key_for_camera(self.last_frame_obs, camera_name)
            raw_segmentation = _normalize_segmentation_image(self.last_frame_obs[seg_key])
            flipped_segmentation = np.flipud(raw_segmentation)
            params = self.get_camera_params(camera_view)
            fallback_points_2d = _project_world_points_to_image(
                all_avg_point_list,
                params["extrinsic"],
                params["intrinsic"],
            )
            fallback_points_2d = _flip_projected_points_vertical(
                fallback_points_2d,
                self.camera_height,
            )
            pixel_points_2d = []
            for idx, entry in enumerate(object_entries):
                instance_name = entry["name"]
                instance_id = instance_id_lookup.get(instance_name)
                centroid = None
                if instance_id is not None:
                    centroid = _largest_component_centroid(flipped_segmentation == int(instance_id))
                if centroid is None:
                    centroid = np.asarray(fallback_points_2d[idx]).reshape(-1)
                pixel_points_2d.append(np.asarray(centroid, dtype=np.float64).tolist())
            output[camera_view] = pixel_points_2d
        return output

    def get_env_feedback(self, task_success, reward=None):
        msg = ""
        msg += (
            f"You are currently performing the task intended to {self.episode_language_instruction} "
            f"At this moment, you have completed executing {self._current_step} steps. "
        )
        if task_success is True:
            msg += "Last action is valid. "
        else:
            msg += f"Last action is invalid. {task_success}."
        msg += f"The current reward obtained is {reward}."
        return msg

    def _serialize_obs(self):
        if self.last_frame_obs is None:
            return None
        keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "gripper_pose",
            "gripper_open",
        ]
        payload = {}
        for key in keys:
            value = self.last_frame_obs[key]
            if isinstance(value, np.ndarray):
                payload[key] = value.tolist()
            else:
                payload[key] = float(value)
        return payload

    def export_state(self):
        camera_params = {}
        for camera_type in self.camera_name_map:
            params = self.get_camera_params(camera_type)
            if params is None:
                continue
            camera_params[camera_type] = {
                "extrinsic": params["extrinsic"].tolist(),
                "intrinsic": params["intrinsic"].tolist(),
            }
        return {
            "number_of_episodes": self.number_of_episodes,
            "_current_episode_num": self._current_episode_num,
            "_current_step": self._current_step,
            "_max_episode_steps": self._max_episode_steps,
            "action_repeat": self.action_repeat,
            "episode_language_instruction": self.episode_language_instruction,
            "current_task_variation": self.current_task_variation,
            "task_class": self.task_class,
            "scene_bounds": self.scene_bounds.tolist(),
            "last_frame_obs": self._serialize_obs(),
            "object_entries": [
                {"name": entry["name"], "position": np.asarray(entry["position"], dtype=np.float64).tolist()}
                for entry in self.get_prompt_object_entries()
            ],
            "camera_params": camera_params,
        }

    @staticmethod
    def encode_png_base64(image):
        buf = io.BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        return buf.getvalue()
