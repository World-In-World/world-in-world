import base64
import logging
import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger("libero_env_client")

LIBERO_VALID_TASKS = ["libero_object", "libero_spatial"]
DEFAULT_LIBERO_SERVER_URL = os.environ.get("LIBERO_ENV_URL", "http://127.0.0.1:8765")


class LiberoEnv:
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
        server_url=None,
        action_repeat=50,
    ):
        del render_mode, selected_indexes, enable_path_obs, exp_name, dataset_root
        self.server_url = (server_url or DEFAULT_LIBERO_SERVER_URL).rstrip("/")
        self.task_class = eval_set
        self.log_path = log_path
        self.camera_height = img_size[0]
        self.camera_width = img_size[1]
        self.dataset = []
        self.number_of_episodes = 0
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = max_step
        self.action_repeat = int(action_repeat)
        self._reset = False
        self.current_task_variation = None
        self.episode_language_instruction = ""
        self.last_frame_obs = None
        self.scene_bounds = None
        self._current_object_entries = []
        self._camera_params = {}
        self._annotation_points = {}
        self._configure(
            eval_set=eval_set,
            img_size=img_size,
            down_sample_ratio=down_sample_ratio,
            log_path=log_path,
            max_step=max_step,
            action_repeat=action_repeat,
        )

    def _request(self, method, path, payload=None, timeout=300):
        url = f"{self.server_url}{path}"
        response = requests.request(method, url, json=payload, timeout=timeout)
        if not response.ok:
            raise requests.HTTPError(
                f"{response.status_code} Server Error for {url}: {response.text}",
                response=response,
            )
        if response.content:
            return response.json()
        return None

    def _configure(self, eval_set, img_size, down_sample_ratio, log_path, max_step, action_repeat):
        payload = {
            "eval_set": eval_set,
            "img_size": list(img_size),
            "down_sample_ratio": down_sample_ratio,
            "log_path": log_path,
            "max_step": max_step,
            "action_repeat": int(action_repeat),
        }
        data = self._request("POST", "/configure", payload)
        self._sync_state(data["state"])

    def _sync_state(self, state):
        self.number_of_episodes = state["number_of_episodes"]
        self._current_episode_num = state["_current_episode_num"]
        self._current_step = state["_current_step"]
        self._max_episode_steps = state["_max_episode_steps"]
        self.action_repeat = int(state.get("action_repeat", self.action_repeat))
        self.episode_language_instruction = state["episode_language_instruction"]
        self.current_task_variation = state["current_task_variation"]
        self.task_class = state["task_class"]
        self.scene_bounds = np.asarray(state["scene_bounds"], dtype=np.float64) if state["scene_bounds"] is not None else None
        self._current_object_entries = [
            {"name": item["name"], "position": np.asarray(item["position"], dtype=np.float64)}
            for item in state["object_entries"]
        ]
        self._camera_params = {
            key: {
                "extrinsic": np.asarray(value["extrinsic"], dtype=np.float64),
                "intrinsic": np.asarray(value["intrinsic"], dtype=np.float64),
            }
            for key, value in state["camera_params"].items()
        }
        obs = state["last_frame_obs"]
        if obs is None:
            self.last_frame_obs = None
        else:
            self.last_frame_obs = {
                "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float64),
                "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
                "gripper_pose": np.asarray(obs["gripper_pose"], dtype=np.float64),
                "gripper_open": float(obs["gripper_open"]),
            }

    def init_dataset_and_tasks(self, eval_task, down_sample_ratio, log_path, selected_indexes=None):
        del selected_indexes
        payload = {
            "eval_task": eval_task,
            "down_sample_ratio": down_sample_ratio,
            "log_path": log_path,
        }
        data = self._request("POST", "/init_dataset_and_tasks", payload)
        self._sync_state(data["state"])

    def reset(self):
        data = self._request("POST", "/reset", {})
        self._sync_state(data["state"])
        self._reset = True
        return self.episode_language_instruction, self.last_frame_obs

    def step(self, action, debug_payload=None):
        payload = {"action": np.asarray(action, dtype=np.float64).tolist()}
        if debug_payload is not None:
            payload["debug_payload"] = debug_payload
        data = self._request("POST", "/step", payload)
        self._sync_state(data["state"])
        return self.last_frame_obs, data["reward"], data["done"], data["info"]

    def close(self):
        try:
            self._request("POST", "/close", {})
        except Exception as exc:
            logger.warning("Failed to close LIBERO server env cleanly: %s", exc)

    def get_datum_states_path(self, curr_episode_num=None) -> str:
        if curr_episode_num is None:
            curr_episode_num = self._current_episode_num
        return os.path.join(self.log_path, "images", f"E{curr_episode_num:03d}")

    def get_datum_result_path(self, curr_episode_num=None) -> str:
        if curr_episode_num is None:
            curr_episode_num = self._current_episode_num
        filename = f"episode_{curr_episode_num}_res.json"
        return os.path.join(self.log_path, "results", filename)

    def save_image(self, key=None):
        if key is None:
            key = ["front_rgb"]
        data = self._request("POST", "/render", {"camera_views": list(key)})
        self._annotation_points = {
            cam_view: np.asarray(points, dtype=np.float64)
            for cam_view, points in data.get("annotation_points", {}).items()
        }
        log_path = os.path.join(self.get_datum_states_path(), f"A{self._current_step:03d}")
        os.makedirs(log_path, exist_ok=True)
        image_path_list = []
        for cam_view in key:
            raw = base64.b64decode(data["images"][cam_view])
            image = Image.open(BytesIO(raw)).convert("RGB")
            image_path = os.path.join(log_path, f"{cam_view}.png")
            image.save(image_path)
            image_path_list.append(image_path)
        return image_path_list

    def get_prompt_object_entries(self):
        return self._current_object_entries

    def get_camera_params(self, camera_type):
        return self._camera_params[camera_type]

    def get_annotation_points(self, camera_type):
        return self._annotation_points.get(camera_type)
