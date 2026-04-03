import numpy as np
import json
from typing import Optional
import torch
import torch.nn.functional as F
from amsolver.backend.observation import Observation
from wiw_manip.planner.base_planner import BasePlanner
from PIL import Image
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation

class OpenpiPlanner(BasePlanner):
    def __init__(
        self,
        task: str,
        executed_action_per_step=50,
        openpi_host="localhost:8011",
        **kwargs,
    ):
        super().__init__(task=task, executed_action_per_step=executed_action_per_step, **kwargs)
        self.openpi_host = openpi_host
        self.client = websocket_client_policy.WebsocketClientPolicy(host=self.openpi_host.split(":")[0], port=int(self.openpi_host.split(":")[1]))
        self.num_history = 1
        
        self.executed_action_per_step = executed_action_per_step
        self.planner_steps = 0
        self.output_json_error = 0
        self.task_name = task

    def reset(self, **kwargs):
        super().reset(**kwargs)
    
    def update_info(self, info, **kwargs):
        super().update_info(info=info, **kwargs)

    def act(self, curr_obs: Observation=None, user_instruction=None, query_num=1, **kwargs):
        observation = {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(curr_obs.front_rgb, 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(curr_obs.wrist_rgb, 224, 224)
            ),
            "observation/state": np.concatenate([curr_obs.gripper_pose, [curr_obs.gripper_open]], dtype=np.float32),
            "prompt": np.random.choice(user_instruction) if user_instruction is not None else self.task_name,
        }
        
        trajectories = []
        for _ in range(query_num):
            action_chunk = self.client.infer(observation)["actions"]
            trajectory = action_chunk.tolist()
            trajectory = [trajectory[1], trajectory[4], trajectory[6], trajectory[9]]
            
            for i in range(len(trajectory)):
                action = trajectory[i]
                euler_angles = action[3:6]
                rotation = Rotation.from_euler('xyz', euler_angles)
                quaternion = rotation.as_quat()  # Returns [qx, qy, qz, qw]
                action_8d = np.concatenate([
                    action[:3],  # x, y, z
                    quaternion,  # qx, qy, qz, qw
                    [np.clip((action[6] + 1) / 2, 0, 1)]  # normalized gripper_pos
                ])
                trajectory[i] = action_8d.tolist()
            trajectories.append(trajectory)
        return trajectories, [{"language_plan": "No reason available"}] * query_num
