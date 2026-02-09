import numpy as np
import json
from typing import Optional
import torch
import torch.nn.functional as F

from diffuser_actor.trajectory_optimization.diffuser_actor import DiffuserActor
from amsolver.backend.observation import Observation
from wiw_manip.planner.base_planner import BasePlanner

from wiw_manip.configs.paths import DIFF_ACTOR_CKPTS, GRIPPER_BOUND_JSON

def get_gripper_loc_bounds(path: str, buffer: float = 0.0, task: Optional[str] = None):
    gripper_loc_bounds = json.load(open(path, "r"))
    if task is not None and task in gripper_loc_bounds:
        gripper_loc_bounds = gripper_loc_bounds[task]
        gripper_loc_bounds_min = np.array(gripper_loc_bounds[0]) - buffer
        gripper_loc_bounds_max = np.array(gripper_loc_bounds[1]) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    else:
        # Gripper workspace is the union of workspaces for all tasks
        gripper_loc_bounds = json.load(open(path, "r"))
        gripper_loc_bounds_min = np.min(np.stack([bounds[0] for bounds in gripper_loc_bounds.values()]), axis=0) - buffer
        gripper_loc_bounds_max = np.max(np.stack([bounds[1] for bounds in gripper_loc_bounds.values()]), axis=0) + buffer
        gripper_loc_bounds = np.stack([gripper_loc_bounds_min, gripper_loc_bounds_max])
    print("Gripper workspace size:", gripper_loc_bounds_max - gripper_loc_bounds_min)
    return gripper_loc_bounds

class DiffPlanner(BasePlanner):
    def __init__(
        self,
        task: str,
        executed_action_per_step=50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert task in DIFF_ACTOR_CKPTS.keys(), f"Diffuser Actor checkpoint for task {task} not found."
        self.num_history = 1
        gripper_loc_bounds = get_gripper_loc_bounds(
            GRIPPER_BOUND_JSON,
            task=task, buffer=0.04,
        )
        self.diff_model = DiffuserActor(
            backbone="clip",
            image_size=(256, 256),
            embedding_dim=192,
            num_vis_ins_attn_layers=2,
            use_instruction=False,
            fps_subsampling_factor=5,
            gripper_loc_bounds=gripper_loc_bounds,
            rotation_parametrization='6D',
            quaternion_format='wxyz',
            diffusion_timesteps=100,
            nhist=1,
            relative=False,
            lang_enhanced=False,
        )
        payload = torch.load(DIFF_ACTOR_CKPTS[task], map_location="cpu")
        payload_weight = {}
        for key in payload["weight"]:
            _key = key[7:]
            payload_weight[_key] = payload["weight"][key]
        self.diff_model.load_state_dict(payload_weight)
        self.diff_model.cuda()
        self.diff_model.eval()
        self.device = next(self.diff_model.parameters()).device
        
        self.executed_action_per_step = executed_action_per_step
        self.planner_steps = 0
        self.output_json_error = 0
        self.task_name = task

    def reset(self, **kwargs):
        super().reset(**kwargs)
    
    def update_info(self, info, **kwargs):
        super().update_info(info=info, **kwargs)

    def act(self, curr_obs: Observation=None, gripper_history=None, query_num=1, **kwargs):
        device = self.device
        trajectory_mask = torch.full((1, self.executed_action_per_step), False).to(device)
        rgb_obs = torch.from_numpy(np.stack([curr_obs.front_rgb, curr_obs.wrist_rgb, curr_obs.left_shoulder_rgb, curr_obs.right_shoulder_rgb, curr_obs.overhead_rgb], axis=0)).unsqueeze(0).permute(0, 1, 4, 2, 3).float().to(device)
        rgb_obs = torch.nn.functional.interpolate(rgb_obs.view(-1, 3, rgb_obs.shape[-2], rgb_obs.shape[-1]), size=(256, 256), mode='nearest').view(1, 5, 3, 256, 256)
        
        pcd_obs = torch.from_numpy(np.stack([curr_obs.front_point_cloud, curr_obs.wrist_point_cloud, curr_obs.left_shoulder_point_cloud, curr_obs.right_shoulder_point_cloud, curr_obs.overhead_point_cloud], axis=0)).unsqueeze(0).permute(0, 1, 4, 2, 3).float().to(device)
        pcd_obs = torch.nn.functional.interpolate(pcd_obs.view(-1, 3, pcd_obs.shape[-2], pcd_obs.shape[-1]), size=(256, 256), mode='nearest').view(1, 5, 3, 256, 256)
        
        grippers = torch.from_numpy(np.stack(gripper_history, axis=0)).reshape(1, -1, 7).float().to(device)
        gripper_input = grippers[:, -self.num_history:]
        npad = self.num_history - gripper_input.shape[1]
        gripper_input = F.pad(
            gripper_input, (0, 0, npad, 0), mode='replicate'
        )
        trajectories = []
        for _ in range(query_num):
            with torch.no_grad():
                trajectory = self.diff_model.forward(
                    None,
                    trajectory_mask,
                    rgb_obs / 255.0,
                    pcd_obs,
                    None,
                    gripper_input,
                    run_inference=True
                )
            trajectory_list = trajectory.squeeze(0).detach().cpu().numpy().tolist()
            trajectories.append(trajectory_list)
        return trajectories, [{"language_plan": "No reason available"}] * query_num
