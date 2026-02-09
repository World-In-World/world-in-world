import tqdm
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from common.pytorch_util import dict_apply
from amsolver.gym.vlmbench_env import VLMBenchEnv
import pytorch3d.transforms as pt
import imageio

import os
from collections import deque
import numpy as np
import torch

class VLMBenchRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 max_steps,
                 n_envs,
                 obs_horizon,
                 action_horizon,
                 env=None
                ):
        self.env = VLMBenchEnv(**env)
        self.max_steps = max_steps
        self.output_dir = output_dir
        self.n_envs = n_envs
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.env.launch()
        
    def undo_transform_action(self, action: np.ndarray):
        quaternions = action[..., 3:7]
        quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
        gripper_open = (action[...,7:] > 0.5).astype(np.float32)
          
        new_action = np.concatenate([
            action[..., :3],  # gripper_xyz
            quaternions,  # gripper_quat                
            gripper_open          
        ], axis=-1)
        return new_action
    
    def stack_last_n_obs(self, all_obs, n_steps):
        assert len(all_obs) > 0
        if isinstance(all_obs[0], np.ndarray):
            result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = np.stack(all_obs[start_idx:], axis=0)
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        elif isinstance(all_obs[0], torch.Tensor):
            result = torch.zeros(
                (n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype
            )
            start_idx = -min(n_steps, len(all_obs))
            result[start_idx:] = torch.stack(all_obs[start_idx:])
            if n_steps > len(all_obs):
                # pad
                result[:start_idx] = result[start_idx]
        else:
            raise RuntimeError(f"Unsupported obs type {type(all_obs[0])}")
        return result
    
    def get_n_steps_obs(self, nobs):
        assert len(nobs) > 0, "no observation is recorded, please update obs first"

        result = dict()
        for key in nobs[0].keys():
            result[key] = self.stack_last_n_obs(
                [obs[key] for obs in nobs], self.obs_horizon
            )

        return result
    
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        log = dict()
        
        policy.reset()
        success_count = 0
        rewards = []
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"test_eval/{current_time}"
        os.makedirs(folder_name, exist_ok=True)
        for env_idx in tqdm.tqdm(range(self.n_envs), desc=f"Evaluating epoch {self.n_envs}", 
            total = self.n_envs):
            done = False
            nobs = deque(maxlen=self.obs_horizon + 1)
            obs = self.env.reset()
            nobs.append(obs)
            frames = []
            n_step = 0
            
            frames.append(obs['img'][0][...,:3].astype(np.uint8))
            while (not done) and n_step < self.max_steps:
                
                np_obs_dict = self.get_n_steps_obs(nobs)
                np_obs_dict = dict(np_obs_dict)
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                for key in obs_dict.keys():
                    obs_dict[key] = obs_dict[key].unsqueeze(0)
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                # env_action = action
                # if self.abs_action:
                env_action = self.undo_transform_action(action)
                
                for action in env_action:
                    try:
                        n_step += 1
                        obs, reward, done = self.env.step(action)
                        
                        frames.append(obs['img'][0][...,:3].astype(np.uint8))
                        
                        nobs.append(obs)
                        if reward > 0:
                            success_count += 1
                            break
                    except Exception as e:
                        print(action)
                        print("Invalid action, task failed")
                        print(str(e))
                        done = True
                        break
            file_name = folder_name + f"/{env_idx}_env.mp4"
            with imageio.get_writer(file_name, fps=3) as writer:
                for frame in frames:
                    writer.append_data(frame)
                print(f"Video saved to {file_name}")
        
        # _ = self.env.shutdown()
        success_rate = success_count / self.n_envs
        log["test_mean_score"] = success_rate
        return log
            