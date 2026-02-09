# Modified From the rlbench: https://github.com/stepjam/RLBench
import pickle
from typing import Union, Dict, Tuple, List
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
import gymnasium as gym
from gymnasium import spaces
from amsolver.environment import Environment
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.observation_config import ObservationConfig
import numpy as np
from amsolver.backend.utils import task_file_to_task_class
from pathlib import Path
import os, re
import time
from PIL import Image
from wiw_manip.main import logger
import json
from collect_dataset_util import save_demo, create_unique_episode_dir
from wiw_manip.configs.paths import TEST_DATASET_PATH
from wiw_manip.envs.eb_man_utils import VALID_TASKS

class RLBenchEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        eval_set,
        render_mode="human",
        img_size=(500, 500),
        down_sample_ratio=1.0,
        log_path=None,
        selected_indexes=[],
        enable_path_obs=False,
        exp_name=None,
        max_step=15,
        dataset_root=TEST_DATASET_PATH
    ):
        # * init env:
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        obs_config.set_image_size(img_size)
        self._max_episode_steps=max_step

        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True,
            dataset_root=dataset_root)
        self.dataset_root = dataset_root
        self.env.launch()
        self._render_mode = render_mode
        self.action_space = spaces.Box(
            low=0.0, high=100.0, shape=(self.env.action_size,))

        # * Load dataset
        self.init_dataset_and_tasks(eval_set, down_sample_ratio, log_path, selected_indexes)

        # * set render mode and camera
        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL)

        # New added:
        self._enable_path_obs = enable_path_obs
        self.exp_name = exp_name

    # redo this method
    def init_dataset_and_tasks(self, eval_task, down_sample_ratio, log_path, selected_indexes=[]):
        assert eval_task in VALID_TASKS
        self.task_class = eval_task
        task_class = task_file_to_task_class(eval_task, parent_folder='vlm')
        self.task = self.env.get_task(task_class)

        def get_variation_episode_pairs(eval_task):
            base_dir = self.dataset_root
            task_dir = os.path.join(base_dir, eval_task)
            if not os.path.exists(task_dir):
                raise ValueError(f"Task directory {task_dir} does not exist.")

            pairs = []
            variation_pattern = re.compile(r"^variation(\d+)$")
            episode_pattern = re.compile(r"^episode(\d+)$")

            for var_name in os.listdir(task_dir):
                var_match = variation_pattern.match(var_name)
                if not var_match:
                    continue
                variation = int(var_match.group(1))
                var_path = os.path.join(task_dir, var_name, "episodes")
                if not os.path.isdir(var_path):
                    continue

                for ep_name in os.listdir(var_path):
                    ep_match = episode_pattern.match(ep_name)
                    if not ep_match:
                        continue
                    episode = int(ep_match.group(1))
                    ep_path = os.path.join(var_path, ep_name)
                    if os.path.isdir(ep_path):
                        pairs.append((variation, episode))

            return sorted(pairs)
        self.dataset = get_variation_episode_pairs(eval_task)
        if down_sample_ratio < 1.0:
            num_ep = max(int(len(self.dataset) * down_sample_ratio), 1)
            self.dataset = self.dataset[:num_ep]

        # Not using
        # if len(selected_indexes) > 0:
        #     self.dataset = [self.dataset[i] for i in selected_indexes]
        # else:
        #     if down_sample_ratio < 1.0:
        #         self.dataset = self.dataset[:int(len(self.dataset) * down_sample_ratio)]
        self.current_task_variation = None
        self.log_path = log_path
        self.number_of_episodes = len(self.dataset)

        # Episode tracking
        self._reset = False
        self._current_episode_num = 0
        self._current_step = 0
        self._episode_start_time = 0
        self.episode_log = []

        # Task-related attributes
        self.episode_language_instruction = ''
        self.episode_data = None
        self.last_frame_obs = None

        # New added:
        self.demo_obs = []

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            return self._gym_cam.capture_rgb()

    def get_demo(self, task_name, variation, episode_index):
        """
        Fetch a demo from the saved environment.
            :param task_name: fetch task name
            :param variation: fetch variation id
            :param episode_index: fetch episode index: 0 ~ 99
            :return: desired demo
        """
        demos = self.env.get_test_demos(
            task_name=task_name,
            variation_number=variation,
            amount=1,
            from_episode_number=episode_index,
        )
        return demos

    def reset(self):
        """
        Reset the environment for a new episode.

        Returns:
            observation
        """
        assert self._current_episode_num <= self.number_of_episodes, "All episodes have been completed."
        self._current_step = 0
        self._current_episode_num += 1
        self._reset = True
        self._episode_log = []
        self._episode_start_time = time.time()
        logger.info("%d episodes in dataset", len(self.dataset))
        logger.info("Resetting episode %d ...", self._current_episode_num)

        variation, episode = self.dataset[self._current_episode_num - 1]
        self.task.set_variation(variation)
        self.current_task_variation = f"{self.task_class}_variation{variation}"

        demo = self.get_demo(
            task_name=self.task_class,
            variation=variation,
            episode_index=episode
        )[0]
        descriptions, obs = self.task.reset_to_demo(demo)

        self.episode_language_instruction = descriptions
        self.last_frame_obs = obs

        # New added:
        self.task.enable_path_observations(self._enable_path_obs)
        self.demo_obs = []

        return descriptions[0], obs

    def _change_action_format(self, action):
        return action

    def step(self, action):
        """
        action: 1d np.ndarray: (x, y, z, qx, qy, qz, qw, openness)
        """
        assert self._reset, "Reset the environment before stepping."
        info = {}
        self._current_step += 1
        action_success = False
        try:
            # return action is A numpy array (x, y, z) + (qx, qy, qz, qw) + openness
            action = self._change_action_format(action)
            # save the action and discrete_action （#use add mode not overwrite）
            # path = os.path.join("outputs/temp", "actions.json")
            # with open(path, 'a') as f:
            #     json.dump({
            #         'action': action.tolist(),
            #         'discrete_action': discrete_action
            #     }, f, indent=4)

            obs, reward, terminate = self.task.step(action)
            # if self.current_task_variation.startswith('stack'):
            #     if terminate:
            #         if action[-1] == 0.0:
            #             reward = 0.0
            #             terminate = False
            #             logger.debug("wrong success condition for stack, setting reward to 0 and terminate to False ...")
            #         elif action[-1] == 1.0:
            #             action[2] += 0.03
            #             logger.debug("checking if the object is stacked properly ...")
            #             obs, reward, terminate = self.task.step(action)
            #             if terminate and reward == 1.0:
            #                 logger.debug("stacking is successful ...")
            #                 reward = 1.0
            #                 terminate = True
            #             else:
            #                 logger.debug("stacking is unsuccessful ...")
            #                 reward = 0.0
            #                 terminate = False
            self.last_frame_obs = obs
            action_success = True
        except Exception as e:
            logger.info(f"*** An unexpected error occurred when env.step: {e}")
            obs, reward, terminate = self.last_frame_obs, -1, False
            action_success = e
        env_feedback = self.get_env_feedback(action_success, reward)     #!
        info['env_feedback'] = env_feedback
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['episode_num'] = self._current_episode_num
        info['action'] = action
        if action_success == True:
            info['action_success'] = 1.0
        else:
            info['action_success'] = 0.0
        if terminate and reward == 1.0:
            info['task_success'] = 1.0
        else:
            info['task_success'] = 0.0
        if self._current_step >= self._max_episode_steps:
            terminate = True
        self.episode_log.append(info)

        # For collect path observations as dataset:
        if self._enable_path_obs:
            self.accumulate_stepped_obs()
            # if True:
            if terminate:
                root = Path(
                    "data/07.16_vlm_demos/"
                    f"datac_{self.exp_name}"
                )
                task_name = f"{self.current_task_variation}-{self.curr_eval_set}"

                # Build   .../<task_name>/variation0/episodes
                episodes_base = root / task_name / "variation0" / "episodes"

                # --- create a fresh episodeN directory (thread/-process safe) ---
                save_dir = create_unique_episode_dir(episodes_base)
                save_demo(self.demo_obs, str(save_dir), also_save_video=True)

        return obs, reward, terminate, info

    def accumulate_stepped_obs(self) -> List[Dict[str, np.ndarray]]:
        """
        Get all observations from the current path.
        """
        obs_list = self.task.get_path_observations()
        self.demo_obs.extend(obs_list)

    def close(self) -> None:
        self.env.shutdown()

    def get_datum_states_path(self, curr_episode_num=None) -> str:
        if curr_episode_num is None:
            curr_episode_num = self._current_episode_num
        return os.path.join(self.log_path, "images", f"E{curr_episode_num:03d}")

    def get_datum_result_path(self, curr_episode_num=None) -> str:
        if curr_episode_num is None:
            curr_episode_num = self._current_episode_num
        filename = 'episode_{}_res.json'.format(curr_episode_num)
        return os.path.join(self.log_path, 'results', filename)

    def save_action_plan(self, actions, post_fix) -> str:
        log_path = os.path.join(self.get_datum_states_path(), f"A{self._current_step:03d}")
        filename = os.path.join(log_path, f'planner{post_fix}.json')
        os.makedirs(log_path, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(actions, f, indent=2, ensure_ascii=False)
        return filename

    def save_image(self, key=['front_rgb']) -> str:
        log_path = os.path.join(self.get_datum_states_path(), f"A{self._current_step:03d}")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        image_path_list=[]
        for cam_view in key:
            single_image = Image.fromarray(getattr(self.last_frame_obs, cam_view))
            time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            image_path = os.path.join(log_path, '{}.png'.format(cam_view))
            single_image.save(image_path)
            image_path_list.append(image_path)
        return image_path_list

    def get_env_feedback(self, task_success, reward=None):
        """
        Generate feedback message for the current step.
        Args:
            info (dict): Action execution information
        Returns:
            str: Descriptive message about step outcome
        """
        msg = ''
        msg += f"You are currently performing the task intended to {self.episode_language_instruction} At this moment, you have completed executing {self._current_step} steps. "
        if task_success == True:
            msg += f"Last action is valid. "        #TODO: readd feedback
        else:
            msg += f"Last action is invalid. {task_success}."
        msg += f"The current reward obtained is {reward}."
        return msg

if __name__ == '__main__':
    """
    Example usage of the EB-Manipulation environment.
    Demonstrates environment interaction with random actions.
    """
    test_env = RLBenchEnv(eval_set='base', selected_indexes=[0])
    description, obs = test_env.reset()
    test_env.save_image()
    print("testing the EB-Manipulation environment ...")
    print("ignore errors like could not create path or target is outside of workspace as actions are randomly sampled ...")
    for _ in range(3):
        action = test_env.action_space.sample()
        action[-1] = 1.0
        obs, reward, terminate, info = test_env.step(action)
        test_env.save_image()
    test_env.close()
    print("testing completed!")
