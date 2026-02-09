import re
import os
import numpy as np
from tqdm import tqdm
import json
import copy
import argparse
import time
from time import sleep
from wiw_manip.evaluator.config.system_prompts import eb_manipulation_system_prompt
# from wiw_manip.envs.EBManEnv import EBManEnv, EVAL_SETS, ValidEvalSets
from wiw_manip.envs.eb_man_utils import (
    form_object_coord_for_input,
    draw_bounding_boxes,
    draw_xyz_coordinate,
)
from wiw_manip.planner.vlm_planner import VLMPlanner
from wiw_manip.evaluator.config.eb_manipulation_example import (
    vlm_examples_baseline,
    llm_examples,
    vlm_examples_ablation,
)
from wiw_manip.main import logger
from wiw_manip.planner.utils.visualize import visualize_ar_baseline
from wiw_manip.planner.utils.planner_utils import _get
from wiw_manip.envs.eb_man_utils import get_continous_action_from_discrete
from pyrep.backend import sim
from pyrep.const import ObjectType

class Base_Evaluator():
    def __init__(self, config):
        pass

    def load_demonstration(self):
        pass

    def save_episode_metric(self, episode_info):
        filename = self.env.get_datum_result_path()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, indent=2, ensure_ascii=False)

    def save_planner_outputs(self, reasoning_list):
        dir = os.path.dirname(self.env.get_datum_result_path())
        os.makedirs(dir, exist_ok=True)
        filename = os.path.join(
            dir, "planner_output_episode_{}.txt".format(self.env._current_episode_num)
        )
        with open(filename, 'w', encoding='utf-8') as f:
            for s in reasoning_list:
                f.write(str(s) + "\n")

    def print_task_eval_results(self, filename):
        folder_path = f"{self.log_path}/results"
        total_number_of_task = 0
        success_number_of_task = 0
        planner_steps = 0
        output_format_error = 0
        task_log = {"details": {}}
        ep_idxs_all = []

        if not os.path.exists(folder_path):
            return
        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".json") and file_name.startswith("episode"):
                file_path = os.path.join(folder_path, file_name)
                ep_idx = re.search(r'episode_(\d+)', file_name)
                assert ep_idx is not None, f"Episode index not found in {file_name}"
                ep_idx = int(ep_idx.group(1))
                ep_idxs_all.append(ep_idx)

                # Open and load the JSON file
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    task_success = data["task_success"]
                    if data["planner_output_error"] > 0:
                        output_format_error += 1
                    if task_success == 1:
                        success_number_of_task += 1
                    planner_steps += data["planner_steps"]
                    total_number_of_task += 1
                    task_log["details"][file_name] = {
                        "vis_path": data.get("vis_path", None),
                        "task_success": task_success,
                        "planner_output_error": data["planner_output_error"],
                        "planner_steps": data["planner_steps"]
                    }

        # find the missing ep_idxs:
        all_ep = list(range(1, self.env.number_of_episodes + 1))
        missing_ep_idxs = [ep for ep in all_ep if ep not in ep_idxs_all]
        missing_ep_paths = [
            self.env.get_datum_states_path(curr_episode_num=missing_ep_idx)
            for missing_ep_idx in missing_ep_idxs
        ]
        task_log['save_path'] = self.log_path
        task_log["total_num_tasks"] = total_number_of_task
        task_log["dataset_len"] = len(self.env.dataset)
        task_log["missing_ep_paths"] = missing_ep_paths
        task_log["num_success"] = success_number_of_task
        task_log["success_rate"] = success_number_of_task / total_number_of_task
        task_log["avg_planner_steps"] = planner_steps / total_number_of_task
        task_log["output_format_error"] = output_format_error

        with open(os.path.join(self.env.log_path, filename), 'w', encoding='utf-8') as f:
            json.dump(task_log, f, indent=2, ensure_ascii=False)

    def env_reset(self):
        raise NotImplementedError("env_reset method should be implemented in evaluator")

    def env_step(self, action):
        raise NotImplementedError("env_step method should be implemented in evaluator")

    def act(self, img_path_list, image_history, user_instruction, avg_obj_coord, obs, img_path_list_origin):
        if self.config["multistep"]:
            action_plans, reasonings_list = self.planner.act(
                image_history,
                user_instruction,
                str(avg_obj_coord),
                self.env.current_task_variation,
                img_path_list_origin=img_path_list_origin,
                curr_obs=obs,
                last_act=self.last_act,
            )
        else:
            action_plans, reasonings_list = self.planner.act(
                img_path_list,
                user_instruction,
                str(avg_obj_coord),
                self.env.current_task_variation,
                img_path_list_origin=img_path_list_origin,
                curr_obs=obs,
                last_act=self.last_act,
            )
            return action_plans, reasonings_list

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            # init states:
            _, obs = self.env_reset()
            # for resume evaluation or parallel evaluation:
            datum_states_path = self.env.get_datum_states_path()
            if os.path.exists(datum_states_path):
                logger.info(f"Datum <{datum_states_path}> already exists, skipping...")
                progress_bar.update()
                continue
            os.makedirs(datum_states_path, exist_ok=True)
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            logger.info(f"Episode data saving path: {datum_states_path}")

            episode_info = {'reward': [], 'action_success': []}
            image_history = []

            if self.config['multiview']:
                camera_views = ['front_rgb', 'wrist_rgb']
            else:
                camera_views = ['front_rgb']
            img_path_list = self.env.save_image(camera_views)
            img_path_list_origin = img_path_list.copy()

            (   avg_obj_coord, all_avg_point_list,
                camera_extrinsics_list, camera_intrinsics_list,
            ) = form_object_coord_for_input(
                vars(copy.deepcopy(obs)), self.env.task_class, camera_views
            )

            if not self.config["language_only"]:
                for i, img_path in enumerate(img_path_list):
                    if 'front_rgb' in img_path:
                        img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
            if self.config['detection_box'] and not self.config['language_only']:
                img_path_list = draw_bounding_boxes(
                    img_path_list,
                    all_avg_point_list,
                    camera_extrinsics_list,
                    camera_intrinsics_list,
                )
            if self.config['multistep']:
                image_history.append(img_path_list[0])
            user_instruction = self.env.episode_language_instruction
            logger.info(f"Instruction: {user_instruction}")
            self.planner.reset()
            done = False
            reasoning_list = []
            self.last_act = None

            waypoint_No = 0
            while not done:
                action_plans, reasonings_list = self.act(
                    img_path_list=img_path_list,
                    image_history=image_history,
                    user_instruction=user_instruction,
                    avg_obj_coord=avg_obj_coord,
                    obs=obs,
                    img_path_list_origin=img_path_list_origin,
                )
                logger.info(f"Planner Output Actions: {action_plans}")
                # Assume there is only one action plan per step for simulator
                assert len(action_plans) == len(reasonings_list) == 1
                actions = action_plans[0]; reasonings = reasonings_list[0]
                if self.config['use_last_action']:
                    self.last_act = "\n\nHere is the action performed by last iteration: " + str(actions) + ". You should consider this action as completed and don't plan it again. \n\nAnd below is the reason you gave in last iteration, notice that even if you gave more than 1 actions to perform, only first action will be performed. So it's not all actions in last plan are performed, but only first one. Even though, you still should plan a complete action sequence and output it. \n\nReasons you gave:" + str(reasonings)
                reasoning_list.append(reasonings)

                if len(actions) == 0:
                    assert False, "Planner should not return empty action plans."
                    episode_info['reward'].append(0)
                    episode_info['action_success'].append(0)
                    info = {'task_success': 0, 'episode_elapsed_seconds': 0}
                    break
                else:
                    executed_action_num = min(
                        self.env._max_episode_steps - self.env._current_step,
                        self.planner.executed_action_per_step, #NOTE: changed the original logic. TODO: check if self.env._current_step update correctly
                    )
                    for action_single in actions[:executed_action_num]:
                        # action_single should be [7,]
                        obs, reward, done, info = self.env_step(action_single)  
                        self.record_action_plan(obs, action_single)
                        logger.info(f"Executed action: {action_single}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        self.planner.update_info(info)
                        img_path_list = self.env.save_image(camera_views)
                        img_path_list_origin = img_path_list.copy()
                        for img_path in img_path_list:
                            if self.config['multistep']:
                                image_history.append(img_path)
                        episode_info['reward'].append(reward)
                        episode_info['action_success'].append(info['action_success'])
                        if done: break

                (   avg_obj_coord, all_avg_point_list,
                    camera_extrinsics_list, camera_intrinsics_list,
                ) = form_object_coord_for_input(
                    vars(copy.deepcopy(obs)), self.env.task_class, camera_views
                )
                if not done:
                    if not self.config['language_only']:
                        for i, img_path in enumerate(img_path_list):
                            if 'front_rgb' in img_path:
                                img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
                    if self.config['detection_box'] and not self.config['language_only']:
                        img_path_list = draw_bounding_boxes(
                            img_path_list,
                            all_avg_point_list,
                            camera_extrinsics_list,
                            camera_intrinsics_list,
                        )
                        if self.config["multistep"]:
                            if image_history[-1].split('.png')[0] in img_path_list[0]:
                                image_history.pop()
                                image_history.append(img_path_list[0])

            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['avg_reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info['num_steps'] = self.env._current_step
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["episode_elapsed_seconds"] = info["episode_elapsed_seconds"]
            episode_info["vis_path"] = self.env.get_datum_states_path()
            self.save_episode_metric(episode_info)
            self.save_planner_outputs(reasoning_list)
            visualize_ar_baseline(
                datum_dir=datum_states_path,
                key="front_rgb",
                label=user_instruction if isinstance(user_instruction, str) else user_instruction[0],
                planner_file_name="planner.json",
            )
            progress_bar.update()
        self.print_task_eval_results(filename="summary.json")
        # self.env.close()      # NOTE: not close env here to allow multiple evals w/o initializing env again

    def record_action_plan(self, obs, action_single, post_fix=""):
        gripper_pose = _get(obs, "gripper_pose").tolist()
        gripper_open = _get(obs, "gripper_open")
        action_plan_path = self.env.save_action_plan(
            [action_single, {"gripper_states_aft_act": [gripper_pose, gripper_open]}],
            post_fix=post_fix,
        )

    # def evaluate_main(self):
    #     valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
    #     valid_eval_sets = list(valid_eval_sets)
    #     if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
    #         valid_eval_sets = ValidEvalSets

    #     for i, eval_set in enumerate(valid_eval_sets):
    #         self.eval_set = eval_set
    #         logger.info(f'Current eval set: {eval_set}')
    #         if "/" in self.model_name:
    #             real_model_name = self.model_name.split('/')[1]
    #         else:
    #             real_model_name = self.model_name
    #         if 'exp_name' not in self.config or self.config['exp_name'] is None:
    #             self.log_path = "running/{}/{}/n_shot={}_resolution={}_detection_box={}_multiview={}_multistep={}_visual_icl={}/{}".format(
    #                 self.config.env,
    #                 real_model_name,
    #                 self.config["n_shots"], self.config["resolution"],
    #                 self.config["detection_box"], self.config["multiview"],
    #                 self.config["multistep"], self.config["visual_icl"],
    #                 self.eval_set,
    #             )
    #         else:
    #             self.log_path = "running/{}/{}/{}/{}".format(
    #                 self.config.env, real_model_name, self.config["exp_name"], self.eval_set
    #             )

    #         if i == 0:
    #             self.env = EBManEnv(
    #                 eval_set=self.eval_set,
    #                 img_size=(self.config["resolution"], self.config["resolution"]),
    #                 down_sample_ratio=self.config["down_sample_ratio"],
    #                 log_path=self.log_path,
    #                 enable_path_obs=self.config["enable_path_obs"],
    #                 exp_name=self.config.get("exp_name", None),
    #                 max_step=self.config["max_step"],
    #             )
    #         else:
    #             self.env.init_dataset_and_tasks(
    #                 eval_set=self.eval_set,
    #                 down_sample_ratio=self.config["down_sample_ratio"],
    #                 log_path=self.log_path,
    #             )
    #         # turn_off_shadow()
    #         ic_examples = self.load_demonstration()
    #         self.initialize_planner(ic_examples)
    #         self.evaluate()
    #         with open(os.path.join(self.log_path, 'config.txt'), 'w') as f:
    #             f.write(str(self.config))

    def initialize_planner(self, ic_examples, task_name):
        self.planner = VLMPlanner(
            model_name=self.model_name,
            model_type=self.config["model_type"],
            system_prompt=eb_manipulation_system_prompt,
            examples=ic_examples,
            n_shot=self.config["n_shots"],
            chat_history=self.config["chat_history"],
            language_only=self.config["language_only"],
            multiview=self.config["multiview"],
            multistep=self.config["multistep"],
            visual_icl=self.config["visual_icl"],
            tp=self.config["tp"],
            executed_action_per_step=self.config["executed_action_per_step"],
            vlm_args={
                "temperature": self.config.get("vlm__temperature"),
                "top_k": self.config.get("vlm__top_k"),
            },
        )

    def check_config_valid(self):
        if self.config['multiview'] + self.config['multistep'] + self.config['visual_icl'] + self.config['chat_history'] > 1:
            raise ValueError("Currently, we only support one of multiview, multistep, visual_icl, chat_history feature at a time.")

        if self.config['language_only']:
            if self.config['multiview'] or self.config['multistep']:
                logger.warning("Language only mode should not have multiview or multistep enabled. Setting these arguments to False ...")
                self.config['multiview'] = 0
                self.config['multistep'] = 0