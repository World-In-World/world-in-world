import os
import numpy as np
import torch
import socket
import json
import time
import datetime as _dt
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import Tensor
from jaxtyping import Float, Int32, UInt8
from wiw_manip.planner.diff_planner import DiffPlanner
from wiw_manip.planner.igenex_planner import IgenexPlanner
from wiw_manip.planner.utils.planner_utils import _get
from wiw_manip.planner.utils.vlm_util import run_concurrent_blocking

from wiw_manip.main import logger
from json_repair import repair_json
from wiw_manip.planner.utils.saver import (
    get_igenex_save_dirs,
    get_save_dir_from_existing,
    save_action_sequence,
    format_chat_dialog,
)
from wiw_manip.planner.planner_config.generation_guide_manip import (
    Igenex_evaluator_prompt,
    Igenex_evaluator_prompt_norepeat,
    genex_vlm_few_shot_examples,
    Igenex_descriptor_prompt,
    compose_visual_state,
)
from copy import deepcopy

VISUAL_ICL_EXAMPLES_PATH = "wiw_manip/evaluator/config/visual_icl_examples/eb_manipulation"
VISUAL_ICL_EXAMPLE_CATEGORY = {
    "pick": "pick_cube_shape",
    "place": "place_into_shape_sorter_color",
    "stack": "stack_cubes_color",
    "wipe": "wipe_table_direction"
}
WORLD_MODEL_TYPES = {
    "text": ["wan21", "ltx", "hunyuan", "nwm", "cosmos", "wan22", "svd", "gen3tur"],
    "FTtext": ["FTcosmos", "FTltx", "FTwan21", "FTwan22", "FTwan22-14B"],
    "action": ["igen"],
}

SHOW_TRAJECTORY_VISUALIZATION = False

def select_diverse_points(points: List[List[float]], k: int) -> List[List[float]]:
    """
    Select k points from a set of 3D points using a greedy max-min distance strategy
    to maximize the minimum pairwise distance among the selected points.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n = len(points)
    if n == 0:
        return []
    if k >= n:
        return list(range(n))

    pts = np.asarray(points, dtype=np.float64)  # (n, 3)

    centroid = pts.mean(axis=0)
    d2_centroid = np.sum((pts - centroid) ** 2, axis=1)
    first_idx = int(np.argmax(d2_centroid))

    selected_indices = [first_idx]
    diff = pts - pts[first_idx]
    min_dists = np.sqrt(np.sum(diff ** 2, axis=1))  # (n,)
    min_dists[first_idx] = 0.0  # mark selected

    while len(selected_indices) < k:
        for idx in selected_indices:
            min_dists[idx] = -1.0
        next_idx = int(np.argmax(min_dists))
        selected_indices.append(next_idx)

        if len(selected_indices) == k:
            break

        new_diff = pts - pts[next_idx]
        new_dists = np.sqrt(np.sum(new_diff ** 2, axis=1))
        mask = min_dists >= 0
        min_dists[mask] = np.minimum(min_dists[mask], new_dists[mask])

    return selected_indices

class DiffIgenexPlanner(DiffPlanner, IgenexPlanner):

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.shot_example_num = 1

    def act(self,
            gripper_history,
            img_path_list_anno,
            user_instruction,
            avg_obj_coord,
            task_variation,
            img_path_list_origin,
            curr_obs,
            last_act,
            **kwargs,
        ):
        proposer_fn = self.generate_trajectories(
            curr_obs=curr_obs, gripper_history=gripper_history
        )

        return self.query_igenex(img_path_list_anno, user_instruction, img_path_list_origin, curr_obs, proposer_fn, self.task_name, **kwargs)

    def generate_trajectories(
        self, curr_obs, gripper_history
    ):
        def proposer_fn(num_trajs, accumulate_trajs=[]):
            actions_plans, reasons = DiffPlanner.act(
                self=self,
                curr_obs=curr_obs,
                gripper_history=gripper_history,
                query_num=self.proposal_num,
            )

            all_trajs = actions_plans + accumulate_trajs
            end_points = [trajectory[-1][:3] for trajectory in all_trajs]
            indexies = select_diverse_points(end_points, num_trajs)
            selected_trajs = [all_trajs[i] for i in indexies]

            return selected_trajs

        return proposer_fn
    
    def generate_model_response(self, msgs, task_type, query_num=1, **kwargs):
        out = self.model.respond(msgs, n=query_num, **kwargs)
        return out
    
    def _parser_response(self, out, type, candidate_idxs=None):
        if type == "descriptor":
            return self._parser_descriptor_response(out, candidate_idxs)
        elif type == "evaluator":
            return self._parser_evaluator_response(out, candidate_idxs)
        else:
            raise ValueError(f"Unknown parser type: {type}")

    def _parser_evaluator_response(self, out, candidate_idxs=None):
        # try to parse the output text as JSON
        json_object = repair_json(out, return_objects=True)
        keys = ("task_goal", "reasoning", "current_best_plan")
        choice_idx = json_object["current_best_plan"]
        assert all(key in json_object for key in keys), "The output JSON is missing some items"
        assert isinstance(choice_idx, int), "The current_best_plan should be an integer"

        if choice_idx == -1:
            logger.info("No valid plan found. Choosing last proposed plan...")
            choice_idx = -1
        else:
            assert (candidate_idxs is None) or (choice_idx in candidate_idxs), "The current_best_plan should be in the candidate_idxs"
        # assert isinstance(json_object["fully_achieved"], bool), "The fully_achieved should be a boolean value"

        logger.info("Collected an effective evaluation")
        return choice_idx, json_object

    def _parser_descriptor_response(self, out, candidate_idxs=None):
        # try to parse the output text as JSON
        json_object = repair_json(out, return_objects=True)
        keys = ("scene_description", "action_trajectory_description")
        assert all(key in json_object for key in keys), "The output JSON is missing some items"

        logger.info("Collected an effective evaluation")
        return None, json_object
    
    def query_vlm(self, msgs, query_num, temperature, task_type, candidate_idxs=None):
        """Query VLM and parser the output to correct format."""
        T = 0; max_try = 3
        collected_temp, parsered_jsons = [], []
        while T < max_try:
            try:
                outs = self.generate_model_response(
                    msgs=msgs,
                    query_num=query_num,
                    temperature=temperature,
                    task_type=task_type,
                )
                logger.info(f"==> Response from query_vlm:\n{outs}\n")
                assert isinstance(outs, list), "Model response should be a list of outputs."

                for out in outs:
                    try:
                        index, json = self._parser_response(
                            out, task_type,
                            candidate_idxs=candidate_idxs
                        )

                    except Exception as e:
                        logger.info("-" * 100)
                        traceback.print_exc()
                        logger.info("-" * 100)
                        continue

                    collected_temp.append(index)
                    parsered_jsons.append(json)
                    if len(collected_temp) >= query_num:
                        break

                if len(collected_temp) < query_num:
                    logger.info("Not enough instructions, retrying...")
                    raise ValueError("Not enough instructions")

                return collected_temp, parsered_jsons
            except Exception as e:
                logger.info("-" * 100)
                traceback.print_exc()
                logger.info("-" * 100)
                T += 1
                time.sleep(1 * T)  # Exponential backoff

        # Finally, if we still fail to get a valid action, return a random action
        logger.error(f"Failed to get a valid choice after {T} attempts, returning 0.")
        return [0], ["No response received after multiple attempts."]

    def save_chat_log(self, current_obs_path, response_texts, messages=None):
        if messages is None:
            messages = self.episode_messages
        chat_logs = []
        for response_text in response_texts:
            chat_log = format_chat_dialog(messages, response_text)
            chat_logs.append(chat_log)

        chat_path = os.path.join(os.path.dirname(current_obs_path), "chat_log.json")
        # append a timestamp to avoid overwriting
        timestamp = _dt.datetime.now().strftime("%m%d_%H%M%S")
        chat_path = chat_path.replace(".json", f"_{timestamp}.json")
        os.makedirs(os.path.dirname(chat_path), exist_ok=True)
        with open(chat_path, "w", encoding="utf-8") as f:
            json.dump(chat_logs, f, indent=2, ensure_ascii=False)

    def gen_pred_image(self, curr_obs, obs_path, action_plans, text_plans, curr_pose):
        text_plans = [plan["language_plan"] for plan in text_plans]
        B = len(action_plans)
        # * 1. interpolation -> batch_action_plans_: Float[NDArray, "B 14 8"]
        seq_len = len(action_plans[0])
        assert all(len(plan) == seq_len for plan in action_plans), f"All action plans must have the same length, but got {[len(plan) for plan in action_plans]}"
        logger.info(f"Generating {B} imagined trajectories with {seq_len} steps each.")

        uni_samp = 14  # Number of uniformly sampled actions
        idxs = np.linspace(0, seq_len - 1, uni_samp).astype(int)
        batch_action_plans_ = [[plan[idx] for idx in idxs] for plan in action_plans]

        # interval is set for 3 for anchor_idx_lists:
        anchor_idx_lists = [list(range(0, uni_samp, 3))] * B
        anchor_idx_lists = [[-1]] * B
        image: Float[Tensor, "B C H W"] = curr_obs.unsqueeze(0).repeat(B, 1, 1, 1)
        image: UInt8[Tensor, "B C H W"] = (image * 255.0).to(torch.uint8)  # convert to uint8

        # * 2. Generate frames for the all actions with batch
        if not hasattr(self, "igenex_sock"):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(
                (self.igenex_host.split(":")[0], int(self.igenex_host.split(":")[1]))
            )
            self.igenex_sock = sock
            logger.info(f"[Client] Connected igenex_socket manager at {self.igenex_host}")

        # create the pred save dir (consistent with convention in Igenex)
        vis_dir = get_save_dir_from_existing(obs_path)
        save_dirs = get_igenex_save_dirs(vis_dir, list(range(B)))

        output_dict = self.imagine_by_model_type(batch_action_plans_, text_plans, image, save_dirs)
        # * 3. postprocess and save the predicted images
        save_action_sequence(
            b_action=[[action_plans[i], batch_action_plans_[i]] for i in range(len(action_plans))],
            save_dirs=save_dirs
        )
        pred_obs_paths = self.post_process_output(output_dict, anchor_idx_lists, action_plans)

        return pred_obs_paths
    
    def get_best_action(
        self,
        imagine_traj,
        action_traj,
        current_obs,
        user_instruction,
        task_name,
        query_num,
        is_final_iter=False,
    ):
        if is_final_iter:
            task_prompt_eval = Igenex_evaluator_prompt_norepeat
        else:
            task_prompt_eval = Igenex_evaluator_prompt
        task_prompt_desc = Igenex_descriptor_prompt

        few_shot_prompt_eval = genex_vlm_few_shot_examples[task_name]["evaluator"][:self.shot_example_num]
        few_shot_prompt_desc = genex_vlm_few_shot_examples[task_name]["descriptor"][:3*self.shot_example_num]
        assert len(few_shot_prompt_eval) == self.shot_example_num == len(few_shot_prompt_desc)/3, \
            f"Expected {self.shot_example_num} few-shot examples, got {len(few_shot_prompt_eval)}"

        # * 1. get description of each action trajectory first:
        episode_messages = self.get_descriptor_message(
            imagine_traj, action_traj, task_prompt_desc, current_obs,
            user_instruction,
            few_shot_prompt_desc
        )
        # ---- descriptor phase (parallelized) ----
        concurrency = int(self.vlm_args.get("concurrency", 8))
        calls = [
            ((), {  # args tuple is empty; we use kwargs only
                "msgs": msg,
                "query_num": query_num,
                "temperature": self.vlm_args["temperature"],
                "task_type": "descriptor",
            })
            for msg in episode_messages
        ]
        results = run_concurrent_blocking(self.query_vlm, calls, concurrency=concurrency)

        scene_descriptions, descriptions, all_jsons_flat = [], [], []
        for idx, (_indices, jsons_i) in enumerate(results):
            scene_descriptions.append(jsons_i[0]["scene_description"])
            descriptions.append(
                f"Candidate Action Plan <{idx}>: " + jsons_i[0]["action_trajectory_description"]
            )
            all_jsons_flat.extend(jsons_i)

        visual_state = compose_visual_state(scene_descriptions[0], descriptions)
        self.save_chat_log(current_obs, all_jsons_flat, messages=episode_messages)

        # * 2. get the final evaluation and choice
        episode_messages = self.get_evaluator_message(
            imagine_traj, action_traj, task_prompt_eval, current_obs,
            user_instruction,
            few_shot_prompt_eval,
            visual_state,
        )

        response = self.query_vlm(
            msgs=episode_messages,
            query_num=query_num,
            temperature=self.vlm_args["temperature"],
            task_type="evaluator",
            candidate_idxs=list(range(len(imagine_traj)))
        )

        # save the chat log
        self.save_chat_log(current_obs, response[1], messages=episode_messages)
        return response
    
    def get_descriptor_message(self, imagine_traj, imagine_action_traj, task_prompt, real_obs,
                                user_instruction, few_shot_prompt):
        prev_messages = []
        # * 1. add general prompt (same as no manip_planner)
        prev_messages.append(
            self._build_user_message(
                task_prompt, [],
            )
        )

        # * 2. add few-shot examples
        few_shot_str = ",".join(few_shot_prompt)
        prev_messages.append(
            self._build_user_message(
                f"\n\n**Few-Shot Examples:**:\n{few_shot_str}", []
            )
        )

        # * 3. add the predicted image and action plan pairs
        final_messages = [deepcopy(prev_messages) for _ in range(len(imagine_traj))]
        for ith_plan in range(len(imagine_traj)):
            if ith_plan == 0:
                prefix = "\n\n**Current Candidate Trajectories**:\n"
            else:
                prefix = ""

            action_plan_text = f"{prefix}"
            final_messages[ith_plan].append(
                self._build_user_message(
                    action_plan_text, imagine_traj[ith_plan], img_first=False
                ),
            )

        return final_messages


    def get_evaluator_message(self, imagine_traj, imagine_action_traj, task_prompt, real_obs,
                              user_instruction, few_shot_prompt, visual_state):
        messages = []
        # * 1. add general prompt (same as no manip_planner)
        messages.append(
            self._build_user_message(
                task_prompt, [],
            )
        )

        # * 2. add few-shot examples
        few_shot_str = ",".join(few_shot_prompt)
        messages.append(
            self._build_user_message(
                f"\n\n**Few-Shot Examples:**:\n{few_shot_str}\n", []
            )
        )

        # * 3. add the auxiliary prompt for describing the pred obs from WMs
        messages.append(
            self._build_user_message(
                f"\n\n**Current Task Instructions**:\n{user_instruction}\n", []
            )
        )
        # assert len(imagine_traj) == len(imagine_action_traj)
        # assert type(images[0]) == List and type(actions[0]) == List

        # * 4. add the predicted image and action plan pairs
        action_plan_text = f"\n\n**Current Visual Descriptions of Candidate Action Plans**:\n{visual_state}"
        messages.append(
            self._build_user_message(
                action_plan_text, [],
            ),
        )

        return messages

    # # -------------- new added methods --------------
    def query_igenex(
        self,
        img_path_list_anno,
        user_instruction,
        img_path_list_origin,
        curr_obs,
        proposer_fn,
        task_name,
    ):
        st = self.st
        gripper_pose = _get(curr_obs, "gripper_pose")   # (3,) or (n,)
        gripper_open = _get(curr_obs, "gripper_open")   # scalar
        curr_pose = np.concatenate([gripper_pose, [gripper_open]], axis=0)
        if type(img_path_list_anno) == dict:
            obs_path = img_path_list_anno[self.obs_key][0]
        else:
            obs_path = img_path_list_anno[0] # input image paths
        obs_path_ori = img_path_list_origin[0]
        self.add_current_step_to_state(st, obs_path_ori, obs_path)

        # Clean previous imagined obs/actions and potential plans
        st.clean_up_history(key=self.imagine_obs_key)
        st.clean_up_history(key=self.imagine_action_key)

        acum_trajs = []
        need_resample = True   # trigger first proposal inside the loop
        actions_plans = None
        reasons = [{"language_plan": "No reason available"}]  # placeholder; overwritten on success

        for iter_count in range(self.max_iterations):
            # (0) propose trajectories once per needed round
            if need_resample:
                actions_plans_valid = proposer_fn(
                    num_trajs=self.proposal_num, accumulate_trajs=acum_trajs
                )
                need_resample = False

            # (1) default reasons per plan (kept as in original)
            reasons_valid = [{"language_plan": "No reason available"}] * len(actions_plans_valid)
            acum_trajs.extend(actions_plans_valid)
            logger.info(f"[iter {iter_count}] New Proposed Action plan num <{len(actions_plans_valid)}>")

            # (2) generate predicted observations
            curr_obs_origin = st.fetch_current_state_obs(self.obs_origin_key)
            obs_path = st.get_from_history(self.obs_origin_key)[-1]
            pred_obs_paths = self.gen_pred_image(
                curr_obs_origin, obs_path, actions_plans_valid, reasons_valid, curr_pose,
            )

            # (3) VLM selection on generated obs + action plans
            current_obs = st.get_from_history(self.obs_key)[-1]
            is_final_iter = (iter_count == self.max_iterations - 1)

            plan_indices, reasons = self.get_best_action(
                imagine_traj=pred_obs_paths,        # list [n, B, C, H, W]
                action_traj=actions_plans_valid,    # list [n, B, 7], 7 is for action dim
                current_obs=current_obs,
                user_instruction=user_instruction,
                task_name=task_name,
                query_num=1,
                is_final_iter=is_final_iter,
            )
            assert len(plan_indices) == 1, "only one plan index is allowed to be returned now"

            # (4) resample or finalize
            if plan_indices[0] == -1 and not is_final_iter:
                need_resample = True
                continue

            actions_plans = [actions_plans_valid[idx] for idx in plan_indices]
            reasons = reasons
            break

        return actions_plans, reasons