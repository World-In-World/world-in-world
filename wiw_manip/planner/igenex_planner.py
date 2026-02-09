import os.path as osp
import os
import numpy as np
import torch
import socket
import json
import logging
from typing import Any, Dict, List
from numpy.typing import NDArray
from torch import Tensor
from jaxtyping import Float, Int32, UInt8
from wiw_manip.envs.eb_man_utils import (
    get_continous_action_from_discrete,
    get_continous_action_from_discrete_batch,
)
from collections import Counter
from wiw_manip.planner.vlm_planner import VLMPlanner
from wiw_manip.planner.utils.planner_utils import (
    local_image_to_data_url,
    template_manip,
    template_lang_manip,
    interpolate_7dof_pose,
    _get, _plan_to_key,
)
from torchvision.utils import save_image

from wiw_manip.main import logger
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
from wiw_manip.planner.utils.worker_manager import read_framed, write_framed, check_inputdict
from wiw_manip.planner.utils.state_traj import State
from wiw_manip.planner.utils.visualize import annotate_frame
from wiw_manip.planner.utils.query_utils import encode_img_to_base64
from wiw_manip.planner.utils.saver import (
    get_igenex_save_dirs,
    get_save_dir_from_existing,
    save_action_sequence,
)

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

class IgenexPlanner(VLMPlanner):

    def __init__(
        self,
        model_name,
        model_type,
        system_prompt,
        revise_aux_prompt,
        pred_img_size,
        examples,
        n_shot=0,
        obs_key="front_rgb",
        chat_history=False,
        language_only=False,
        multiview=False,
        multistep=False,
        visual_icl=False,
        tp=1,
        executed_action_per_step=1,
        proposal_num=2,
        igenex_host="localhost:6000",
        mpc_mode="ranking",  # "iterative" or "ranking"
        max_iterations=4,
        evaluator_examples=[],
        exp_name=None,
        **kwargs,
    ):
        super().__init__(
            model_name,
            model_type,
            system_prompt,
            examples,
            n_shot,
            obs_key,
            chat_history,
            language_only,
            multiview,
            multistep,
            visual_icl,
            tp,
            executed_action_per_step,
            **kwargs,
        )
        self.system_prompt = system_prompt
        self.revise_system_prompt = revise_aux_prompt

        self.proposal_num = proposal_num
        self.igenex_host = igenex_host
        self.mpc_mode = mpc_mode
        self.max_iterations = max_iterations
        self.img_height, self.img_width = pred_img_size, pred_img_size
        self.save_separate_imgs = False
        self.evaluator_examples = evaluator_examples
        self.exp_id = exp_name
        self.set_world_model_type()
        if mpc_mode == "ranking":
            self.max_iterations = 1

        self.imagine_obs_key = "front_rgb_pred"
        self.imagine_action_key = "action_seq_pred"
        self.potential_plan_key = "action_seq_pred_potential"
        self.curr_query_plan_key = "action_seq_pred_current"
        self.obs_origin_key = "front_rgb_origin"
        self.reset()

    def set_world_model_type(self):
        world_model_type = None
        for type, models in WORLD_MODEL_TYPES.items():
            for model in models:
                if f"_{model}" in self.exp_id:
                    world_model_type = type
                    logger.info(f"World model type is set to {world_model_type}.")
                    break
        if world_model_type is None:
            logger.warning("World model type is not specified. Using default <igen>.")
            world_model_type = "action"
        self.world_model_type = world_model_type

    def reset(self):
        # at the beginning of the episode
        super().reset()
        all_keys = {
            self.obs_key,       # "front_rgb"
            self.obs_origin_key,
            self.imagine_obs_key,
            self.imagine_action_key,
        }
        self.st = State(list(all_keys))

    def gen_pred_image(self, curr_obs, obs_path, action_plans, text_plans, curr_pose):
        text_plans = [plan["language_plan"] for plan in text_plans]
        batch_action_plans = [get_continous_action_from_discrete_batch(plan) for plan in action_plans]
        B = len(batch_action_plans)
        # * 1. interpolation -> batch_action_plans_: Float[NDArray, "B 14 8"]
        batch_action_plans_, anchor_idx_lists = self._construct_action_seqs(
            curr_pose, batch_action_plans, out_seq_len=14
        )
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

    def imagine_by_model_type(self, action_plans, text_plans, image, save_dirs):
        if self.world_model_type == "text":
            action_plans_ = text_plans
        elif self.world_model_type == "action":
            action_plans_ = action_plans
        else: raise ValueError(f"Unknown world model type: {self.world_model_type}")

        input_dict = {
            "b_image": image.numpy(),  # image is uint8[ndarray, "b 3 384 384"], so return a list
            "b_action": action_plans_,
            "save_dirs": save_dirs,
            "return_objects": [True]*len(save_dirs),
        }
        check_inputdict(input_dict)
        write_framed(self.igenex_sock, input_dict)
        output_dict = read_framed(self.igenex_sock)
        return output_dict

    def _construct_action_seqs(self, init_pose, b_action_plans, out_seq_len=14):
        batch_action_plans, anchor_idx_lists = [], []

        for plan in b_action_plans:
            action_seq, anchor_idx_list = [], []
            current_pose = np.array(init_pose)  # initial pose without gripper state
            # distribute out_seq_len over the |plan| sub-goals
            n_sub_goals = len(plan)
            base      = out_seq_len // n_sub_goals          # minimum per segment
            remainder = out_seq_len %  n_sub_goals          # leftover frames
            # e.g. out_seq_len=14, n_sub_goals=3  → [5,5,4]
            step_num_list = [
                base + (1 if i >= n_sub_goals - remainder else 0)
                for i in range(n_sub_goals)
            ]

            for j, (end_pose, step_num) in enumerate(zip(plan, step_num_list)):
                if j == len(plan) - 1:
                    kwargs = {"num_points": step_num, "include_end": True}
                    change_idx    = step_num - 1                # last frame_idx
                    anchor_idx_list.append(out_seq_len - 1)
                else:
                    kwargs = {"num_points": step_num + 1, "include_end": False}
                    change_idx    = step_num
                    anchor_idx_list.append(
                        step_num + anchor_idx_list[-1]
                        if len(anchor_idx_list) > 0
                        else step_num
                    )

                # 1. Cartesian / joint interpolation (7-DoF, gripper handled later)
                pose_traj: Float[NDArray, "step 7"] = interpolate_7dof_pose(
                    start_pose=[current_pose[:7]],
                    end_pose  =[end_pose[:7]],
                    **kwargs
                )[0]

                # 2. Build gripper trajectory (keep then switch at last frame)
                ts = np.arange(step_num)
                gripper_start = current_pose[7]             #TODO check switch
                gripper_end   = end_pose[7]                 # or another target if needed

                gripper_seq = np.where(
                    ts < change_idx,
                    gripper_start,
                    gripper_end,
                ).astype(pose_traj.dtype)[:, None]          # (T, 1)

                # 3. Concatenate: (T, 7) → (T, 8)
                seg_traj = np.concatenate([pose_traj, gripper_seq], axis=-1)
                action_seq.append(seg_traj)
                current_pose = np.array(end_pose)       # update for next segment

            # stack segments for this batch element and store
            anchor_idx_lists.append(anchor_idx_list)
            batch_action_plans.append(np.vstack(action_seq).tolist())   # (out_seq_len, 8)

        return batch_action_plans, anchor_idx_lists

    def get_best_action(
        self,
        imagine_traj,
        action_traj,
        current_obs,
        user_instruction,
        avg_obj_coord,
        task_variation,
        query_num,
    ):
        # build message for revise
        if self.visual_icl and not self.language_only:
            raise NotImplementedError
            first_prompt, task_prompt = self.process_prompt_visual_icl(user_instruction, avg_obj_coord, prev_act_feedback=self.episode_act_feedback)
            if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or 'Qwen2.5-VL' in self.model_name:
                task_prompt += "\n\n"
                task_prompt = task_prompt + template_lang_manip if self.language_only else task_prompt + template_manip
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)
            else:
                if self.chat_history:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation, self.episode_messages)
                else:
                    self.episode_messages = self.get_message_visual_icl(obs, first_prompt, task_prompt, task_variation)
        else:
            full_example_prompt, task_prompt = self.process_prompt(
                self.system_prompt,
                user_instruction,
                avg_obj_coord,
                task_variation,
                prev_act_feedback=self.episode_act_feedback,
            )
            if ("claude" in self.model_name or "InternVL" in self.model_name
                or "Qwen2-VL" in self.model_name or "Qwen2.5-VL" in self.model_name
            ):
                task_prompt += "\n\n"
                task_prompt = task_prompt + template_lang_genex_manip if self.language_only else task_prompt + template_manip
            if len(self.episode_messages) == 0:
                self.episode_messages = self.get_revise_message(imagine_traj, action_traj, full_example_prompt, task_prompt, current_obs)
            else:
                if self.chat_history:
                    raise NotImplementedError
                    self.episode_messages = self.get_revise_message(obs, full_example_prompt, task_prompt, self.episode_messages)
                else:
                    self.episode_messages = self.get_revise_message(
                        imagine_traj, action_traj, full_example_prompt, task_prompt, current_obs
                    )

        response = self.query_vlm(None, None, None, query_num=query_num, temperature=self.vlm_args["temperature"])

        # save the chat log
        self.save_chat_log(current_obs, response[1])
        return response

    # ---------------- Message Construction Helpers ----------------
    def _build_user_message(self, prompt: str, image_paths, img_first=True):
        """
        Create a single “user”-role OpenAI message with **one prompt** (may be
        empty) followed by N images.
        """
        image_paths = [image_paths] if isinstance(image_paths, str) else image_paths
        self.__assert_paths_ok(image_paths)

        content = []
        for path in image_paths:
            data_url  = encode_img_to_base64(path)
            rel_path = osp.relpath(path, start=os.getcwd())
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url, "detail": "high",
                        "name": rel_path,
                    }
                }
            )

        if prompt:
            text = {"type": "text", "text": str(prompt)}
            content.append(text) if img_first else content.insert(0, text)

        return {"role": "user", "content": content}

    def __assert_paths_ok(self, paths: List[str]) -> None:
        """Fail fast if any path is missing or None."""
        for p in paths:
            if p is None:
                raise ValueError("Image path is required")
            if not osp.exists(p):
                raise FileNotFoundError(p)
            
    # ---------------- Message Construction Helpers ----------------

    def get_revise_message(self, imagine_traj, imagine_action_traj, full_example_prompt, task_prompt, real_obs):
        messages = []
        # * 1. add general prompt (same as no manip_planner)
        full_prompt = full_example_prompt + task_prompt
        messages.append(self._build_user_message(full_prompt, [real_obs], img_first=False))

        # * 2. add the auxiliary prompt for describing the pred obs from WMs
        messages.append(self._build_user_message(
            self.revise_system_prompt, []
        ))
        # assert len(imagine_traj) == len(imagine_action_traj)
        # assert type(images[0]) == List and type(actions[0]) == List

        # * 3. add the predicted image and action plan pairs
        for i_action in range(len(imagine_action_traj)):
            action_plan_text = f"{{\n\nHypothetical Action Plan <{i_action}>: {imagine_action_traj[i_action][1]}}}. \nSimulated Observation of Action Plan <{i_action}>: "
            messages.append(
                self._build_user_message(
                    action_plan_text, imagine_traj[i_action], img_first=False
                ),
            )

        return messages

    # -------------- new added methods --------------
    def post_process_output(self, output_dict, anchor_idx_lists, action_plans, selected_idx=-1):
        """
        1. output_dict example:
        output_dict = {
            "save_dirs": save_dirs,	                # same with input
            "video_tensors": video_tensors.tolist()	# video_tensors is Float[Tensor, "b 14 C H W"], so return a list
        }
        2. input_dict example:
        input_dict = {
            "b_image": image.tolist(),              # image is Float[Tensor, "b 3 384 384"], Note HW can only be 384x384 currently
            "b_action": batch_actions.tolist(),     # batch_actions is Float[Tensor, "b 7"], so return a list
            "save_dirs": save_dirs,                 # [list of str]
            "return_objects": [True]*len(save_dirs),# [list of True], which should have the same length as save_dirs
        }
        Return:
            return pred_frames # tensor [B C H W]
        """
        pred_frames = torch.tensor(np.array(output_dict["pred_frames"]))
        save_dirs = output_dict["save_dirs"]
        assert len(save_dirs) == len(anchor_idx_lists) == pred_frames.shape[0], \
            f"save_dirs: {len(save_dirs)}, anchor_idx_lists: {len(anchor_idx_lists)}, pred_frames: {pred_frames.shape[0]}"

        # selected_frames: Float[Tensor, "B C H W"] = pred_frames[:, selected_idx]
        pred_save_paths = []
        for save_dir, imgs, anchor_idxs, action_plan in zip(
            save_dirs, pred_frames, anchor_idx_lists, action_plans
        ):
            pil_imgs = [to_pil_image(imgs[i]) for i in anchor_idxs]
            pil_imgs_anno = [
                annotate_frame(
                    pil_img,
                    text=f"Simulation after Action <{j+1}>",    #: {action}
                    font_size=19, title_height=28,    # for 384*384
                ) # text "actions" on the top center of the img
                for j, (pil_img, action) in enumerate(zip(pil_imgs, action_plan))
            ]
            imgs_anchor = torch.stack(
                [to_tensor(img.resize((self.img_width, self.img_height)))
                 for img in pil_imgs_anno],
            )

            if self.save_separate_imgs:
                img_group = []
                for i, img in enumerate(imgs_anchor):
                    save_path = os.path.join(save_dir, f"{self.imagine_obs_key}_{i}.png")
                    save_image(img, save_path)
                    img_group.append(save_path)
                pred_save_paths.append(img_group)
            else:
                save_path = os.path.join(save_dir, f"{self.imagine_obs_key}.png")
                save_image(
                    imgs_anchor, save_path, nrow=imgs_anchor.shape[0]
                )
                pred_save_paths.append(save_path)

        # get a relative path
        save_path_rel = os.path.relpath(save_path, start=os.getcwd())
        logger.info(f"Predicted images saved to: {save_path_rel}")
        return pred_save_paths

    def parser_valid_action_plans(
        self,
        proposal_plans: List[List[Any]],
        max_allowed_action: int = 1,
        include_incomplete: bool = True,
    ) -> List[List[Any]]:
        """
        A *proposal plan* is a list of *actions* (each action can be any JSON‑serialisable
        structure, typically a list of ints/floats).  For most downstream callers we need a
        plan of exactly ``expected_action_num`` actions.  This helper makes sure of that and
        removes duplicates so the search/planning space stays small.

        * Parameters:
        proposal_plans
            A list where each element is itself a list of actions.
        max_allowed_action
            The maximum number of actions we allow in a valid plan.
        include_incomplete
            If *False* (default) we drop any plan that has fewer than
            expected_action_num actions; if *True* we keep them (handy during iterative
            planning where the agent can propose partial plans).

        * Returns:
        valid_plans
            A list of unique plans that satisfy the length constraints.
        valid_plan_idxs
            The indexes of the plans that are valid.
        """
        # 1  Validate input & accumulate into *actions_plans*
        actions_plans: List[List[Any]] = []
        for plan in proposal_plans:
            if len(plan) < max_allowed_action:
                if not include_incomplete:
                    logging.debug(f"Skipping incomplete plan: {plan}")
                    continue
            # Truncate or leave as‑is; we'll slice later for uniformity
            actions_plans.append(plan)

        # 2  Normalise length (truncate to *expected_action_num*)
        normalised_plans = [plan[:max_allowed_action] for plan in actions_plans]

        # 3  De‑duplicate while preserving order
        seen = set()
        valid_plans, valid_plan_idxs = [], []
        for i, plan in enumerate(normalised_plans):
            # Hashable representation – convert nested lists/tuples to tuples recursively
            signature = json.dumps(plan, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            valid_plans.append(plan)
            valid_plan_idxs.append(i)

        return valid_plans, valid_plan_idxs

    def check_is_repeated_plan(self, history_plans: List[List[Any]], curr_plan) -> int:
        """
        Check if the history_plan and the provided action plans are repeated.
        Return the idx of the repeated plan in the history_plans if found, otherwise return -1.
        """
        for i, history_plan in enumerate(history_plans):
            if len(history_plan) != len(curr_plan):
                continue
            if np.array_equal(history_plan, curr_plan):
                return i
        return -1

    # -------------- new added methods --------------
    def act(self,
        img_path_list_anno,
        user_instruction,
        avg_obj_coord,
        task_variation,
        img_path_list_origin,
        curr_obs,
        last_act,
        **kwargs,
    ):
        st = self.st
        gripper_pose = _get(curr_obs, "gripper_pose")   # shape (3,) or (n,)
        gripper_open = _get(curr_obs, "gripper_open")   # scalar
        curr_pose = np.concatenate([gripper_pose, [gripper_open]], axis=0)
        if type(img_path_list_anno) == dict:
            obs = img_path_list_anno[self.obs_key]
        else:
            obs = img_path_list_anno # input image paths

        obs_path = obs[0] # camera 0, i.e. front view
        obs_path_ori = img_path_list_origin[0] # same as above
        self.add_current_step_to_state(st, obs_path_ori, obs_path)

        actions_plans, reasons = super().act(
            img_path_list_anno,
            user_instruction,
            avg_obj_coord,
            task_variation,
            query_num=self.proposal_num,
            temperature=self.vlm_args["temperature"],
            last_act=last_act,
        )
        # clean prev imagined obs and action
        st.clean_up_history(key=self.imagine_obs_key)
        st.clean_up_history(key=self.imagine_action_key)
        st.clean_up_history(key=self.potential_plan_key)

        iter_count = 0
        while iter_count < self.max_iterations:
            # act should return list of shape [proposal_num, plan_length, 7]
            # 0. Collect prev explored plans
            prev_action_plans = (
                st.get_from_history(key=self.imagine_action_key)[-1]
                if len(st.get_from_history(key=self.imagine_action_key)) > 0
                else []
            )

            actions_plans_valid, valid_idxs = self.parser_valid_action_plans(
                actions_plans,
                max_allowed_action=self.executed_action_per_step
            )
            reasons_valid = [reasons[i] for i in valid_idxs]
            logger.info(f"[iter {iter_count}] New Proposed Action <{len(actions_plans_valid)}> "
                        f"plans: {actions_plans_valid}")

            # 1. Collect most potential plans, and filtered out the unexplored ones (plans without synthesized obs)
            revised_plans, revised_reasons = self.generate_revised_action_plans(
                st, prev_action_plans, actions_plans_valid, reasons_valid,
            )

            if self.should_early_terminate(iter_count, len(revised_plans)):
                break

            # 2. add new imagined  action
            curr_query_plans = prev_action_plans + [[plan, reason] for plan, reason in zip(revised_plans, revised_reasons)]
            st.add_to_recent_state(
                curr_query_plans, self.imagine_action_key, mode="replace"   #1
            )

            # 3. add predicted obs
            curr_obs_origin = st.fetch_current_state_obs(self.obs_origin_key)
            obs_path = st.get_from_history(self.obs_origin_key)[-1]
            pred_obs_paths = self.gen_pred_image(
                curr_obs_origin, obs_path, revised_plans, revised_reasons, curr_pose,
            )
            st.add_to_recent_state(
                pred_obs_paths, self.imagine_obs_key, mode="extend"         #2
            )

            # 4. query VLM with all generated obs and action plans to extend new plans
            imagine_traj = st.get_from_history(self.imagine_obs_key)[-1]  # list [n B C H W].
            action_traj = st.get_from_history(self.imagine_action_key)[-1]# list [n, B, 7], 7 is for action
            current_obs = st.get_from_history(self.obs_key)[-1]
            actions_plans, reasons = self.get_best_action(
                imagine_traj, action_traj, current_obs,
                user_instruction, avg_obj_coord, task_variation,
                query_num=self.proposal_num if self.mpc_mode == "iterative" else 1, # when mode="ranking", the final num=1
            )
            iter_count += 1

        # * Get final decision if necessary:
        if self.mpc_mode == "iterative":
            actions_plans, reasons = self.get_final_decision(
                st, current_obs
            )

        return actions_plans, reasons

    def generate_revised_action_plans(self, st, prev_action_plans, actions_plans_valid, reasons_valid):
        """
        return:
            revised_plans: List[List[Any]]
                A list of action plans that are not repeated in the previous action plans.
            revised_reasons: List[str]
                A list of reasons corresponding to the revised action plans.
        """
        prev_action_plans_ = [plan[0] for plan in prev_action_plans]
        reasons = [plan[1] for plan in prev_action_plans]
        revised_plans, revised_reasons = [], []
        potential_plans = (
            st.get_from_history(key=self.potential_plan_key)[-1]
            if len(st.get_from_history(key=self.potential_plan_key)) > 0
            else []
        )
        current_len = len(potential_plans)
        for i, (actions_plan, reason) in enumerate(zip(actions_plans_valid, reasons_valid)):
            idx = self.check_is_repeated_plan(
                prev_action_plans_, actions_plan,
            )
            st.add_to_recent_state(
                [[actions_plan, reason, current_len]], self.potential_plan_key, mode="extend"
            )
            if idx == -1:
                # if not found, we need to add the current action plan to the curr_query_plan list
                revised_plans.append(actions_plan)
                revised_reasons.append(reason)

        return revised_plans, revised_reasons

    def get_final_decision(self, st, current_obs):
        """
        Choose the action plan that should actually be executed on the robot after all MPC iterations have finished.

        Strategy
        1. Retrieve every plan ever proposed (`potential_plans`).
        2. Look at the plans that were proposed in the *last* iteration
           (`latest_action_plans`).  These are the only ones that still
           have a chance to be executed.
        3. Among those latest plans, pick the one that appeared **most
           often** throughout the whole search history
           (`potential_action_plans`).  This gives a majority-vote style
           tie-break.
        4. If there is still a tie, prefer
              (a) the shorter plan (fewer primitive actions), then
              (b) the first one in `latest_action_plans`.
        5. Persist the selected plan and reason in the chat log.  Return them to the caller.
        """
        potential_plans = st.get_from_history(self.potential_plan_key)[-1]
        if not potential_plans:
            raise RuntimeError("No potential plans recorded – cannot decide.")

        potential_action_plans = [plan[0] for plan in potential_plans]
        action_plan_reasons = [plan[1] for plan in potential_plans]
        lens_log = [plan[2] for plan in potential_plans]

        # --- 1. isolate plans proposed in the final iteration ----------
        max_len = max(lens_log)
        latest_idxs = [i for i, l in enumerate(lens_log) if l == max_len]
        latest_action_plans = [potential_action_plans[i] for i in latest_idxs]

        # --- 2. if only one candidate, we are done ----------------------
        if len(latest_action_plans) == 1:
            chosen_idx = latest_idxs[0]

        # --- 3. majority vote over the whole history --------------------
        else:
            # count occurrences of every plan in the entire search
            hist_keys = [_plan_to_key(p) for p in potential_action_plans]
            counts = Counter(hist_keys)

            # restrict to the latest plans
            latest_keys = [_plan_to_key(p) for p in latest_action_plans]
            max_count = max(counts[k] for k in latest_keys)
            candidates = [
                idx for idx, k in zip(latest_idxs, latest_keys)
                if counts[k] == max_count
            ]

            # --- 4. tie-break rules ------------------------------------
            if len(candidates) > 1:
                # (a) choose the shortest plan
                lens = [len(potential_action_plans[i]) for i in candidates]
                min_len = min(lens)
                candidates = [
                    i for i, l in zip(candidates, lens) if l == min_len
                ]

            # still tied → fall back to first occurrence
            chosen_idx = candidates[0]

        actions_plan = potential_action_plans[chosen_idx]
        reason = action_plan_reasons[chosen_idx]

        # --- 5. persist selection ---------------------------------------
        self.save_chat_log(current_obs, [reason])
        return [actions_plan], [reason]

    def should_early_terminate(self, iter_count, num_revised_plans):
        """
        Different processing logic for different MPC modes:
        """
        if self.mpc_mode == "ranking":
            enable_break = False
            assert self.max_iterations == 1
        elif self.mpc_mode == "iterative":
            if num_revised_plans == 0 or iter_count+1 >= self.max_iterations:    #early break
                enable_break = True
            else:
                enable_break = False
        else:
            raise ValueError(f"Unknown MPC mode: {self.mpc_mode}")

        return enable_break

    def add_current_step_to_state(self, st, obs_path_ori, obs_path):
        new_state_dict = {
            self.obs_key: obs_path,
            self.obs_origin_key: obs_path_ori,
        }

        # Read images and convert to tensor (C, H, W), keep original channel order
        obs_img = Image.open(obs_path).convert("RGB")
        obs_img = to_tensor(obs_img)  # C, H, W
        obs_img_ori = Image.open(obs_path_ori).convert("RGB")
        obs_img_ori = to_tensor(obs_img_ori)  # C, H, W
        state_imgs = {
            self.obs_key: obs_img,
            self.obs_origin_key: obs_img_ori,
        }
        st.add_new_state(new_state_dict, state_imgs=state_imgs)
