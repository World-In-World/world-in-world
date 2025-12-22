import json
import os
import os.path as osp
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from jaxtyping import Float, Int32, UInt8
from numpy.typing import NDArray
from PIL import Image
from torch import Tensor
from tqdm import tqdm, trange
import shutil

from downstream.downstream_datasets import IGDataset
from downstream.prompts import AR_ACTION_SPACE_NO_STOP, VLM, UNIT_DEGREE, UNIT_DISTANCE
from downstream.visualize import (
    visualize_ar_baseline,
    annotate_frame,
)
from downstream.vlm import COMMERCIAL_MODELS, LOCAL_MODELS, WORLD_MODEL_TYPES
from downstream.utils.saver import get_igenex_save_dirs, Saver, format_chat_dialog
from downstream.utils.state_traj import State
from downstream.evaluator import compute_vln_eval_metrics
from downstream.utils.igenex_util import (
    IGENEX_ACTION_IDS,
    compose_turn_actions,
)
from downstream.utils.util import (
    format_time, log_metric,
    compute_rot_difference,
)
from downstream.process_IGnav_dataset.pickle_dataset import load_igdataset_from_zip
from utils.logger import setup_logger
from utils.svd_utils import rotate_coord_by_degrees, rotate_by_degrees
from torchvision.transforms.functional import to_pil_image, to_tensor
from downstream.solver_base import Solver, build_common_arg_parser, launch_multiprocessing
from utils.util import is_empty
from downstream.utils.util import get_distance, calc_traj_distance
from habitat_data.equi2cube import convert_equi2per
from evaluation.FVD.cal_4metrics import calculate_lpips


class IGSolver(Solver):
    def __init__(
        self,
        max_actions: int,
        max_decisions: int,
        api_key: str,
        use_heur: bool,
        use_WM: bool,
        planner: str,
        answerer: str,
        obs_key: List[str],
        recog_thres: float,
        parallel_ith: int,
        parallel_total: int,
        WM_host: str,
        sam2_host: str,
        debug_len=None,
        args=None,
    ):
        # * data
        self.args = args
        self.dataset = load_igdataset_from_zip(
            "data/WIW_datasets/eval_datasets/IGNav/igdataset_goal_imgs.zip",
            "data/WIW_datasets/eval_datasets/IGNav/episodes_IGNav.json.gz",
        )

        self.parallel_ith = parallel_ith
        all_devices = torch.cuda.device_count()
        self.device = parallel_ith % all_devices
        self.parallel_total = parallel_total
        self.WM_host = WM_host
        self.sam2_host = sam2_host
        num_ofvlm = len(self.args.vllm_host)
        self.base_url = f"http://{self.args.vllm_host[self.parallel_ith % num_ofvlm]}/v1"
        # self.set_parallel_dataset(parallel_ith, parallel_total)
        self.args.vlm_input_format = self.set_vlm_input_format(planner, answerer)

        # Create a Saver instance to handle path calls
        self.task = "IGNav"
        self.saver = Saver(parallel_ith, parallel_total, self.args.exp_id, self.task)

        # * logging
        self.metrics_path = self.saver.get_metric_path()

        # * simulator
        self.sim = None
        # ^ for AR
        self.use_WM = use_WM
        self.TTS_ratio = args.TTS_ratio
        assert len(obs_key) > 0 and set(obs_key).issubset({"rgb", "rgb_front"})
        if self.use_WM:
            self.imagine_obs_key = "pred_action_seq"
            self.imagine_action_key = "action_plan"
        self.obs_key = obs_key.copy()
        self.obs_key.append("goal_image")
        self.obs_hfov = 90.0
        self.obs_height, self.obs_width = 512, int(512 * (self.obs_hfov / 90))
        self.pred_obs_height, self.pred_obs_width = 384, int(384 * (self.obs_hfov / 90))

        # * planning
        self.action_space = AR_ACTION_SPACE_NO_STOP
        self.action_space_map = {  # map action to action_id in genex NOTE: connected with IGENEX_ACTION_IDS, should be set manually
            1: self.action_space[0],
            2: self.action_space[1],
            3: self.action_space[2],
        }
        self.action_space_map_inv = {v: k for k, v in self.action_space_map.items()}
        # ["_l22.5", "_r22.5", "_l45", "_r45", "_f0.2"]
        self.sim_actions_space = {  # format [(sim_action, "suffix")]
            self.action_space[0]: [("move_forward", f"_f{UNIT_DISTANCE}")],
            self.action_space[1]: [("turn_left", f"_l{UNIT_DEGREE}")],
            self.action_space[2]: [("turn_right", f"_r{UNIT_DEGREE}")],
        }
        # compose turn_actions according to action_space:
        self.init_turn_actions = compose_turn_actions(self.sim_actions_space)
        self.init_turn_actions_inv = {
            v: k for k, v in self.init_turn_actions.items()
        }
        self.set_world_model_type()

        self.use_heur = use_heur
        self.use_LPIPS_reward = args.use_LPIPS_reward
        self.query_num = 3
        self.look_ahead_action_num = 5      #range (1, igenex_n_frame-1)
        self.igenex_n_frame = 14
        # * initialize VLM
        vlm_kwargs = dict(obs_key=obs_key)
        if planner in COMMERCIAL_MODELS and answerer in COMMERCIAL_MODELS:
            vlm_kwargs.update(
                api_key=api_key,
                classify_method="classify",
                image_use_abs=False,
            )
        else:
            assert planner in LOCAL_MODELS and answerer in LOCAL_MODELS
            vlm_kwargs.update(
                api_key="xxxx",
                classify_method="classify_plain",
                base_url=self.base_url,
                image_use_abs=False,
            )

        self.answerer = VLM(
            categories=["continue navigating", "stop"],
            model=answerer,
            top_logprobs=5,
            query_task="ignav_answerer",
            **vlm_kwargs,
        )
        self.max_decisions = max_decisions
        self.max_action_num = max_actions
        self.recog_thres = recog_thres

        # * New client
        self.set_new_vlm_client(planner, vlm_kwargs)
        self.temp = []


    def set_new_vlm_client(self, planner, vlm_kwargs):
        vlm_kwargs_N = vlm_kwargs.copy()
        vlm_kwargs_N["classify_method"] = None
        vlm_kwargs_N["look_ahead_action_num"] = self.look_ahead_action_num
        self.planner_N_action = VLM(
            categories=self.action_space,
            model=planner,
            query_task="ignav_planner_N_action",
            **vlm_kwargs_N,
        )
        self.evaluator_N_action = VLM(
            categories=self.action_space,
            model=planner,
            query_task="ignav_evaluator_N_action",
            **vlm_kwargs_N,
        )


    def inference(self):
        demo_lens, traj_lens = [], []
        is_success_list = []
        self.start_time = time.time()
        log_metric(self.metrics_path, "start")
        for ith_datum, datum in enumerate(
            tqdm(
                self.dataset,
                desc="For ith_datum",
                position=0,
            )
        ):
            datum_states_path = self.saver.get_datum_path_pref(datum)
            if os.path.exists(datum_states_path):
                print(f"[Worker {self.parallel_ith}] Datum <{datum_states_path}> already exists, skipping...")
                continue

            metric_datum_path = self.saver.get_metric_path(datum)
            is_reload = osp.exists(metric_datum_path)
            print(f"Datum: {self.saver.get_datum_path_pref(datum)}")
            if is_reload:
                # * load datum metrics
                print(f"[Worker {self.parallel_ith}] Loading datum metrics from {metric_datum_path}")
                with open(metric_datum_path, "r") as f:
                    curr_metrics = json.load(f)
            else:
                #* preparation:
                sim = self.get_simulator(datum["scene_id"],  enable_semantic=False)
                # save the goal image into the saver provied epoch path
                goal_img_path = osp.join(self.saver.get_datum_path_pref(datum), "goal_image.png")
                shutil.copy2(datum["goal_image_path"], goal_img_path)

                print(f"[Worker {self.parallel_ith}] Loaded simulator for scene: {datum['scene_id']}")  # fmt: skip

                traj, action_len = self.inference_ignav(datum, sim)

                # * calc metrics
                final_pos = traj[-1][0]
                final_rot = traj[-1][1]
                goal_dist, rad_diff, is_success = self.evaluate_goal_success(
                    datum, sim, final_pos, final_rot,
                )

                curr_metrics = dict(
                    is_success=is_success,
                    traj_len=action_len,
                    gt_traj_len=datum["step_to_goal"],
                    demo_len=datum["step_to_goal"],
                    rad_diff=rad_diff,
                    dist_to_goal=goal_dist,
                    point_traj=traj,
                )
                # * save datum metrics
                with open(metric_datum_path, "w") as f:
                    json.dump(curr_metrics, f, ensure_ascii=False)

                # * visualize
                if len(traj) > 0:
                    visualize_ar_baseline(
                        datum_dir=self.saver.get_datum_path_pref(datum),
                        key=self.obs_key,
                        label="vis_ignav",
                        vis_order="planner_only",
                        answer_file_name="answerer.json",
                        planner_file_name=f"planner_next-{self.look_ahead_action_num}.json",
                    )

            demo_lens.append(curr_metrics["demo_len"])
            traj_lens.append(curr_metrics["traj_len"])
            is_success_list.append(curr_metrics["is_success"])

            if not is_reload:
                metrics = self.evaluate(is_success_list, traj_lens, demo_lens)

        log_metric(self.metrics_path, "end")

        return metrics

    def evaluate_goal_success(self, datum, sim, final_pos, final_rot):
        goal_dist = get_distance(final_pos, datum["goal_position"], sim.pathfinder)[0]
        rad_diff = compute_rot_difference(final_rot, datum["goal_rotation"])
        assert 0 <= rad_diff <= np.pi, f"rad_diff: {rad_diff}"
        # if rad_diff <= datum["goal_rot_diff"] and goal_dist <= datum["goal_radius"]:
        if goal_dist <= datum["goal_radius"]:
            is_success=True
        else:
            is_success = False
        return goal_dist, rad_diff, is_success

    def evaluate(self, is_success, traj_lens, demo_lens):
        total = len(is_success)
        mean_sr, mean_spl = compute_vln_eval_metrics(
            s=is_success, p=traj_lens, l=demo_lens
        )
        print(f"==> Success Rate: {mean_sr:.1f}")
        print(f"==> SPL: {mean_spl:.1f}")

        time_elapsed = time.time() - self.start_time
        samples_remaining = len(self.dataset) - total
        eta = time_elapsed / total * samples_remaining
        time_elapsed_str = format_time(time_elapsed)
        eta_str = format_time(eta)
        print(f"==> Time Elapsed: {time_elapsed_str}")
        print(f"==> ETA: {eta_str}")

        metrics = dict(
            total=total,
            mean_sr=mean_sr,
            mean_spl=mean_spl,
            mean_traj_len=np.mean(traj_lens),
            mean_demo_len=np.mean(demo_lens),
            eta=eta_str,
            time_elapsed=time_elapsed_str,
        )
        print(f"=" * 100)
        print(metrics)
        print(f"=" * 100)

        # * save task metrics
        if total in [1, 2, 5] or total % 10 == 0:
            log_metric(self.metrics_path, metrics)

        return metrics

    def interact(
        self, sim, position, rotation, datum, ith_action=None, suffix="", save_imgs=True
    ):
        obs, state_imgs = self.get_observations(sim, position, rotation)
        rgb_front = to_pil_image(obs["rgb_sensor"])  # tensor can be on GPU or CPU

        # * load and annotate goal image
        goal_image_tensor = self.fetch_perspective_goal_img(
            datum, self.obs_height, self.obs_width
        )
        goal_image = to_pil_image(goal_image_tensor.squeeze(0))  # remove batch dimension

        goal_image = annotate_frame(
            goal_image, f"Goal Image",
            text_color="red", bg_color="white", align="center"
        )
        state_imgs["goal_image"] = to_tensor(goal_image)

        # * annotate rgb_front image
        rgb_front = annotate_frame(
            rgb_front, f"Current View: <Front>",
            text_color="red", bg_color="white", align="center"
        )
        state_imgs["rgb_front"] = to_tensor(rgb_front)

        all_keys = {
            "goal_image",
            "rgb", "depth",
            "rgb_front",
        }
        assert set(state_imgs.keys()) == all_keys

        if save_imgs:
            state_img_paths = self.save_on_disk(
                datum, ith_action, suffix, state_imgs, verbose=False
            )
        else:
            state_img_paths = None

        return state_img_paths, state_imgs

    def fetch_perspective_goal_img(self, datum, h_pers, w_pers):
        goal_image = Image.open(datum["goal_image_path"])
        goal_image_tensor = to_tensor(goal_image).unsqueeze(0)  # add batch dimension
        goal_image_tensor = convert_equi2per(
            goal_image_tensor,
            h_pers=h_pers,
            w_pers=w_pers,
            fov_x=self.obs_hfov,
        )
        return goal_image_tensor


    def inference_ignav(self, datum, sim):
        st, skip_flag = self.init_states(datum, sim)
        if skip_flag:
            return None, None

        position = datum["start_position"]
        rotation = datum["start_rotation"]

        st.update_position_traj((position, rotation))

        for ith_decision in trange(
            0, self.max_decisions + 1,  # fmt: skip
            desc="For ith_action",
            leave=False,
            position=1,
        ):
            if self.use_WM and np.random.random() < self.TTS_ratio:
                st = self.forward_with_WM(
                    datum, sim,     # fmt: skip
                    st, position, rotation, ith_decision, # fmt: skip
                )

            if self.use_LPIPS_reward:
                stop_flag, position, rotation, st = self._take_step_and_recognize_LPIPS(
                    datum, sim, ith_decision, st, position, rotation, # fmt: skip
                )
            else:
                stop_flag, position, rotation, st = self._take_step_and_recognize(
                    datum, sim, ith_decision, st, position, rotation, # fmt: skip
                )
            st.update_position_traj((position, rotation))

            st = self.clean_cache(st)

            if stop_flag:
                break

        return st.position_traj, sum(len(sublist) for sublist in st.get_action_traj())

    def fetch_recognize_answer(self, datum, sim, st, ith_action, answerer):
        # NOTE: Assume the answerer is performed first in loop
        answer_path = self.saver.get_answerer_output_path(datum, ith_action)

        # Gather all value-lists first
        list_of_lists = [st.get_from_history(key) for key in self.obs_key]
        states = [list(items) for items in zip(*list_of_lists)]

        if answerer is None:
            curr_pos = st.position_traj[-1][0]
            curr_rot = st.position_traj[-1][1]
            goal_dist, rad_diff, is_success = self.evaluate_goal_success(
                datum, sim, curr_pos, curr_rot,
            )
            if is_success:
                answer_probs = {"stop": 1.0}
            else:
                answer_probs = {"continue navigating": 1.0}
        else:
            messages = answerer.assemble_messages(
                states,
                st.get_action_traj(),
                enable_history=False,
            )
            answer_probs = answerer.query_VLM(
                messages=messages,
            )

        os.makedirs(osp.dirname(answer_path), exist_ok=True)
        with open(answer_path, "w") as f:
            json.dump(answer_probs, f, indent=2, ensure_ascii=False)

        answer = list(answer_probs.keys())[0]
        answer_value = answer_probs[answer]
        return answer, answer_value

    def _take_step_and_recognize(
        self,
        datum, sim,
        ith_action: int,
        st: State,
        position, rotation,
    ) -> Tuple[bool, Any, Any, State]:
        """
        1) Fetch & update recognition answer in 'st'.
        2) If recognition passes threshold, signal we should stop.
        3) Fetch & perform next move.
        4) Update 'st' with new row/action.
        """
        # 1) check whether to stop
        stop_flag = self.check_for_completion(datum, sim, ith_action, st)

        # 3) Fetch the next action & perform the move (only if not stopping yet)
        if not stop_flag:
            ith_action = ith_action + 1  # NOTE: ith_action for move starts from 1 not 0
            # if self.use_heur:
            #     action_seqs_u, _ = self.heur_sample_next_action_seqs(
            #         action_seq=st.get_action_traj(),
            #         query_num=self.query_num,
            #     )
            #     print(f"Sampled heuristic action seq: {action_seqs_u}")
            # else:
            action_seqs = self.fetch_action_decision_vlm(
                datum, st,
                ith_action,
                self.planner_N_action,
                is_proposing_process=False,
            )
            action_seqs_u, _ = self.extract_unique_action_seq(action_seqs)
            assert len(action_seqs_u) == 1
            action_seqs_u = action_seqs_u[0]

            # perform the len_action - 1 in the action sequence:
            executed_len = max(len(action_seqs_u) - 2, 1)
            for action in action_seqs_u[:executed_len]:
                position, rotation = self.perform_agent_move(
                    sim, action,
                )

            # 4) Interact with the simulator
            state, state_imgs = self.interact(
                sim, position, rotation, datum, ith_action
            )
            # Add the new state row and record the action
            st.add_new_state(state, state_imgs)
            st.record_past_action(action_seqs_u[:executed_len])

        return stop_flag, position, rotation, st

    def check_for_completion(self, datum, sim, ith_action, st):
        stop_flag = False

        # 1) Fetch recognition answer
        answer, answer_value = self.fetch_recognize_answer(
            datum, sim,
            st, ith_action,
            # self.answerer,
            answerer=None,
        )
        st.add_answer(answer, answer_value)  # auto-update best answer

        # 2) Possibly update best answer
        is_done = ("stop" in answer.lower())
        accmulate_action_len = sum(len(sublist) for sublist in st.get_action_traj())
        if is_done or accmulate_action_len >= self.max_action_num:
            stop_flag = True
            st.set_best_answer(is_done)
        return stop_flag

    def _take_step_and_recognize_LPIPS(
        self,
        datum, sim,
        ith_action: int,
        st: State,
        position, rotation,
    ) -> Tuple[bool, Any, Any, State]:
        """
        1) Fetch & update recognition answer in 'st'.
        2) If recognition passes threshold, signal we should stop.
        3) Fetch & perform next move.
        4) Update 'st' with new row/action.
        """
        # 1) check whether to stop
        stop_flag = self.check_for_completion(datum, sim, ith_action, st)

        # 3) Fetch the next action & perform the move (only if not stopping yet)
        if not stop_flag:
            ith_action = ith_action + 1  # NOTE: ith_action for move starts from 1 not 0
            action_seqs_u = self.fetch_action_by_LPIPS(st)

            # perform the len_action - 1 in the action sequence:
            executed_len = max(len(action_seqs_u) - 2, 1)
            for action in action_seqs_u[:executed_len]:
                position, rotation = self.perform_agent_move(
                    sim, action,
                )

            # 4) Interact with the simulator
            state, state_imgs = self.interact(
                sim, position, rotation, datum, ith_action
            )
            # Add the new state row and record the action
            st.add_new_state(state, state_imgs)
            st.record_past_action(action_seqs_u[:executed_len])

        return stop_flag, position, rotation, st

    def fetch_action_by_LPIPS(self, st):
        # Fetch data
        goal_img = st.fetch_current_state_obs("goal_image").unsqueeze(0)  # [1, C, H, W]
        origin_imagines = st.get_from_history(key="origin_imagine")[0]    # List[List[Tensor CxHxW]]
        action_seq_candidates = st.get_from_history(key="origin_action_plan")[0]

        # 1) Handle empty candidates; use first non-empty frame to set resolution
        assert len(origin_imagines) > 0, "No candidate imaginations found"
        valid_indices = [i for i, frames in enumerate(origin_imagines) if len(frames) > 0]
        assert len(valid_indices) != 0

        first_i = valid_indices[0]
        tgt_h, tgt_w = origin_imagines[first_i][0].shape[-2], origin_imagines[first_i][0].shape[-1]
        goal_img_resized = torch.nn.functional.interpolate(
            goal_img, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False
        )  # [1, C, H, W]

        # 2) Pad each valid candidate to the max length by repeating last frame
        T_max = max(len(origin_imagines[i]) for i in valid_indices)
        padded_videos = []
        for i in valid_indices:
            frames = origin_imagines[i]
            if len(frames) < T_max:
                frames = frames + [frames[-1]] * (T_max - len(frames))
            padded_videos.append(torch.stack(frames, dim=0))  # [T_max, C, H, W]
        gen_videos = torch.stack(padded_videos, dim=0)  # [B_valid, T_max, C, H, W]
        B_valid, T = gen_videos.shape[0], gen_videos.shape[1]

        # 3) Repeat the goal image for each frame and candidate
        gt_videos = goal_img_resized.unsqueeze(0).repeat(B_valid, T, 1, 1, 1)

        # 4) Use the shared calculate_lpips (expects [0,1] inputs)
        lpips_result = calculate_lpips(
            videos1=gt_videos,
            videos2=gen_videos,
            device="cpu",
            only_final=True,
        )
        # 5) Per-candidate rewards from returned per-video LPIPS
        per_video = lpips_result["per_video"]
        rewards_map = {idx: 1.0 - float(v) for idx, v in zip(valid_indices, per_video)}
        # Assign very low reward to empty candidates to avoid selection
        for i, frames in enumerate(origin_imagines):
            if len(frames) == 0:
                rewards_map[i] = float('-inf')

        # Pick best across all candidates
        best_candidate_idx = max(rewards_map, key=rewards_map.get)
        best_reward = rewards_map[best_candidate_idx]
        print(f"best_candidate_idx: {best_candidate_idx} with reward: {best_reward:.4f}")
        return action_seq_candidates[best_candidate_idx]


    def fetch_action_decision_vlm(
        self, datum, st, ith_action, planner, is_proposing_process=True, query_num=1
    ):
        """Fetch the action decision from the planner."""
        action_path = self.saver.get_planner_output_path(
            datum, ith_action, action_num=planner.look_ahead_action_num,
            postfix="_proposal" if is_proposing_process else "",
        )
        chat_log_path = self.saver.get_chat_log_output_path(action_path)
        if self.use_WM and not is_proposing_process:
            imagine_traj = st.get_from_history(key=self.imagine_obs_key)
            imagine_plan = st.get_from_history(key=self.imagine_action_key)
            obs_keys = self.obs_key.copy()
            obs_keys.remove("rgb_front")
            obs_keys.remove("rgb")

            planner = self.evaluator_N_action
        else:
            imagine_traj, imagine_plan = [], []
            obs_keys = self.obs_key

        # Gather all value-lists first
        list_of_lists = [st.get_from_history(key) for key in obs_keys]
        states = [list(items) for items in zip(*list_of_lists)]

        messages = planner.assemble_messages(
            states,
            st.get_action_traj(),
            enable_history=False,
            imagine_traj=imagine_traj,
            imagine_action_traj=imagine_plan,
        )
        action_probs = planner.query_VLM(
            messages=messages,
            query_num=query_num,
        )
        chat_log = format_chat_dialog(messages, action_probs)
        os.makedirs(osp.dirname(action_path), exist_ok=True)
        with open(action_path, "w") as f:
            json.dump(action_probs, f, indent=2, ensure_ascii=False)
        with open(chat_log_path, "w") as f:
            json.dump(chat_log, f, indent=2, ensure_ascii=False)

        if isinstance(action_probs, dict):
            action = list(action_probs.keys())[0]
        elif isinstance(action_probs, list):
            action = action_probs
        return action


    def forward_with_WM(self, datum, sim, st, position, rotation, ith_action):
        # * 0: generate look_ahead action seq
        if self.use_heur:
            prev_all_actions = [
                act for sublist in st.get_action_traj()
                for act in sublist
            ]
            action_seqs_u, action_seqs_u_ori = self.heur_sample_next_action_seqs(
                action_seq=prev_all_actions,
                query_num=self.query_num,
            )
            # save the sampled action seqs
            action_path = self.saver.get_planner_output_path(
                datum, ith_action, action_num=self.planner_N_action.look_ahead_action_num,
                postfix="_proposal",
            )
            os.makedirs(osp.dirname(action_path), exist_ok=True)
            with open(action_path, "w") as f:
                json.dump([action_seqs_u, action_seqs_u_ori], f, indent=2, ensure_ascii=False)
            print(f"Sampled heuristic action seq: {action_seqs_u}")
        else:
            action_seqs = self.fetch_action_decision_vlm(
                datum, st,
                ith_action,
                self.planner_N_action,
                is_proposing_process=True,
                query_num=self.query_num,
            )
            action_seqs_u, action_seqs_u_ori = self.extract_unique_action_seq(action_seqs)

        # * 1.1 Generate predicted frames for the candidate actions
        if len(action_seqs_u) != 0:
            priors = self.get_action_info_from_prior(action_seqs_u)
            init_turn_degrees, prior_action_ids, origin_action_ids = priors
        else:
            init_turn_degrees, prior_action_ids = self.get_action_seqs_noprior()

        output_dict0 = self.imagine_by_model_type(
            datum, st,      # fmt: skip
            ith_action,
            init_turn_degrees, prior_action_ids, origin_action_ids, # fmt: skip
            init_rotate_type="by_shift",
        )
        save_dirs, pred_frames, coord_type = (
            output_dict0["save_dirs"],
            output_dict0["pred_frames"],
            output_dict0["coord_type"]
        )

        # * 1.2 align frames / get front view fron panos
        fn_postprocess = self.get_postprocess_fn(self.task, coord_type)
        rgbs_wo_mask = fn_postprocess(
            output_dict0, per_hfov=self.obs_hfov,
            start_idx=0, img_size=(self.pred_obs_height, self.pred_obs_width),
        )

        init_rgbs, init_turn_actions = self.get_init_frames_from_init_degrees(
            st.fetch_current_state_obs("rgb"), init_turn_degrees,
        )
        # merge the actions:
        actions_all, rgbs_wo_bbox_all = self.get_merged_preds(
            prior_action_ids, rgbs_wo_mask, init_rgbs, init_turn_actions
        )

        pred_save_paths = self.compose_action_results_from_preds(
            save_dirs, rgbs_wo_bbox_all, actions_all,
        )
        st.add_to_recent_state(pred_save_paths, key=self.imagine_obs_key)
        if self.use_LPIPS_reward:
            st.clean_up_history(key="origin_imagine")
            st.clean_up_history(key="origin_action_plan")
            origin_imagine = [v[1:] for v in list(rgbs_wo_bbox_all.values())]  # remove the first frame
            origin_action_plans = [v[1:] for v in list(actions_all.values())]  # remove the first action
            st.add_to_recent_state(origin_imagine, key="origin_imagine")
            st.add_to_recent_state(origin_action_plans, key="origin_action_plan")
        st.add_to_recent_state(
            [{f"Action Plan {i+1}": plan} for i, plan in enumerate(action_seqs_u_ori)],
            key=self.imagine_action_key,
        )

        return st


    def compose_action_results_from_preds(
        self, save_dirs, rgbs_w_bbox_all, actions
    ):
        """
        Compose the action results from the predicted frames and save them to disk.
        """
        pred_save_paths = []
        # assert len(save_dirs) == len(rgbs_w_bbox_all)

        for i, (id, rgbs) in enumerate(rgbs_w_bbox_all.items()):
            action_results_frames = []
            all_idxs = torch.arange(0, len(actions[id]))

            for idx in all_idxs:       #NOTE: retain_idxs[i] can be len 0
                action_str = actions[id][idx]
                curr_frame = rgbs[idx]

                if idx != 0:
                    action_str = f"Imagined action <{idx}>: {action_str}"

                # text "actions" on the top center with pink color
                pil_img = to_pil_image(curr_frame)
                anno_frame = annotate_frame(
                    pil_img, action_str,
                    font_size=18,
                )
                curr_frame = to_tensor(anno_frame)

                action_results_frames.append(curr_frame)

            pred_save_path = osp.join(save_dirs[i], f"{self.imagine_obs_key}.png")
            action_results_frames = torch.stack(action_results_frames, dim=0)
            pred_save_path = self.save_vlm_input_media(action_results_frames, pred_save_path)
            pred_save_paths.append(pred_save_path)

        return pred_save_paths

    def get_init_frames_from_init_degrees(self, rgb, init_turn_degrees):
        if isinstance(rgb, np.ndarray):
            rgb: UInt8[NDArray, "H W C"] = rgb
            assert rgb.shape[2] == 3, f"Expected RGB shape (H, W, 3), got {rgb.shape}"
            rgb = torch.einsum("hwc->chw", torch.from_numpy(rgb))
        elif isinstance(rgb, Tensor):
            rgb: UInt8[Tensor, "C H W"] = rgb
            assert rgb.shape[0] == 3, f"Expected RGB shape (C, H, W), got {rgb.shape}"

        init_rgbs = defaultdict(list)
        init_turn_actions = defaultdict(list)
        for i, (id, degrees) in enumerate(init_turn_degrees.items()):
            # First append the original frame
            init_rgbs[id].append(rgb)
            init_turn_actions[id].append("It is the current observation before acting")

            num_turn = abs(int(degrees // UNIT_DEGREE))
            unit_turn_degree = UNIT_DEGREE if degrees > 0 else -UNIT_DEGREE
            rgb_r = rgb.clone()

            for j in range(num_turn):
                rgb_r = rotate_by_degrees(rgb_r, unit_turn_degree)
                init_rgbs[id].append(rgb_r)
                action = self.init_turn_actions_inv[unit_turn_degree]
                init_turn_actions[id].append(action)

            init_rgbs_front = torch.stack(init_rgbs[id], dim=0)
            init_rgbs_front: UInt8[Tensor, "B C H W"] = convert_equi2per(
                init_rgbs_front,
                h_pers=self.pred_obs_height,
                w_pers=self.pred_obs_width,
                fov_x=self.obs_hfov
            )
            init_rgbs[id] = [frame for frame in init_rgbs_front]
        return init_rgbs, init_turn_actions


def run_solver_process(parallel_ith, args, api_key):
    setup_logger(
        osp.join(
            args.log_output_dir, f"{args.exp_id}", f"subProcess_{parallel_ith}.log"
        )
    )
    print(f"Logger set up for worker {parallel_ith}")
    print(f"All args:\n {args}")

    solver = IGSolver(
        max_actions=1000,
        max_decisions=20,
        api_key=api_key,
        use_heur=args.use_heur,
        debug_len=300,
        use_WM=args.use_WM,
        recog_thres=0.8,
        obs_key=["rgb_front", "rgb"],  # ["rgb_bbox_front" "rgb_bbox"]
        answerer=args.answerer_model,
        planner=args.planner_model,
        parallel_ith=parallel_ith,
        parallel_total=args.worker_num,
        WM_host=args.WM_host,
        sam2_host=args.sam2_host,
        args=args,
    )

    result = solver.inference()
    print(f"Worker {parallel_ith} result: {result}")
    return result



if __name__ == "__main__":
    parser = build_common_arg_parser()
    # Task-specific options
    parser.add_argument("--use_heur", action="store_true", help="Use heuristic policy or not")
    parser.add_argument("--use_LPIPS_reward", action="store_true", help="Use LPIPS metric as a reward model")
    parser.add_argument("--TTS_ratio", type=float, default=1.0, help="Probability of running forward_with_WM (0.1 means 10% probability)")

    args, unused_cli_tokens = parser.parse_known_args()

    launch_multiprocessing(args, run_solver_process)
