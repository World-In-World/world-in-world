import json
import os
import os.path as osp
import socket
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from openeqa.evaluation.llm_match import get_llm_match_score
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import save_image
from tqdm import tqdm, trange

from downstream.simulator import (
    get_simulator,
)
from data_filtering.pcd_reproject import habitat_camera_intrinsic, pos_normal_to_habitat
from downstream.downstream_datasets import AEQADataset
from downstream.prompts import (
    AR_ACTION_SPACE,
    VLM,
    UNIT_DEGREE,
    UNIT_DISTANCE,
)
from downstream.utils.igenex_util import (
    compute_theta_deviation_from_depth, filter_by_distance,
)
from downstream.utils.pcd_util import get_pointcloud_from_depth_mask
from downstream.utils.saver import get_igenex_save_dirs, Saver, format_chat_dialog
from downstream.utils.habitat_visualize import get_azimuth_from_ponts, get_azimuth_from_quat
from downstream.utils.state_traj import State
from downstream.utils.state_obj import DetectedObjects
from downstream.utils.util import (
    format_time,
    log_metric,
    rotate_agent,
    ActionFinder,
)
from downstream.utils.worker_manager import read_framed, write_framed, check_inputdict
from downstream.visualize import (
    annotate_frame,
    annotate_frame_masks,
    visualize_ar_baseline,
)
from downstream.vlm import CHOICES, COMMERCIAL_MODELS, LOCAL_MODELS, VIEW_ORDER, VIEW_ID_OFFSET, WORLD_MODEL_TYPES
from utils.logger import setup_logger
from utils.util import is_empty
from downstream.evaluator import compute_aeqa_eval_metrics, load_aeqa_demo_trajlens
from downstream.solver_base import Solver, build_common_arg_parser, launch_multiprocessing
from habitat_sim.nav import GreedyGeodesicFollower
from downstream.utils.util import get_distance, calc_traj_distance


class AEQASolver(Solver):
    def __init__(
        self,
        max_actions: int,
        api_key: str,
        use_WM: bool,
        planner: str,
        answerer: str,
        evaluator: str,
        obs_key: List[str],
        parallel_ith: int,
        parallel_total: int,
        WM_host: str,
        sam2_host: str,
        gd_sam2_host: str,
        enable_hist_planner: bool,
        enable_hist_answerer: bool,
        obs_hfov: float,
        view_order: List[str],
        surround_obs_key = [f"rgb_surround_{view}" for view in VIEW_ORDER],
        subset_size=None,
        debug_len=None,
        args=None,
    ):
        # * data
        self.debug_len = debug_len
        self.dataset = AEQADataset(
            subset_size=subset_size,
            saved_episodes_path="data/WIW_datasets/eval_datasets/AEQA/episodes_AEQA.json.gz"
        )
        self.gt_key = "answer"
        self.args = args
        self.api_key = api_key

        self.parallel_ith = parallel_ith
        all_devices = torch.cuda.device_count()
        self.device = parallel_ith % all_devices
        self.parallel_total = parallel_total
        self.WM_host = WM_host
        self.sam2_host = sam2_host
        self.gd_sam2_host = gd_sam2_host
        self.args.vlm_input_format = self.set_vlm_input_format(planner, answerer)

        num_ofvlm = len(self.args.vllm_host)
        self.base_url = f"http://{self.args.vllm_host[self.parallel_ith % num_ofvlm]}/v1"
        # self.set_parallel_dataset(parallel_ith, parallel_total)

        # Create a Saver instance to handle path calls
        self.saver = Saver(parallel_ith, parallel_total, self.args.exp_id, task="AEQA")
        self.task = "AEQA"

        # * logging
        self.metrics_path = self.saver.get_metric_path()

        # * simulator
        self.sim = None
        # * solver
        self.use_WM = use_WM
        # assert obs_key == "rgb_bbox"
        self.imagine_obs_key = "pred_perspectives"
        self.imagine_action_key = "high_level_plan_imagine"
        self.obs_key = obs_key
        self.surround_obs_key = surround_obs_key

        # ^ new for aeqa:
        self.high_level_obs_key = ["high_level_obs"]
        self.detected_obj_key = "detected_obj_ids"
        self.view_orders = view_order
        self.obs_hfov = obs_hfov
        self.obs_height, self.obs_width = 480, int(480 * (obs_hfov / 90))
        self.pred_obs_height, self.pred_obs_width = 384, int(384 * (obs_hfov / 90))
        self.surround_camera_intrinsic = habitat_camera_intrinsic(
            self.obs_width, self.obs_height, hfov=self.obs_hfov
        )

        # * planning
        self.action_space = AR_ACTION_SPACE
        self.action_space_map = {  # map action to action_id in genex NOTE: connected with IGENEX_ACTION_IDS, should be set manually
            1: self.action_space[0],
            2: self.action_space[1],
            3: self.action_space[2],
            4: self.action_space[3],
        }
        self.action_space_map_inv = {v: k for k, v in self.action_space_map.items()}
        # ["_l22.5", "_r22.5", "_l45", "_r45", "_f0.2"]
        self.sim_actions_space = {  # format [(sim_action, "suffix")]
            self.action_space[0]: [("move_forward", f"_f{UNIT_DISTANCE}")],
            self.action_space[1]: [("turn_left", f"_l{UNIT_DEGREE}")],
            self.action_space[2]: [("turn_right", f"_r{UNIT_DEGREE}")],
            self.action_space[3]: [],       #action: stop
        }
        self.sim_actions_space_inv = {
            v[0][0]: k for k, v in self.sim_actions_space.items() if len(v) > 0
        }
        self.set_world_model_type()

        self.max_action_one_seq = 4      #range (1, igenex_n_frame-1)
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

        self.max_actions = max_actions

        # * New client
        self.set_new_vlm_client(planner, vlm_kwargs)

        # * llm eval
        self.evaluator = evaluator

        # ^ episode memory
        self.enable_hist_planner = enable_hist_planner
        self.enable_hist_answerer = enable_hist_answerer

        # ^ use detected objects as map
        self.detected_objs = DetectedObjects(explore_radius=1.2)
        self.temp = []

    def set_new_vlm_client(self, planner_model, vlm_kwargs):
        vlm_kwargs_N = vlm_kwargs.copy()
        vlm_kwargs_N["classify_method"] = None
        self.highlevel_planner = VLM(
            model=planner_model,
            query_task="aeqa_highlevel_planner",
            **vlm_kwargs_N,
        )
        self.action_choices = {
            0: list(CHOICES["letter"][0: 5]),
            1: list(CHOICES["letter"][5: 10]),
            2: list(CHOICES["letter"][10: 15]),
            3: list(CHOICES["letter"][15: 20]),
        }
        labeled_dist = 1.8
        self.view_id_offset = VIEW_ID_OFFSET
        self.polar_actions: List[Tuple[float, float]] = {i: [      # view_id: (dist, angle)
            (labeled_dist, -0.30 * np.pi + self.view_id_offset[i]),
            (labeled_dist, -0.20 * np.pi + self.view_id_offset[i]),
            (labeled_dist, 0.0 * np.pi + self.view_id_offset[i]),
            (labeled_dist, 0.20 * np.pi + self.view_id_offset[i]),
            (labeled_dist, 0.30 * np.pi + self.view_id_offset[i]),
        ] for i in range(0, 4)
        }
        vlm_kwargs_N = vlm_kwargs.copy()
        vlm_kwargs_N["classify_method"] = None
        vlm_kwargs_N["look_ahead_action_num"] = self.max_action_one_seq
        vlm_kwargs_N["obs_key"] = ['rgb_front']
        self.planner_N_action = VLM(
            categories=self.action_space,
            model=planner_model,
            query_task="aeqa_planner_N_action",
            **vlm_kwargs_N,
        )


    def get_simulator(self, scene_id):
        try:
            self.sim.close()
            print(f"[Worker {self.parallel_ith}] Deleted previous simulator")
        except Exception as e:
            pass
        self.sim, self.cube2equirect_tfms = get_simulator(
            scene_id, self.device,      # fmt: skip
            sensor_hfov=self.obs_hfov,
            enable_surround_sensor=True,
            sensor_height=self.obs_height,
            sensor_width=self.obs_width,
            enable_depth=True,
            enable_semantic=False,
            sensor_pitch=np.deg2rad(-10),
        )
        self.sim.reset()
        planner = GreedyGeodesicFollower(
            pathfinder=self.sim.pathfinder,
            agent=self.sim.get_agent(0),
            goal_radius=0.25,
        )
        self.action_finder = ActionFinder(
            self.sim, planner,
        )
        return self.sim

    def inference(self):
        gts, preds = [], []
        traj_lens, traj_dists, demo_dists = [], [], []
        scores = []
        full_demo_lens = load_aeqa_demo_trajlens()

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
            demo_dist = full_demo_lens[datum["question_id"]]

            if is_reload:
                # * load datum metrics
                print(f"[Worker {self.parallel_ith}] Loading datum metrics from {metric_datum_path}")   # fmt: skip
                with open(metric_datum_path, "r") as f:
                    curr_metrics = json.load(f)
            else:
                #* preparation:
                sim = self.get_simulator(datum["scene_id"])
                print(f"[Worker {self.parallel_ith}] Loaded simulator for Q_ID: {datum_states_path}")   # fmt: skip
                self.detected_objs.clean_all()

                best_answer, position_traj = self.inference_aeqa(datum, sim)

                # * calc metrics
                extra_answers = datum.get("extra_answers")
                # remove anything after the last period
                end_idx = best_answer.rfind(".")
                if end_idx >= 0 and end_idx + 1 < len(best_answer):
                    best_answer = best_answer[: end_idx + 1]

                if self.evaluator in COMMERCIAL_MODELS:
                    openai_key = self.api_key
                    base_url = "https://api.openai.com/v1/"
                else:
                    openai_key = "xxxx"
                    base_url = self.base_url
                score = get_llm_match_score(
                    question=datum["question"],
                    answer=datum[self.gt_key],
                    prediction=best_answer,
                    extra_answers=extra_answers,
                    openai_key=openai_key,
                    openai_model=self.evaluator,
                    openai_base_url=base_url,
                )
                traj_len = len(position_traj)
                traj_dist = calc_traj_distance(position_traj, sim.pathfinder)

                curr_metrics = dict(
                    question=datum["question"],
                    gt=datum[self.gt_key],
                    pred=best_answer,
                    traj_len=traj_len,
                    demo_dist=demo_dist,
                    score=score,
                    traj_dist=traj_dist,
                    position_traj=position_traj,
                )
                # * save datum metrics
                with open(metric_datum_path, "w") as f:
                    json.dump(curr_metrics, f, ensure_ascii=False)

                # * visualize
                if traj_len > 0:
                    visualize_ar_baseline(
                        datum_dir=self.saver.get_datum_path_pref(datum),
                        key="visual_prompt",
                        label=datum[self.gt_key].replace(" ", "-"),
                        answer_file_name="planner_highlevel.json",
                        planner_file_name=f"planner_next-{self.max_action_one_seq}.json",
                    )

            demo_dists.append(curr_metrics["demo_dist"])
            traj_dists.append(curr_metrics["traj_dist"])

            # HACK: save self.temp as a json file to path: "cond_pano_paths.json"
            # temp_path = "cond_pano_paths.json"
            # with open(temp_path, "w") as f:
            #     json.dump(self.temp, f, ensure_ascii=False, indent=4)

            gts.append(curr_metrics["gt"])
            preds.append(curr_metrics["pred"])
            traj_lens.append(curr_metrics["traj_len"])
            scores.append(curr_metrics["score"])
            if not is_reload:
                curr_metrics = self.evaluate(
                    scores, traj_lens, traj_dists, demo_dists, datum, best_answer
                )

        log_metric(self.metrics_path, "end")

        return curr_metrics

    def evaluate(self, scores, traj_lens, traj_dists, demo_dists, datum, best_answer):
        total = len(scores)
        mean_score, mean_traj_dist, mean_efficiency = compute_aeqa_eval_metrics(
            scores, traj_dists, demo_dists
        )
        print(f"GT: {datum[self.gt_key]} | PRED: {best_answer}")
        print(f"==> Cumulative Score: {mean_score:.1f}")
        print(f"==> Cumulative Trajectory dist (Excluding Failures): {mean_traj_dist:.1f}")
        print(f"==> Mean Efficiency: {mean_efficiency:.1f}")

        time_elapsed = time.time() - self.start_time
        samples_remaining = len(self.dataset) - total
        eta = time_elapsed / total * samples_remaining
        time_elapsed_str = format_time(time_elapsed)
        eta_str = format_time(eta)
        print(f"==> Time Elapsed: {time_elapsed_str}")
        print(f"==> ETA: {eta_str}")

        metrics = dict(
            total=total,
            mean_score=mean_score,
            mean_efficiency=mean_efficiency,
            mean_traj_len=np.mean(traj_lens),
            mean_traj_dist=mean_traj_dist,
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

        # * cache other views:
        all_keys = (
            {"depth", "rgb"}
            | {f"rgb_surround_{view}" for view in self.view_orders}
            | {f"depth_surround_{view}" for view in self.view_orders}
        )
        for k in all_keys:
            if k not in state_imgs:
                state_imgs[k] = obs[k]

        assert set(state_imgs.keys()) == all_keys

        if save_imgs:
            saved_imgs = {k: state_imgs[k] for k in {"depth", "rgb"}}
            state_img_paths = self.save_on_disk(
                datum, ith_action, suffix, saved_imgs, verbose=False
            )
            # ^ create stitched_rgb
            keys = [f"rgb_surround_{view}" for view in self.view_orders]
            rgb_surround_imgs = [state_imgs[k] for k in keys]
            anno_frames = []
            for i in range(len(self.view_orders)):
                pil_img = to_pil_image(rgb_surround_imgs[i])
                anno_frame = annotate_frame(
                    pil_img, f"Current View: <{self.view_orders[i]}>",
                    text_color="red", bg_color="white", align="center"
                )
                anno_frames.append(np.array(anno_frame))

            stitched_rgb = np.stack(anno_frames, axis=0)
            image_path = self.saver.get_image_path(datum, ith_action, "stitched_rgb")

            save_image(
                torch.from_numpy(stitched_rgb).permute(0, 3, 1, 2) / 255.0,
                image_path,
                nrow=len(stitched_rgb),
            )
            state_img_paths["stitched_rgb"] = image_path

            # ^ make visual_prompt and visual_prompt_waypoint/rgb_surround_paths
            surround_obs = {k: v for k, v in state_imgs.items() if "surround" in k}
            vp_paths, rgb_surround_paths, detected_obj_ids = self.compose_visual_prompt(
                datum, ith_action, surround_obs, sim
            )
            state_img_paths["visual_prompt"] = vp_paths
            state_img_paths["rgb_surround"] = rgb_surround_paths
            state_img_paths["detected_obj_ids"] = detected_obj_ids
        else:
            state_img_paths = None

        return state_img_paths, state_imgs


    def inference_aeqa(self, datum, sim):
        st, skip_flag = self.init_states(datum, sim)
        complete_flag = True
        if skip_flag:
            return None, None

        position = datum["start_position"]
        rotation = datum["start_rotation"]
        consecutive_action_num = 0
        max_consecutive_num = 4

        st.update_position_traj(position)

        for ith_action in trange(
            1, self.max_actions + 1,  # fmt: skip
            desc="For ith_action",
            leave=False,
            position=1,
        ):
            # ^ AEQA baseline pipeline:
            action = st.pop_next_pending_action()
            if is_empty(action):
                if not complete_flag and consecutive_action_num < max_consecutive_num:
                    complete_flag, st, is_consecutive = self._lowlevel_planning(
                        datum, ith_action, st, init_history=False
                    )
                    if not is_consecutive:
                        consecutive_action_num = max_consecutive_num - 1

                while complete_flag or consecutive_action_num >= max_consecutive_num:
                    print("Finished high-level planning, complete_flag is now `True`")
                    if self.use_WM:
                        st = self.generate_imagine(
                            datum, sim.get_agent(0).state, st, ith_action
                        )

                    stop_flag, st = self._highlevel_planning(datum, ith_action, st)
                    if stop_flag: break

                    complete_flag, st, is_consecutive = self._lowlevel_planning(
                        datum, ith_action, st, init_history=True
                    )
                    if not is_consecutive:
                        consecutive_action_num = max_consecutive_num - 1
                    else:
                        consecutive_action_num = 0

                consecutive_action_num += 1
                if stop_flag: break
                action = st.pop_next_pending_action()

            # 4) Interact with the simulator
            position, rotation = self.perform_agent_move(
                sim, action["action"], action["initial_theta"]
            )
            st.record_past_action(action)
            st.update_position_traj(position)

            if st.get_pending_action_num() == 0 or ith_action == self.max_actions or ith_action % 15 == 0:
                state, state_imgs = self.interact(
                    sim, position, rotation, datum, ith_action
                )
                # Add the new state row and record the action
                st.add_new_state(state, state_imgs)

        best_answer = st.get_best_answer()  # might be None or the recognized name
        if best_answer is None:
            best_answer = "No Answer"
        return best_answer, st.position_traj


    def _highlevel_planning(
        self,
        datum,
        ith_action: int,
        st: State,
    ) -> Tuple[bool, Any, Any, State]:
        """
        1) Perform up to two fetches of high-level instructions.
        2) If we get a non-empty 'Answer' once, we try a second time to confirm.
        3) If the second fetch is also non-empty, we finalize the answer and stop.
        4) Otherwise, keep going.
        Returns:
            stop_flag (bool),
            st (updated State).
        """
        stop_flag = False
        best_answer = None
        max_attempts = 2    # when agent generates the answer for the first time, we confirm it for the second time
        attempts = 0
        # We'll store the final instructions/obs to record in state
        final_instructions, final_obs = None, None

        while attempts < max_attempts:
            use_WM = self.use_WM
            instructions, high_level_obs = self.fetch_highlevel_action_decision(
                datum, ith_action, st,
                self.highlevel_planner,
                use_WM=use_WM,
                prepare_imagine=False,
            )
            assert len(instructions) == 1, "Only one instruction plan is supported for now."

            # Keep track of the last instructions we fetched
            final_instructions = instructions
            final_obs = high_level_obs

            # Check if the plan has a non-empty answer
            if not is_empty(instructions[0]["Answer"]):
                if best_answer is not None:
                    # Already found one once. This is the second time -> finalize
                    best_answer = instructions[0]["Answer"]
                    print(f"Confirmed best_answer after second check: {best_answer}")
                    st.set_best_answer(best_answer)
                    stop_flag = True
                    break
                else:
                    # First time seeing a non-empty answer; store but do not stop yet
                    best_answer = instructions[0]["Answer"]
                    print(f"First time seeing non-empty answer: {best_answer}")
                    attempts += 1
            else:
                break

        # Record whichever instructions we ended with
        st.add_to_recent_state(final_instructions, "high_level_plan")
        st.add_to_recent_state(final_obs, "high_level_obs")
        st.clean_up_history(key=self.imagine_action_key)
        st.clean_up_history(key=self.imagine_obs_key)

        return stop_flag, st


    def parser_highlevel_plan_imagine(self, agent_state, highlevel_plan_imagine, st):
        """
        parser the dict plan into action seqs
        highlevel_plan_imagine: List[{
            "Reason": str,
            "Action Plan": str,
            "Chosen View": str,
            "Chosen Landmark": int,
            "Answer": str,
        }
        """
        # view_idx = self.view_orders.index(chosen_view)
        init_turn_degrees = []
        agent_position = agent_state.position
        agent_rotation = agent_state.rotation
        for plan in highlevel_plan_imagine:
            landmark = plan["Chosen Landmark"]
            if is_empty(landmark):
                chosen_view = plan["Chosen View"]
                chosen_view_depth = st.fetch_current_state_obs(
                    key=f"depth_surround_{chosen_view}"
                )
                devation = compute_theta_deviation_from_depth(
                    chosen_view_depth, self.obs_hfov
                )
                if devation is None:
                    continue
                init_theta = self.view_id_offset[
                    self.view_orders.index(chosen_view)
                ]
                init_theta = -(init_theta + devation)
            else:
                # calculate the rotation angle btw the current heading and the chosen landmark
                landmark_pos = self.detected_objs.get_object_positions([landmark])[0]
                landmark_pos = pos_normal_to_habitat(landmark_pos)

                if filter_by_distance(landmark_pos, agent_position):
                    continue
                agent_orientation_target = get_azimuth_from_ponts(agent_position, landmark_pos)
                # agent_orientation_target = get_azimuth_from_quat(landmark_pos)
                agent_orientation_curr = get_azimuth_from_quat(agent_rotation)
                init_theta = agent_orientation_target - agent_orientation_curr  #should be a negative value

            init_degree = init_theta / np.pi * 180
            # init_degree = init_degree + 180 if init_degree < 0 else init_degree
            while init_degree > 180:
                init_degree -= 360
            while init_degree < -180:
                init_degree += 360
            if init_degree not in init_turn_degrees:    #To avoid duplicate
                init_turn_degrees.append(init_degree)

        prior_action_ids = [[] for _ in range(len(init_turn_degrees))]
        # change into dict format:
        init_turn_degrees = {
            i: init_turn_degrees[i] for i in range(len(init_turn_degrees))
        }
        prior_action_ids = {
            i: prior_action_ids[i] for i in range(len(init_turn_degrees))
        }
        return init_turn_degrees, prior_action_ids, prior_action_ids


    def _highlevel_planning_imagine(
        self,
        datum,
        ith_action: int,
        st: State,
    ) -> Tuple[bool, Any, Any, State]:
        """
        1) Fetch & update recognition answer in 'st'.
        2) If recognition passes threshold, signal we should stop.
        3) Fetch & perform next move.
        4) Update 'st' with new row/action.
        """
        # * 0: generate high level plan
        instructions, high_level_obs = self.fetch_highlevel_action_decision(
            datum, ith_action,
            st,
            self.highlevel_planner,
            use_WM=True,
            prepare_imagine=True,
            query_num=3,
        )
        st.add_to_recent_state(instructions, self.imagine_action_key)
        return instructions


    def _lowlevel_planning(self, datum, ith_action, st, init_history):
        """Low-level planning for the next action."""
        # * method 1: generate look_ahead action seq
        recent_highlevel_plan = st.get_from_history(key="high_level_plan")[-1]
        chosen_view = recent_highlevel_plan[0]["Chosen View"]
        recent_highlevel_plan = self._filter_keys(
            [recent_highlevel_plan],
            keep_keys=["Action Plan", "Chosen Landmark"],
        )[0]
        # only use the first plan:
        recent_highlevel_plan = recent_highlevel_plan[0]

        if not is_empty(recent_highlevel_plan["Chosen Landmark"]):
            # get the chose landmark position:
            obj_id = recent_highlevel_plan["Chosen Landmark"]
            landmark_pos = self.detected_objs.get_object_positions([obj_id])[0]
            landmark_rad = self.detected_objs.get_object_radius([obj_id])[0]
            landmark_pos = pos_normal_to_habitat(landmark_pos)

            success = self.action_finder.set_new_nav_pt(
                object_pos=landmark_pos, object_radius=landmark_rad,
            )
            if success:
                actions = self.action_finder.get_next_action_seq()
                if len(actions) == 1 and actions[0] is None:
                    complete_flag = True
                else:
                    refactor_actions = []
                    for i, a_c in enumerate(actions):
                        if a_c is not None:
                            a = self.sim_actions_space_inv[a_c]
                            refactor_actions.append({"initial_theta": 0, "action": a})
                    complete_flag = False
                    if len(refactor_actions) > 50:
                        print(f"Too many pending actions one time, current len: {len(refactor_actions)}, reduce it to 50")
                        refactor_actions = refactor_actions[:50]
                    st.add_pending_actions(refactor_actions)

                return complete_flag, st, False

        # * method 2: reinitialize the state
        init_theta = 0
        chosen_view_idx = 0
        if init_history:
            st.clean_up_history(key="low_level_obs")
            st.clean_up_history(key="low_level_plan")
            init_theta = self.view_id_offset[self.view_orders.index(
                chosen_view,
            )]

            chosen_view_idx = self.view_orders.index(chosen_view)

        actions, record_low_level_obs = self.fetch_action_decision_vlm(
            datum, ith_action,
            st,
            self.planner_N_action,
            enable_history=True,
            view_idx=chosen_view_idx,
            tobe_filled={"high_level_plan": recent_highlevel_plan},
        )
        actions = actions[0]        #currently process only one action_seq

        # ^ new added for processing heading:
        if not actions["is_stop"]:
            refactor_actions = []
            for i, a_c in enumerate(actions["convert_answer"]):
                # Use the initial heading adjustment only for the first action.
                theta = init_theta if i == 0 else 0
                refactor_actions.append({"initial_theta": theta, "action": a_c})

            # Optionally update the state with the original answer as a string.
            st.add_to_recent_state(str(actions["origin_answer"]), "low_level_plan")
            st.add_to_recent_state(record_low_level_obs, "low_level_obs")
            st.add_pending_actions(refactor_actions)
            complete_flag = False
        else:
            complete_flag = True

        return complete_flag, st, True


    def fetch_action_decision_vlm(
        self, datum, ith_action, states, planner, enable_history, view_idx, tobe_filled,
    ):
        """Fetch the action decision from the planner."""
        action_path = self.saver.get_planner_output_path(
            datum, ith_action, action_num=planner.look_ahead_action_num
        )
        chat_log_path = self.saver.get_chat_log_output_path(action_path)
        if (
            self.use_WM and planner.query_task == "aeqa_highlevel_planner"
            and self.imagine_action_key in states.get_all_recorded_keys()
        ):
            imagine_traj = states.get_from_history(key=self.imagine_action_key)
        else:
            imagine_traj = []

        # Gather all value-lists first
        curr_low_level_obs = states.get_from_history("rgb_surround")[-1]
        curr_low_level_obs = curr_low_level_obs[view_idx]

        curr_history = states.get_from_history("low_level_obs")
        obs_traj = curr_history + [curr_low_level_obs]
        action_traj = states.get_from_history(key="low_level_plan")
        record_low_level_obs = [curr_low_level_obs]

        messages = planner.assemble_messages(     #decision #2
            obs_traj,
            action_traj,
            enable_history=enable_history,
            tobe_filled=tobe_filled,
            imagine_traj=imagine_traj,
            enable_system_prompt=True,
        )
        lowlevel_action = planner.query_VLM(
            messages=messages,
            tobe_filled=tobe_filled,
        )
        chat_log = format_chat_dialog(messages, lowlevel_action)
        os.makedirs(osp.dirname(action_path), exist_ok=True)
        with open(action_path, "w") as f:
            json.dump(lowlevel_action, f, indent=2, ensure_ascii=False)
        with open(chat_log_path, "w") as f:
            json.dump(chat_log, f, indent=2, ensure_ascii=False)

        return lowlevel_action, record_low_level_obs


    def fetch_highlevel_action_decision(
        self, datum, ith_action, states, planner, use_WM, query_num=1, prepare_imagine=False,
    ):
        """Fetch the action decision from the planner."""
        if prepare_imagine:      #when not is_imagine, assume calling from _highlevel_planning_imagine
            action_path = self.saver.get_highlevel_planner_imagine_output_path(
                datum, ith_action,
            )
        else:
            action_path = self.saver.get_highlevel_planner_output_path(
                datum, ith_action,
            )
        chat_log_path = self.saver.get_chat_log_output_path(action_path)

        if (
            use_WM and not prepare_imagine and planner.query_task == "aeqa_highlevel_planner"
            and self.imagine_action_key in states.get_all_recorded_keys()
        ):
            imagine_plan = states.get_from_history(key=self.imagine_action_key)
            imagine_plan = self._filter_keys(imagine_plan, keep_keys=["Reason", "Action Plan", "Chosen View", "Chosen Landmark"])
            imagine_traj = states.get_from_history(key=self.imagine_obs_key)
        else:
            imagine_traj, imagine_plan = [], []

        # 1. Gather all value-lists first
        curr_obs = states.get_from_history("stitched_rgb")[-1]
        curr_visual_prompt = states.get_from_history("visual_prompt")[-1]

        curr_high_level_obs = [curr_obs] + [curr_visual_prompt]
        curr_history = states.get_from_history(key="high_level_obs")
        obs_traj = curr_history + [curr_high_level_obs]

        record_high_level_obs = [curr_visual_prompt]
        # states.add_to_recent_state([curr_visual_prompt], key="high_level_obs")

        # 2.1. fetch the detected object names
        det_obj_ids = states.get_from_history(key=self.detected_obj_key)[-1]
        det_obj_names = {
            view_id: self.detected_objs.fetch_pred_obj_names(obj_ids)
            for view_id, obj_ids in det_obj_ids.items()
        }
        # 2.2. replace the vals in det_obj_ids with det_obj_names
        det_obj_dict = {self.view_orders[view_id]: {} for view_id in range(len(self.view_orders))}
        for view_id, obj_ids in det_obj_ids.items():
            assert len(obj_ids) == len(det_obj_names[view_id])
            for nav_idx, obj_name in zip(obj_ids, det_obj_names[view_id]):
                det_obj_dict[self.view_orders[view_id]][nav_idx] = obj_name

        # 3. fetch the previous high level plan history
        action_traj = states.get_from_history(key="high_level_plan")
        action_traj_ = self._filter_keys(action_traj)
        # * 0: generate high level plan
        messages = planner.assemble_messages(     #decision #1
            obs_traj,
            action_traj_,
            enable_history=self.enable_hist_planner,
            imagine_traj=imagine_traj,
            imagine_action_traj=imagine_plan,
            tobe_filled={"question": datum["question"], "detected_objs": det_obj_dict},
        )
        instruction = planner.query_VLM(
            messages=messages,
            query_num=query_num,
            tobe_filled={"question": datum["question"], "detected_objs": det_obj_dict},
        )
        chat_log = format_chat_dialog(messages, instruction)
        os.makedirs(osp.dirname(action_path), exist_ok=True)
        with open(action_path, "w") as f:
            json.dump(instruction, f, indent=2, ensure_ascii=False)
        with open(chat_log_path, "w") as f:
            json.dump(chat_log, f, indent=2, ensure_ascii=False)

        return instruction, record_high_level_obs

    def _filter_keys(self, action_traj,
        keep_keys=["Reason", "Action Plan", "Chosen View", "Chosen Landmark", "Answer"],
    ):
        if is_empty(action_traj):
            return action_traj
        action_traj_ = [
            [{k: v for k, v in plan.items() if k in keep_keys} for plan in plans]
            for plans in action_traj
        ]
        return action_traj_


    def compose_visual_prompt(self, datum, ith_action, surround_obs, sim):
        """
        get the visual prompt from grounding_sam2 for the high level planner
        """
        # Initialize the grounding_sam2_sock
        if not hasattr(self, "gd_sam2_sock"):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(
                (self.gd_sam2_host.split(":")[0], int(self.gd_sam2_host.split(":")[1]))
            )
            self.gd_sam2_sock = sock
            print(f"[Client] Connected grounding_sam2_sock manager at {self.gd_sam2_host}")

        action_folder_path = self.saver.get_action_path_pref(datum, ith_action)
        view_keys = self.view_orders

        # * get the image path from the state_traj
        new_folders = []
        for i, view in enumerate(view_keys):
            key = [k for k in self.surround_obs_key if view in k]
            assert len(key) == 1
            key = key[0]
            # image_path = surround_obs[key]
            new_folder = osp.join(action_folder_path, f"{key}")
            os.makedirs(new_folder, exist_ok=True)
            new_image_path = osp.join(new_folder, f"0.jpg")
            # save the surround_obs[key] as .jpg img:
            image = to_pil_image(surround_obs[key])
            image.save(new_image_path)
            new_folders.append(new_folder)

        new_folders = [os.path.abspath(folder) for folder in new_folders]
        input_dict = {
            "save_dirs": new_folders,
            # "text_prompts": [self.dataset.OBJECT_SET[::2]] * len(new_folders),
        }
        check_inputdict(input_dict, server_type="gd_sam2")
        write_framed(self.gd_sam2_sock, input_dict)
        output_dict = read_framed(self.gd_sam2_sock)
        obj_mask_info = output_dict["obj_mask_infos"]
        # obj_mask_infos is a list (len 4): [{"masks": [], "text_labels": [], "boxes": []}, {...}]

        # * match the gt_obj_id to the detected masks
        mask_mapping = []
        depth_keys = [k.replace("rgb", "depth") for k in self.surround_obs_key]
        agent_state = sim.get_agent(0).state
        sensor_states = agent_state.sensor_states
        depth_sensor_states = [sensor_states[k] for k in depth_keys]
        for i, key in enumerate(view_keys):
            # read the image
            depth_img = surround_obs[depth_keys[i]]
            masks = np.array(obj_mask_info[i]["masks"])
            text_labels = obj_mask_info[i]["text_labels"]

            if is_empty(text_labels):
                continue

            # Do a new filtering according to the depth and mask
            camera_points_all, text_label_all, masks_all = [], [], []
            for cls, mask in zip(text_labels, masks):
                camera_points = get_pointcloud_from_depth_mask(
                    depth_img, mask, self.surround_camera_intrinsic,
                )
                if is_empty(camera_points):
                    continue
                text_label_all.append(cls)
                camera_points_all.append(camera_points)
                masks_all.append(mask)

            curr_obj_ids = self.detected_objs.add_new_frame(
                camera_points=camera_points_all,
                obj_names=text_label_all,
                cam_position=depth_sensor_states[i].position,
                cam_rotation=depth_sensor_states[i].rotation,
            )
            assert len(curr_obj_ids) == len(masks_all), f"len(curr_obj_ids): {len(curr_obj_ids)}, len(masks_all): {len(masks_all)}"
            mask_mapping.extend([
                (obj_id, i, mask) for obj_id, mask in zip(curr_obj_ids, masks_all)
            ])

        # * remove the obj_id that has been visited
        visited_ids = self.detected_objs.get_visited_object_ids()
        candidate_masks = {i:[] for i in range(len(view_keys))}
        candidate_obj_ids = {i:[] for i in range(len(view_keys))}       #{view_id: {nav_idx: obj_idx}}
        for i, (obj_id, view_id, mask) in enumerate(mask_mapping):
            if obj_id not in visited_ids:
                candidate_masks[view_id].append(mask)
                candidate_obj_ids[view_id].append(obj_id)

        # * label the unique navigation_obj_idx on the rgb images
        pil_images = []
        for i, (view_id, mask_list) in enumerate(candidate_masks.items()):
            key = [k for k in self.surround_obs_key if self.view_orders[view_id] in k][0]
            image = to_pil_image(surround_obs[key])

            # label the mask on the image
            image = annotate_frame_masks(
                image, mask_list,           # fmt: skip
                list(candidate_obj_ids[view_id]),
                title=f"Current View: <{self.view_orders[view_id]}>",
                # contour_alpha=20,
            )
            pil_images.append(image)

        image_tensors = [to_tensor(pil_image) for pil_image in pil_images]
        rgb_surround_paths = []
        for i in range(len(image_tensors)):     #* save self.surround_obs_key[i]
            image_path = self.saver.get_image_path(
                datum, ith_action, self.surround_obs_key[i],
            )
            save_image(image_tensors[i], image_path, nrow=len(image_tensors))
            rgb_surround_paths.append(image_path)

        vp_path = self.saver.get_visual_prompt_path(datum, ith_action)
        save_image(
            torch.stack(image_tensors),
            vp_path,
            nrow=len(image_tensors),
        )

        print(f"Visual prompt saved to {vp_path}")
        return vp_path, rgb_surround_paths, candidate_obj_ids


    def perform_agent_move(self, sim, action, theta=0):
        # Perform turning actions if the given action is in the mapping.
        agent = sim.get_agent(0)
        position, rotation = (
            agent.state.position.tolist(), agent.state.rotation.components.tolist()
        )
        def dist_fn(a, b):
            """compute euclidean distance using np"""
            return np.linalg.norm(a - b)

        rotate_agent(agent, -theta)  #NOTE: the -theta here
        for move, suffix in self.sim_actions_space.get(action):
            position, rotation = self.agent_move(sim, move)

            # dist_fn = sim.nav.pathfinder.geodesic_distance
            self.detected_objs.update_object_state(np.array(position), dist_fn)

        return position, rotation


    def generate_imagine(self, datum, agent_state, st, ith_action):
        # * 0: generate look_ahead action seq
        sensor_state = agent_state.sensor_states["rgb_front"]
        highlevel_plan_imagine = self._highlevel_planning_imagine(datum, ith_action, st)
        # highlevel_plan_imagine = [{'Reason': "The provided query asks about what is above the wooden table in the living room. The stitched panoramic image with annotations shows a wooden coffee table with object index '1' located in the 'front' view. However, there is no object directly above the coffee table that can be identified from the images. There are pictures on the wall nearby, but they are not directly above the table. Therefore, the current observation does not provide a definite answer.", 'Action Plan': "Move towards the coffee table (object index '1') in the 'current view and stop once the area above the table is clearly visible.", 'Chosen View': 'front', 'Chosen Landmark': 1, 'Answer': None, 'landmark_ori_idxs': [(0, 1), (2, 1)]}, {'Reason': 'The query is about what is above the wooden table in the living room. From the annotated image, I can see the wooden table is labeled as "desk" (object index 3) in the front and right views. The desk location is now clear, and I should move towards it to observe what is directly above it.', 'Action Plan': 'Move towards the desk (object index "3") in the current view and stop once the area directly above the desk is visible.', 'Chosen View': 'front', 'Chosen Landmark': 3, 'Answer': None, 'landmark_ori_idxs': [(0, 3), (2, 7)]}]

        # * 1.1 Generate predicted frames for the candidate actions
        priors = self.parser_highlevel_plan_imagine(
            agent_state, highlevel_plan_imagine, st
        )
        init_turn_degrees, prior_action_ids, origin_action_ids = priors
        if is_empty(init_turn_degrees):
            print("WARNING: No space for generate imagine frames, skipping this imagine.")
            return st

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

        # * 1.3 align frames / get front view fron panos
        fn_postprocess = self.get_postprocess_fn(self.task, coord_type)
        rgbs_wo_mask = fn_postprocess(
            output_dict0, per_hfov=self.obs_hfov,
            selected_idx=13,
            img_size=(self.pred_obs_height, self.pred_obs_width)
        )

        pred_save_paths = self.select_and_save_preds(
            save_dirs, rgbs_wo_mask,
        )
        st.add_to_recent_state(pred_save_paths, key=self.imagine_obs_key)

        return st



def run_solver_process(parallel_ith, args, api_key):
    setup_logger(
        osp.join(args.log_output_dir, f"{args.exp_id}", f"subProcess_{parallel_ith}.log")   # fmt: skip
    )
    print(f"Logger set up for worker {parallel_ith}")
    print(f"All args:\n {args}")

    solver = AEQASolver(
        max_actions=250,
        api_key=api_key,
        use_WM=args.use_WM,
        debug_len=None,
        obs_key=["stitched_rgb"],
        view_order=VIEW_ORDER,
        obs_hfov=105,
        answerer=args.answerer_model,
        planner=args.planner_model,
        evaluator=args.eval_model,
        parallel_ith=parallel_ith,
        parallel_total=args.worker_num,
        WM_host=args.WM_host,
        sam2_host=args.sam2_host,
        gd_sam2_host=args.grounding_sam2_host,
        enable_hist_planner=args.enable_hist_planner,
        enable_hist_answerer=args.enable_hist_answerer,
        subset_size=args.subset_size,
        args=args,
    )

    result = solver.inference()
    print(f"Worker {parallel_ith} result: {result}")
    return result


if __name__ == "__main__":
    parser = build_common_arg_parser()
    # Task-specific options
    parser.add_argument("--eval_model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--grounding_sam2_host", type=str, default="127.0.0.1:6002")
    parser.add_argument("--enable_hist_planner", type=bool, default=False)
    parser.add_argument("--enable_hist_answerer", type=bool, default=False)
    parser.add_argument("--subset_size", type=int, default=184)

    args, unused_cli_tokens = parser.parse_known_args()

    launch_multiprocessing(args, run_solver_process)

