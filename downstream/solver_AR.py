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
from torch import Tensor
from tqdm import tqdm, trange

from downstream.downstream_datasets import ARDataset
from downstream.prompts import AR_ACTION_SPACE_NO_STOP, VLM, UNIT_DEGREE, UNIT_DISTANCE
from downstream.simulator import (
    draw_target_bbox,
    get_observations,
    default_settings,
)
from downstream.visualize import (
    visualize_ar_baseline,
    annotate_frame,
)
from downstream.vlm import COMMERCIAL_MODELS, LOCAL_MODELS, WORLD_MODEL_TYPES
from downstream.utils.saver import get_igenex_save_dirs, Saver, format_chat_dialog
from downstream.utils.state_traj import State
from downstream.utils.igenex_util import (
    IGENEX_ACTION_IDS,
    compose_turn_actions,
    bbox_to_mask,
    generate_aligned_bbox_frames,
)
from downstream.utils.util import format_time, log_metric, visualize_semantic_img, mask_semantic_by_target
from utils.logger import setup_logger
from utils.svd_utils import rotate_coord_by_degrees, rotate_by_degrees
from habitat_data.equi2cube import convert_equi2per
from torchvision.transforms.functional import to_pil_image, to_tensor
from downstream.solver_base import Solver, build_common_arg_parser, launch_multiprocessing


class ARSolver(Solver):
    def __init__(
        self,
        max_actions: int,
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
        use_igenex_planner: bool,
        args=None,
    ):
        # * data
        self.args = args
        self.dataset = ARDataset()

        self.parallel_ith = parallel_ith
        all_devices = torch.cuda.device_count()
        self.device = parallel_ith % all_devices
        self.parallel_total = parallel_total
        self.WM_host = WM_host
        self.sam2_host = sam2_host
        self.use_igenex_planner = use_igenex_planner
        self.args.vlm_input_format = self.set_vlm_input_format(planner, answerer)

        num_ofvlm = len(self.args.vllm_host)
        self.base_url = f"http://{self.args.vllm_host[self.parallel_ith % num_ofvlm]}/v1"

        # Create a Saver instance to handle path calls
        self.saver = Saver(parallel_ith, parallel_total, self.args.exp_id)
        self.task = "AR"

        # * logging
        self.metrics_path = self.saver.get_metric_path()

        # * simulator
        self.sim = None
        # ^ for AR
        self.use_WM = use_WM
        self.TTS_ratio = args.TTS_ratio
        assert len(obs_key) > 0 and set(obs_key).issubset({"rgb_bbox", "rgb_bbox_front"})
        if self.use_WM:
            # assert obs_key == "rgb_bbox"
            self.imagine_obs_key = "pred_bbox_front"
            self.imagine_action_key = "pred_bbox_action_seq"
        self.obs_key = obs_key
        if "rgb_bbox_front" in obs_key:
            self.obs_hfov = default_settings["sensor_hfov"]
        else:   #if use pano as obs_key, the genex generated frame hfov is 90
            self.obs_hfov = 90.0
        self.obs_height, self.obs_width = 512, 512
        self.pred_obs_height, self.pred_obs_width = 384, 384

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
        self.query_num = 2
        self.look_ahead_action_num = 4      #range (1, igenex_n_frame-1)
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
        self.planner = VLM(
            categories=self.action_space,
            model=planner,
            query_task="ar_planner",
            top_logprobs=5,
            **vlm_kwargs,
        )

        self.answerer = VLM(
            categories=self.dataset.OBJECT_SET,
            model=answerer,
            query_task="ar_answerer",
            top_logprobs=5,
            **vlm_kwargs,
        )
        self.max_actions = max_actions
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
            query_task="ar_planner_N_action",
            **vlm_kwargs_N,
        )


    def inference(self):
        gts, preds, traj_lens = [], [], []
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
            print(f"Datum: {self.saver.get_category_path(datum, datum['target_categrory'])}")
            if is_reload:
                # * load datum metrics
                print(f"[Worker {self.parallel_ith}] Loading datum metrics from {metric_datum_path}")
                with open(metric_datum_path, "r") as f:
                    curr_metrics = json.load(f)
            else:
                sim = self.get_simulator(datum["scene_id"])
                print(f"[Worker {self.parallel_ith}] Loaded simulator for scene: {datum['scene_id']}")  # fmt: skip

                best_answer, traj_len = self.inference_ar(datum, sim)

                # * visualize
                if traj_len > 0:
                    visualize_ar_baseline(
                        datum_dir=self.saver.get_datum_path_pref(datum),
                        key=self.obs_key,
                        vis_order="planner_first",
                        answer_file_name="answerer.json",
                        planner_file_name="planner_next-1.json",
                    )

                curr_metrics = dict(
                    gt=datum["target_categrory"],
                    pred=best_answer,
                    traj_len=traj_len,
                )
                # * save datum metrics
                with open(metric_datum_path, "w") as f:
                    json.dump(curr_metrics, f, indent=2, ensure_ascii=False)

            gts.append(curr_metrics["gt"])
            preds.append(curr_metrics["pred"])
            traj_lens.append(curr_metrics["traj_len"])

            if not is_reload:
                metrics = self.evaluate(gts, preds, traj_lens, datum, best_answer)

        log_metric(self.metrics_path, "end")

        return metrics

    def evaluate(self, gts, preds, traj_lens, datum, best_answer):
        total = len(gts)
        acc = np.mean(np.array(gts) == np.array(preds))
        mean_traj_len = np.mean([t for t in traj_lens if t is not None])

        print(f"GT: {datum['target_categrory']} | PRED: {best_answer}")
        print(f"==> Cumulative Accuracy: {acc:.1%}")
        print(f"==> Cumulative Trajectory Length (Excluding Failures): {mean_traj_len}")

        time_elapsed = time.time() - self.start_time
        samples_remaining = len(self.dataset) - total
        eta = time_elapsed / total * samples_remaining
        time_elapsed_str = format_time(time_elapsed)
        eta_str = format_time(eta)
        print(f"==> Time Elapsed: {time_elapsed_str}")
        print(f"==> ETA: {eta_str}")

        metrics = dict(
            total=total,
            acc=acc,
            mean_traj_len=mean_traj_len,
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

        # * only return the single item semantic of the front sensor
        obs = mask_semantic_by_target(datum["target_id"], obs)

        # * draw target bbox on rgb
        state_imgs["rgb_bbox"], exist_pano, bbox_coord = draw_target_bbox(
            state_imgs["semantic"],
            torch.einsum("chw->hwc", state_imgs["rgb"]),
            target_id=datum["target_id"],
            bbox_color=(255, 0, 0, 255),  # red
            scale_factor=1.15,
        )

        # * process sem_obs only for visualization
        state_imgs["semantic_target"] = visualize_semantic_img(state_imgs["semantic"])

        obs = get_observations(
            sim, position, rotation
        )  # ! hardcode, only for visualization
        sem_obs_raw = self.process_pano_obs(obs)[2].squeeze(0)  # 2 for semantic
        state_imgs["semantic"] = visualize_semantic_img(sem_obs_raw)

        # * visualize front sensor
        state_imgs["rgb_front"] = obs["rgb_sensor"]
        state_imgs["rgb_bbox_front"], exist_front, bbox_coord1 = draw_target_bbox(
            obs["semantic_sensor"],
            obs["rgb_sensor"],
            target_id=datum["target_id"],
            bbox_color=(255, 0, 0, 255),  # red
            scale_factor=1.15,
        )
        state_imgs["semantic_front"] = visualize_semantic_img(obs["semantic_sensor"])

        all_keys = {
            "semantic",
            "semantic_front",
            "semantic_target",
            "rgb", "depth",
            "rgb_front",
            "rgb_bbox",
            "rgb_bbox_front",
        }
        assert set(state_imgs.keys()) == all_keys

        if save_imgs:
            state_img_paths = self.save_on_disk(
                datum, ith_action, suffix, state_imgs, verbose=False
            )
            self.save_target_category(datum, ith_action, verbose=False)
        else:
            state_img_paths = None

        state_imgs["rgb_bbox_coord"] = bbox_coord   # add bbox_coord to state_imgs
        state_imgs["frontview_bbox_coord"] = bbox_coord1
        if not (exist_pano or exist_front):
            # NOTE when not exist_pano or exist_front, changed into only returning None for state_img_paths
            return None, state_imgs  # fmt: skip
        else:
            return state_img_paths, state_imgs


    def fetch_recognize_answer(
        self, datum, st, ith_action, answerer, use_saved_file=False
    ):
        # NOTE: Assume the answerer is performed first in loop
        answer_path = self.saver.get_answerer_output_path(datum, ith_action)
        if osp.exists(answer_path) and use_saved_file:
            with open(answer_path, "r") as f:
                answer_probs = json.load(f)
        else:
            if self.use_WM:
                # imagine_traj = state_traj[self.imagine_obs_key]
                imagine_traj = st.get_from_history(key=self.imagine_obs_key)
            else:
                imagine_traj = []

            # Gather all value-lists first
            list_of_lists = [st.get_from_history(key) for key in self.obs_key]
            states = [list(items) for items in zip(*list_of_lists)]

            messages = answerer.assemble_messages(
                states,
                st.get_action_traj(),
                enable_history=False,
                imagine_traj=imagine_traj,
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
        stop_flag = False

        # 1) Fetch recognition answer
        answer, answer_value = self.fetch_recognize_answer(
            datum,
            st,
            ith_action,
            self.answerer,
        )
        st.add_answer(answer, answer_value)  # auto-update best answer

        # 2) Possibly update best answer
        if answer_value > st.get_best_answer_val():
            st.set_best_answer_val(answer_value)
            st.set_best_answer(answer)
            print(f"Update best_answer: P({answer})={answer_value:.1%}, GT={datum['target_categrory']}")
            if answer_value >= self.recog_thres:
                stop_flag = True

        # 3) Fetch the next action & perform the move (only if not stopping yet)
        if not stop_flag:
            ith_action = ith_action + 1  # NOTE: ith_action for move starts from 1 not 0
            # if self.use_heur:
            # action, _ = self.heur_sample_next_action(
            #     action_seq=st.get_action_traj(), seed=ith_action
            # )
            # # save the sampled action seqs
            # action_path = self.saver.get_planner_output_path(
            #     datum, ith_action, action_num=1
            # )
            # os.makedirs(osp.dirname(action_path), exist_ok=True)
            # with open(action_path, "w") as f:
            #     json.dump({action: 1.0}, f, indent=2, ensure_ascii=False)
            # print(f"Sampled heuristic action: {action}")
            # else:
            action = self.fetch_action_decision_vlm(
                datum, st,
                ith_action,
                self.planner
            )
            position, rotation = self.perform_agent_move(
                sim, action,
            )

            # 4) Interact with the simulator
            state, state_imgs = self.interact(
                sim, position, rotation, datum, ith_action
            )
            if state is None:
                stop_flag = True
            else:
                # Add the new state row and record the action
                st.add_new_state(state, state_imgs)
                st.record_past_action(action)

        return stop_flag, position, rotation, st


    def fetch_action_decision_vlm(
        self, datum, st, ith_action, planner, use_saved_file=False
    ):
        """Fetch the action decision from the planner."""
        action_path = self.saver.get_planner_output_path(
            datum, ith_action, action_num=planner.look_ahead_action_num
        )
        chat_log_path = self.saver.get_chat_log_output_path(action_path)
        if osp.exists(action_path) and use_saved_file:
            with open(action_path, "r") as f:
                action_probs = json.load(f)
        else:
            if (
                self.use_WM and planner.query_task == "ar_planner"
            ):
                # imagine_traj = state_traj[self.imagine_action_key].tolist()
                imagine_traj = st.get_from_history(key=self.imagine_action_key)
            else:
                imagine_traj = []

            # Gather all value-lists first
            list_of_lists = [st.get_from_history(key) for key in self.obs_key]
            states = [list(items) for items in zip(*list_of_lists)]

            messages = planner.assemble_messages(
                states,
                st.get_action_traj(),
                enable_history=False,
                imagine_traj=imagine_traj,
            )
            action_probs = planner.query_VLM(
                messages=messages,
                query_num=self.query_num,  # only work for planner_N
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


    def inference_ar(self, datum, sim):
        st, skip_flag = self.init_states(datum, sim)
        if skip_flag:
            return None, None

        position = datum["start_position"]
        rotation = datum["start_rotation"]

        for ith_action in trange(
            0, self.max_actions + 1, # fmt: skip
            desc="For ith_action",
            leave=False,
            position=1,
        ):
            # * add the genex forward step:
            if self.use_WM and np.random.random() < self.TTS_ratio:
                st = self.forward_with_WM(datum, sim, st, position, rotation, ith_action)

            # (B) Now reuse the same snippet for recognition and moving:
            stop_flag, position, rotation, st = self._take_step_and_recognize(
                datum, sim, ith_action, st, position, rotation
            )
            best_answer = st.get_best_answer()

            st = self.clean_cache(st)

            if stop_flag:
                break

        if best_answer is None:
            ith_action = None
        return best_answer, ith_action

    def forward_with_WM(self, datum, sim, st, position, rotation, ith_action):
        # * 0: generate look_ahead action seq
        if self.use_heur:
            action_seqs_u, _ = self.heur_sample_next_action_seqs(
                action_seq=st.get_action_traj(),
                query_num=self.query_num,
            )
            # save the sampled action seqs
            action_path = self.saver.get_planner_output_path(
                datum, ith_action, action_num=self.planner_N_action.look_ahead_action_num
            )
            os.makedirs(osp.dirname(action_path), exist_ok=True)
            with open(action_path, "w") as f:
                json.dump(action_seqs_u, f, indent=2, ensure_ascii=False)
            print(f"Sampled heuristic action seq: {action_seqs_u}")
        else:
            action_seqs = self.fetch_action_decision_vlm(
                datum, st,
                ith_action,
                self.planner_N_action
            )
            action_seqs_u, _ = self.extract_unique_action_seq(action_seqs)

        # * 1.1 Generate predicted frames for the candidate actions
        if len(action_seqs_u) != 0:
            priors = self.get_action_info_from_prior(action_seqs_u)
            init_turn_degrees, prior_action_ids, origin_action_ids = priors
        else:
            init_turn_degrees, prior_action_ids = self.get_action_seqs_noprior()

        output_dict0 = self.imagine_by_model_type(
            datum, sim, st,
            position, rotation, ith_action,      # fmt: skip
            init_turn_degrees, prior_action_ids, origin_action_ids, # fmt: skip
            init_rotate_type="by_degrees",
        )   #List[dict], List[str]

        save_dirs, init_frame_bbox_coords, pred_frames, coord_type = (
            output_dict0["save_dirs"],
            output_dict0["bbox_coords"],
            output_dict0["pred_frames"],
            output_dict0["coord_type"],
        )

        output_dict = self.generate_bbox_for_preds(
            init_frame_bbox_coords, save_dirs, pred_frames,
        )

        # * 1.3 align frames / get front view fron panos
        fn_postprocess = self.get_postprocess_fn(self.task, coord_type)
        out = fn_postprocess(output_dict, per_hfov=self.obs_hfov,
                              img_size=(self.pred_obs_height, self.pred_obs_width))
        rgbs_aligned_w_bbox, rgbs_w_bbox, retain_idxs = out

        if self.use_igenex_planner:
            init_rgbs, init_turn_actions = self.get_init_frames_from_init_degrees(
                st.fetch_current_state_obs("rgb_bbox"), init_turn_degrees,
            )

            # merge the actions:
            actions_all, rgbs_w_bbox_all = self.get_merged_preds(
                prior_action_ids, rgbs_w_bbox, init_rgbs, init_turn_actions
            )

            pred_save_paths = self.compose_action_results_from_preds(
                save_dirs, rgbs_w_bbox_all, actions_all,
            )
            st.add_to_recent_state(pred_save_paths, key=self.imagine_action_key)

        pred_save_paths = self.select_and_save_preds(
            save_dirs, rgbs_aligned_w_bbox, interval=2, start_idx=3
        )
        st.add_to_recent_state(pred_save_paths, key=self.imagine_obs_key)  #NOTE: comment here for only plan

        return st

    def imagine_by_model_type(
            self, datum, sim, st,
            position, rotation, ith_action,
            init_turn_degrees, prior_action_ids, origin_action_ids,
            init_rotate_type,   # "by_degrees" | "by_shift"
        ):
        output_dict = super().imagine_by_model_type(
            datum, st,
            ith_action,
            init_turn_degrees, prior_action_ids, origin_action_ids,
            init_rotate_type=init_rotate_type,
        )

        # * 1.2 Generate predicted bbox for the candidate frames by SAM2:
        output_dict = self.prepare_gt_bbox_coord(
            position, rotation,         # fmt: skip
            datum, sim,                 # fmt: skip
            rotate_degrees=init_turn_degrees,
            output_dict=output_dict,
        )
        # save_dirs = [f"downstream/states/AR_08.17_ar_filter_10step95/wc2JMjhGNzB/E043/A001/igenex/PredA-0"]

        return output_dict


    def select_and_save_preds(self, save_dirs, rgbs_aligned_w_bbox, interval, start_idx):
        pred_save_paths = []
        for i in range(len(rgbs_aligned_w_bbox)):
            if rgbs_aligned_w_bbox[i] is None:
                continue
            frames = rgbs_aligned_w_bbox[i]
            # select the frames according to the interval:
            if frames.shape[0] == 0:
                continue
            else:
                frames_selected = frames[start_idx::interval]
                if frames_selected.shape[0] == 0:
                    frames_selected = frames[-1:]
            # nrow = find_best_nrow(bs=frames_selected.shape[0])
            # for j in range(frames_selected.shape[0]):
            pred_save_path = osp.join(save_dirs[i], f"{self.imagine_obs_key}.png")
            pred_save_path = self.save_vlm_input_media(frames_selected / 255.0, pred_save_path)
            pred_save_paths.append(pred_save_path)
        return pred_save_paths

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

            pred_save_path = osp.join(save_dirs[i], f"{self.imagine_action_key}.png")
            action_results_frames = torch.stack(action_results_frames, dim=0)
            pred_save_path = self.save_vlm_input_media(action_results_frames, pred_save_path)
            pred_save_paths.append(pred_save_path)

        return pred_save_paths

    def get_init_frames_from_init_degrees(self, rgb, init_turn_degrees):
        rgb: UInt8[NDArray, "H W C"] = rgb
        rgb = torch.einsum("hwc->chw", torch.from_numpy(rgb))
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
                w_pers=self.pred_obs_width, h_pers=self.pred_obs_height,
                fov_x=self.obs_hfov
            )
            init_rgbs[id] = [frame for frame in init_rgbs_front]
        return init_rgbs, init_turn_actions


    def prepare_gt_bbox_coord(
        self, position, rotation,
        datum, sim,
        output_dict: Dict[str, Any],
        rotate_degrees: Optional[Dict[Any, float]] = None,
    ) -> Dict[str, Any]:
        """
        Updates and returns `output_dict` with:
        - "bbox_coords": List[Dict[str, int]]
        - "save_dirs":   List[str] (possibly unchanged)
        - "pred_frames": List[torch.Tensor] or torch.Tensor (depending on coord_type)

        Notes:
        * For non_pano: writes a seed frame "-1.jpg" next to each save_dir and
            prepends the aligned seed frame to each clip.
        * For pano:     rotates the bbox for each action and passes frames through.
        """
        # fetch original vals from output_dict:
        save_dirs = output_dict["save_dirs"]
        coord_type = output_dict["coord_type"]  # "non_pano" | "pano"
        pred_frames: UInt8[NDArray, "b 14 C H W"] = np.array(output_dict["pred_frames"])

        out_height, out_width = pred_frames.shape[-2], pred_frames.shape[-1]
        bbox_coords, save_dirs_, pred_frames_ = [], [], []
        if rotate_degrees is None:  # walk through all possible actions
            rotate_degrees = self.init_turn_actions
        # * get gt_coord of rgb bbox
        _, state_imgs = self.interact(sim, position, rotation, datum, save_imgs=False)
        bbox_coord: Dict[str, int] = state_imgs["rgb_bbox_coord"]
        img_width = state_imgs["rgb"].shape[-1]
        if bbox_coord is None:
            output_dict.update({"bbox_coords": [], "save_dirs": [], "pred_frames": []})
            return output_dict

        # NOTE: if save_dirs is not None, then we would hack into the save_dirs
        # and insert a "-1.jpg" as the initial frame for SAM2
        if coord_type == "non_pano":
            assert save_dirs is not None, "save_dirs must be provided for non_pano coord_type"
            mask_rgb: bool[NDArray, "HW"] = bbox_to_mask(
                bbox_coord, img_shape=state_imgs["rgb"].shape[-2:]
            )
            mask_frames = mask_rgb.reshape(
                1, 1, mask_rgb.shape[0], mask_rgb.shape[1]
            )   #BxCxHxW
            rgb_frames: UInt8[Tensor, "BCHW"] = state_imgs["rgb"].unsqueeze(0).numpy()
            rgbs_aligned_w_bbox, bbox_coords_ = generate_aligned_bbox_frames(
                rgb_frames, mask_frames,
                per_hfov=self.obs_hfov,
                img_size=(out_height, out_width),   #HW, NOTE: align it with the config resultions of WM server
                draw_bbox=False,
            )
            for i, dir in enumerate(save_dirs):
                if rgbs_aligned_w_bbox is None:
                    continue

                save_dirs_.append(dir)
                bbox_coords.append(bbox_coords_[0])
                pred_frames_.append(
                    np.concatenate([rgbs_aligned_w_bbox.numpy(), pred_frames[i]], axis=0)
                )
            pred_frames_ = np.array(pred_frames_)  # Bx15xCHW
        elif coord_type == "pano":
            for action_id, degrees in rotate_degrees.items():
                bbox_coord_ftm = rotate_coord_by_degrees(
                    bbox_coord, degrees, img_width=img_width
                )
                # Add a batch dimension to the image: [1, 4, H, W] (assuming original shape [4, H, W])
                bbox_coords.append(bbox_coord_ftm)
            save_dirs_ = save_dirs
            pred_frames_ = pred_frames

        # assign new vals to output_dict
        output_dict["bbox_coords"] = bbox_coords
        output_dict["save_dirs"] = save_dirs_
        output_dict["pred_frames"] = pred_frames_
        return output_dict


def run_solver_process(parallel_ith, args, api_key):
    setup_logger(
        osp.join(
            args.log_output_dir, f"{args.exp_id}", f"subProcess_{parallel_ith}.log"
        )
    )
    print(f"Logger set up for worker {parallel_ith}")
    print(f"All args:\n {args}")

    solver = ARSolver(
        max_actions=10,
        api_key=api_key,
        use_heur=args.use_heur,
        use_WM=args.use_WM,
        recog_thres=0.95,
        obs_key=["rgb_bbox_front", "rgb_bbox"],  # ["rgb_bbox_front" "rgb_bbox"]
        answerer=args.answerer_model,
        planner=args.planner_model,
        parallel_ith=parallel_ith,
        parallel_total=args.worker_num,
        WM_host=args.WM_host,
        sam2_host=args.sam2_host,
        use_igenex_planner=True,
        args=args,
    )

    result = solver.inference()
    print(f"Worker {parallel_ith} result: {result}")
    return result



if __name__ == "__main__":
    parser = build_common_arg_parser()
    # Task-specific options
    parser.add_argument("--use_heur", action="store_true", help="Use heuristic policy or not")
    parser.add_argument("--TTS_ratio", type=float, default=1.0, help="Probability of running forward_with_WM (0.1 means 10% probability)")

    args, unused_cli_tokens = parser.parse_known_args()

    launch_multiprocessing(args, run_solver_process)
