import argparse
import os
import os.path as osp
import socket
import sys
import time
import copy
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
from jaxtyping import Float, Int32, UInt8
from numpy.typing import NDArray
from torch import Tensor
from torchvision.utils import save_image
import cv2
from data_filtering.filter_util import save_img, save_img_stitch, save_video_from_tensor

from downstream.utils.saver import prepare_saved_imgs
import habitat_sim
from downstream.prompts import (
    construct_action_space_text,
)
from downstream.simulator import (
    get_observations,
    get_simulator,
)
from downstream.utils.igenex_util import (
    IGENEX_ACTION_IDS,
    post_process_output_ar, post_process_output_ar_non_pano,
    post_process_output_aeqa, post_process_output_aeqa_non_pano,
    post_process_output_ignav, post_process_output_ignav_non_pano,
    prepare_init_panos,
    compute_2d_bbox_from_8_corners, is_wrapped_by_width,
    annotate_perspective_views,
)
from downstream.utils.saver import Saver, get_igenex_save_dirs
from downstream.utils.state_traj import State
from downstream.utils.util import (
    rotate_and_forward_agent,
)
from downstream.utils.worker_manager import read_framed, write_framed, check_inputdict
from downstream.vlm import WORLD_MODEL_TYPES
from downstream.utils.workers_cfg import OUT_WIDTH, OUT_HEIGHT
from habitat_data.equi2cube import convert_equi2per
from collect_bbox.coordinate_transformation import world_to_spherical
from collect_bbox.draw_bbox import spherical_to_equirectangular


class Solver:

# ================= setting Helpers =================
    def get_simulator(self, scene_id, enable_depth=True, enable_semantic=True):
        if hasattr(self, "current_sim_id") and self.current_sim_id == scene_id:
            pass
        else:
            try:
                self.sim.close()
                print(f"[Worker {self.parallel_ith}] Deleted previous simulator")
            except Exception as e:
                pass
            self.sim, self.cube2equirect_tfms = get_simulator(
                scene_id, self.device,      # fmt: skip
                sensor_hfov=self.obs_hfov,
                sensor_height=self.obs_height,
                sensor_width=self.obs_width,
                enable_depth=enable_depth,
                enable_semantic=enable_semantic,
            )
            self.sim.reset()
            self.current_sim_id = scene_id

        return self.sim

    def set_vlm_input_format(self, planner, answerer):
        if "InternVL" in planner or "InternVL" in answerer:
            # vlm_input_format = "video"
            vlm_input_format = "image"
        else:
            vlm_input_format = "image"
        return vlm_input_format


    def set_world_model_type(self):
        if self.args.world_model_type is not None:
            return

        # Auto-detect world model type from experiment ID
        detected_models = []
        for model_type, models in WORLD_MODEL_TYPES.items():
            for model in models:
                if f"_{model}" in self.args.exp_id:
                    detected_models.append(model)
                    self.args.world_model_type = model_type
                    self.args.world_model_name = model
                    print(f"World model type set to {model_type} based on experiment ID.")

        # Validation
        if len(detected_models) > 1:
            raise ValueError(f"Ambiguous world model types found: {detected_models}")
        if self.args.world_model_type is None and self.use_WM:
            raise ValueError("Please specify the world model type in the experiment ID or args.")

# ================= task piplining Helpers =================
    def save_target_category(self, datum, ith_action, verbose):
        if ith_action == 0:
            target_category = datum["target_categrory"]
            if verbose:
                print(f"==> Ground truth: {target_category}")
            category_path = self.saver.get_category_path(datum, target_category)
            with open(category_path, "w") as f:
                f.write(target_category)


    def get_observations(self, sim, position, rotation):
        obs = get_observations(sim, position, rotation)

        state_imgs = self.process_pano_obs(obs)
        state_imgs = {
            sensor_type: state_imgs[i].squeeze(0)
            for i, sensor_type in enumerate(("rgb", "depth", "semantic"))
            if state_imgs[i] is not None
        }

        return obs, state_imgs

    def save_on_disk(self, datum, ith_action, suffix, state_imgs, verbose):
        state_img_paths = dict()
        for sensor_type, state_img in state_imgs.items():
            image_path = self.saver.get_image_path(
                datum, ith_action, sensor_type, suffix
            )
            save_img(state_img, image_path, verbose=verbose)
            state_img_paths[sensor_type] = image_path

        return state_img_paths

    def agent_move(self, sim, action):
        agent = sim.get_agent(0)
        obs = sim.step(action)
        position, rotation = agent.state.position, agent.state.rotation.components

        return position.tolist(), rotation.tolist()


    def init_states(self, datum, sim):
        ith_action = 0
        skip_flag = False
        position = datum["start_position"]
        rotation = datum["start_rotation"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state, state_imgs = self.interact(
                sim, position, rotation, datum, ith_action
            )

        # Check for any warnings or invalid state.
        if len(w) > 0 or state is None:
            skip_flag = True
            if len(w) > 0:
                for warn in w:
                    print(f"WARNING: Skipping datum due to warnings: {warn.message}")
                sys.exit(1)
            if state is None:
                print("WARNING: Skipping datum due to invalid bbox at starting")
            print(f"Skipping datum: {datum}")
            st = None
        else:
            # Create a new StateTraj with the columns from the 'state' keys
            st = State(columns=list(state.keys()))
            st.add_new_state(state, state_imgs)

        return st, skip_flag

    def get_action_seqs_noprior(self):
        forward_id = IGENEX_ACTION_IDS["forward"]

        action_ids = {}
        for i, (k, v) in enumerate(self.init_turn_actions.items()):
            action_ids[i] = [forward_id] * (self.igenex_n_frame - 1)

        return self.init_turn_actions, action_ids

    def select_and_save_preds(self, save_dirs, rgbs_w_bbox, interval=10**10):
        pred_save_paths_all = []
        for i in range(len(save_dirs)):
            frames = rgbs_w_bbox[i]
            frames = torch.einsum("VNCHW->NVCHW", frames)

            # select the frames according to the interval:
            idxs = torch.arange(0, frames.shape[0], interval)
            frames_selected = frames[idxs]

            pred_save_paths = []
            for j in range(frames_selected.shape[0]):   #for each perspective views:
                frames_selected_anno = annotate_perspective_views(
                    frames_selected[j],
                )   #shape (4, C, H, W)

                pred_save_path = osp.join(save_dirs[i], f"{self.imagine_obs_key}_{idxs[j]}.png")
                save_image(
                    frames_selected_anno,
                    pred_save_path,
                    nrow=frames_selected_anno.shape[0],
                )
                # print(f"Predicted frames saved to: {pred_save_path}")
                pred_save_paths.append(pred_save_path)
            pred_save_paths_all.append(pred_save_paths)
        return pred_save_paths_all


    def generate_bbox_for_preds(self, bbox_coords, save_dirs, pred_frames):
        # unique variables for genex:
        if not hasattr(self, "sam2_sock"):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(
                (self.sam2_host.split(":")[0], int(self.sam2_host.split(":")[1]))
            )
            self.sam2_sock = sock
            print(f"[Client] Connected sam2_sock manager at {self.sam2_host}")

        assert len(bbox_coords) == len(save_dirs) == len(pred_frames), \
            f"len(bbox_coords): {len(bbox_coords)}, len(save_dirs): {len(save_dirs)}, len(pred_frames): {len(pred_frames)} \n Values: {bbox_coords}, {save_dirs}, {pred_frames}"
        save_dirs = [os.path.abspath(f) for f in save_dirs]
        if len(save_dirs) != 0:
            input_dict = {
                "save_dirs": save_dirs,
                "bbox_coords": bbox_coords,
                "pred_frames": pred_frames, #UInt8[NDarray, "b T C H W"]
            }
            check_inputdict(input_dict, server_type="sam2")
            write_framed(self.sam2_sock, input_dict)
            output_dict = read_framed(self.sam2_sock)
        else:
            output_dict = {"save_dirs": []}
        return output_dict

    def perform_agent_move(self, sim, action):
        # Perform turning actions if the given action is in the mapping.
        for move, suffix in self.sim_actions_space.get(action):
            position, rotation = self.agent_move(sim, move)

        return position, rotation

    def prepare_gt_bbox_coord(self, img_save_dirs, det_obj_ids, RTs):
        """
        get the 2d bbox coord of the detected objects for all the frames coresponding to the RTs
        """
        assert len(img_save_dirs) == len(RTs), f"len(img_save_dirs): {len(img_save_dirs)}, len(RTs): {len(RTs)}"
        bbox_corners_all = self.detected_objs.get_object_3d_bbox_corners(
            det_obj_ids, flip_yz=False,
        )

        first_frames, output_paths = [], []
        for dir in img_save_dirs:
            # read the first image in each dir (sorted by name)
            for f in sorted(os.listdir(dir)):
                if f.endswith(".jpg"):
                    img_path = osp.join(dir, f)
                    break
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_width, img_height = img.shape[1], img.shape[0]
            first_frames.append(img)
            output_paths.append(img_path.replace(".jpg", "_bbox.jpg"))

        # Convert the bounding box corners to spherical coordinates and draw them on the images.
        bbox_pixel_coords = defaultdict(list)
        bbox_det_obj_ids = defaultdict(list)
        remove_idxs = []
        for i, RT in enumerate(RTs):
            for j, bbox_corners in enumerate(bbox_corners_all):
                _, corners_spherical = world_to_spherical(
                    bbox_corners, RT
                )
                pixel_coords = spherical_to_equirectangular(
                    corners_spherical, img_width, img_height
                )
                if is_wrapped_by_width(pixel_coords, img_width):
                    # This bounding box crosses the back/ wrap boundary => skip
                    continue
                else:
                    # safe to project
                    bbox = compute_2d_bbox_from_8_corners(pixel_coords)
                    bbox_pixel_coords[i].append(bbox)
                    bbox_det_obj_ids[i].append(det_obj_ids[j])

                # draw_bbox_from_spherical_coords(
                #     output_paths[i],
                #     first_frames[i], pixel_coords,  # fmt: skip
                # )
            # Instead of bbox_pixel_coords[i], use .get() or membership check:
            bboxes_for_i = bbox_pixel_coords.get(i, [])
            if bboxes_for_i:
                stacked = np.stack(bboxes_for_i, axis=0).tolist()
                bbox_pixel_coords[i]: Int32[NDArray, "N 4"] = stacked
            else:
                # if no bbox in the pano, remove the frames from the list
                remove_idxs.append(i)
        img_save_dirs = [
            img_save_dirs[i] for i in range(len(img_save_dirs)) if i not in remove_idxs
        ]
        return list(bbox_pixel_coords.values()), list(bbox_det_obj_ids.values()), img_save_dirs

    def process_pano_obs(self, obs):
        """
        Processes the observation by converting cube maps to equirectangular format.
        Args:
            obs (dict): A dictionary containing cube map observations from habitat-sim.
                        Keys like 'rgb_back', 'depth_front' map to numpy arrays [0~255]
                        representing the sensor images.
        Returns:
            tuple: A tuple containing:
                - rgb (torch.Tensor): Equirectangular RGB image tensor of shape [1, 4, H, W].
                - depth (torch.Tensor or None): Equirectangular depth image tensor of shape [1, H, W, 1],
                                                if available; otherwise, None.
                - semantic (torch.Tensor or None): Equirectangular semantic image tensor of shape [1, H, W, 1],
                                                if available; otherwise, None.
        """
        # Convert numpy arrays in obs to torch tensors with a batch dimension.
        obs_tensor = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                # Convert uint32 arrays to int32.
                if v.dtype == np.uint32:
                    v = v.astype(np.int32)
                # tensor = torch.from_numpy(v).unsqueeze(0)  # Shape: [1, H, W, X]
                tensor: Int32[Tensor, "1 H W ?"] = torch.from_numpy(v).unsqueeze(0)
                # For depth or semantic data, add an extra channel dimension.
                if "semantic" in k or "depth" in k:
                    tensor: Float[Tensor, "1 H W 1"] = tensor.unsqueeze(-1)
                obs_tensor[k] = tensor

        # Process RGB: convert cube map to equirectangular and adjust dimensions.
        rgb = self.cube2equirect_tfms["rgb"](obs_tensor)["rgb_back"]
        rgb: UInt8[Tensor, "1 4 H W"] = torch.einsum("bhwc->bchw", rgb)
        # Process depth if available.
        depth = None
        if "depth" in self.cube2equirect_tfms:
            depth: Float[Tensor, "1 H W 1"] = self.cube2equirect_tfms["depth"](
                obs_tensor
            )["depth_back"]

        # Process semantic if available.
        semantic = None
        if "semantic" in self.cube2equirect_tfms:
            semantic: Int32[Tensor, "1 H W 1"] = self.cube2equirect_tfms["semantic"](
                obs_tensor
            )["semantic_back"]

        return rgb, depth, semantic


    def get_postprocess_fn(self, task_name, coord_type):
        task_map = {
            "AR": (post_process_output_ar, post_process_output_ar_non_pano),
            "IGNav": (post_process_output_ignav, post_process_output_ignav_non_pano),
            "AEQA": (post_process_output_aeqa, post_process_output_aeqa_non_pano),
        }
        if task_name not in task_map:
            raise ValueError(f"Unknown task: {task_name}")
        pano_fn, non_pano_fn = task_map[task_name]

        if coord_type == "pano":
            return pano_fn
        elif coord_type == "non_pano":
            return non_pano_fn

    def get_merged_preds(self, prior_action_ids, rgbs_w_bbox, init_rgbs, init_turn_actions):
        """
        Common method to merge predicted frames with initial frames and actions.
        Args:
            prior_action_ids: List of action ID sequences for each candidate
            rgbs_w_bbox: List of predicted RGB frame sequences
            init_rgbs: Dict mapping candidate ID to initial RGB frames
            init_turn_actions: Dict mapping candidate ID to initial turn actions
        Returns:
            actions: Dict mapping candidate ID to full action sequences
            rgbs_w_bbox_all: Dict mapping candidate ID to full RGB frame sequences
        """
        actions, rgbs_w_bbox_all = {}, {}
        if len(rgbs_w_bbox) != len(init_rgbs):
            print(f"WARNING: len(rgbs_w_bbox) != len(init_rgbs), len(rgbs_w_bbox): "
                  f"{len(rgbs_w_bbox)}, len(init_rgbs): {len(init_rgbs)}")
            return actions, rgbs_w_bbox_all

        for i, (k, v) in enumerate(init_rgbs.items()):
            if rgbs_w_bbox[i] is None:
                continue

            prior_actions = [self.action_space_map[a_id] for a_id in prior_action_ids[i]]
            actions[k] = init_turn_actions[k] + prior_actions

            rgbs_w_bbox_list = list(rgbs_w_bbox[i])
            if self.args.world_model_type == "GTsim":
                rgbs_w_bbox_all[k] = rgbs_w_bbox_list
            else:
                # NOTE we start from 1 to skip the same frame
                rgbs_w_bbox_list = rgbs_w_bbox_list[1:]
                # and keep only as many frames as there are prior actions.
                rgbs_w_bbox_all[k] = init_rgbs[k] + rgbs_w_bbox_list

            min_len = min(len(actions[k]), len(rgbs_w_bbox_all[k]))
            actions[k] = actions[k][:min_len]
            rgbs_w_bbox_all[k] = rgbs_w_bbox_all[k][:min_len]
            assert (len(actions[k]) <= (self.look_ahead_action_num + 1)
            ), f"len(rgbs_w_bbox_all): {len(rgbs_w_bbox_all[k])}, len(actions): {len(actions[k])}"

        return actions, rgbs_w_bbox_all


# ================== New Helpers ==================
    def save_vlm_input_media(self, action_results_frames, pred_save_path):
        if self.args.vlm_input_format == "video":
            pred_save_path = pred_save_path.replace(".png", ".mp4")
            save_video_from_tensor(
                action_results_frames,
                pred_save_path,
                fps=2,
            )
        elif self.args.vlm_input_format == "image":
            save_image(
                action_results_frames,
                pred_save_path,
                nrow=action_results_frames.shape[0],
            )

        return pred_save_path

    @staticmethod
    def extract_unique_action_seq(action_seqs):
        seen = set()
        action_seqs_u, action_seqs_u_ori = [], []
        for seq in action_seqs:
            tup = tuple(seq["convert_answer"])
            if tup not in seen:
                seen.add(tup)
                action_seqs_u.append(seq["convert_answer"])
                action_seqs_u_ori.append(seq["origin_answer"])
        return action_seqs_u, action_seqs_u_ori

    def get_action_info_from_prior(self, prior_actions):
        init_turn_degrees: dict[int, int] = {}
        remain_action_ids: dict[int, list[int]] = {}
        all_action_ids: dict[int, list[int]] = {}

        for seq_idx, seq in enumerate(prior_actions):
            cumulative_turn = 0
            leftover_ids: list[int] = []

            for act_idx, act in enumerate(seq[: self.look_ahead_action_num]):
                turn_angle = self.init_turn_actions[act]

                # Condition 1: “forward” action
                # Condition 2: next turn reverses the current heading trend
                if turn_angle == 0 or (cumulative_turn * turn_angle < 0):
                    leftover_ids = [
                        self.action_space_map_inv[a] for a in seq[act_idx:]
                    ]
                    break
                cumulative_turn += turn_angle

            init_turn_degrees[seq_idx] = cumulative_turn
            remain_action_ids[seq_idx] = leftover_ids
            all_action_ids[seq_idx] = [self.action_space_map_inv[a] for a in seq]

        return init_turn_degrees, remain_action_ids, all_action_ids


# ================== WM_forward Helpers ==================
    def _prepare_common_data(self, rgb, init_turn_degrees, prior_action_ids, save_dir, rotate_type):
        """
        Common preparation logic shared across all look_ahead_explore methods.

        Returns:
            tuple: (vis_path, action_ids_list, batch_actions, image_tensors, save_dirs)
        """
        vis_path = osp.join(save_dir, "igenex")
        os.makedirs(vis_path, exist_ok=True)
        forward_id = IGENEX_ACTION_IDS["forward"]
        start_id = IGENEX_ACTION_IDS["stop"]
        action_ids_list = list(range(len(init_turn_degrees)))

        # * complement the remain_action_ids to the designed len with action_id=1(forward)
        action_tensors = []
        for i in range(len(prior_action_ids)):
            prior_actions = prior_action_ids[i]
            remain_len = (self.igenex_n_frame - 1) - len(prior_actions)
            assert remain_len >= 0
            acton_seq = [start_id] + prior_actions + [forward_id]*remain_len
            action_tensors.append(torch.tensor(acton_seq, dtype=torch.long))

        image_tensors = prepare_init_panos(rgb, init_turn_degrees, rotate_type)
        batch_actions: Int32[Tensor, "b 14"] = torch.stack(action_tensors, dim=0)
        save_dirs = get_igenex_save_dirs(vis_path, action_ids_list)

        return vis_path, action_ids_list, batch_actions, image_tensors, save_dirs

    def look_ahead_explore(self, rgb, init_turn_degrees, prior_action_ids, save_dir, rotate_type):
        rgb: UInt8[Tensor, "C H W"] = rgb
        vis_path, action_ids_list, batch_actions, image_tensors, save_dirs = self._prepare_common_data(
            rgb, init_turn_degrees, prior_action_ids, save_dir, rotate_type
        )

        batch_images: UInt8[Tensor, "b C H W"] = torch.cat(image_tensors, dim=0)

        # * 2. Generate frames for the all actions with batch
        self.connect_to_WM_server()

        # * 3. Send the batch to the server and get response
        output_dict = self.send_batch_to_server(batch_actions, save_dirs, batch_images)
        return output_dict

    def look_ahead_explore_non_pano(self, rgb, init_turn_degrees, prior_action_ids, save_dir, rotate_type):
        rgb: Float[Tensor, "C H W"] = rgb
        vis_path, action_ids_list, batch_actions, image_tensors, save_dirs = self._prepare_common_data(
            rgb, init_turn_degrees, prior_action_ids, save_dir, rotate_type
        )

        batch_images: UInt8[Tensor, "b C H W"] = torch.cat(image_tensors, dim=0)
        perspective_views = convert_equi2per(
            batch_images, w_pers=OUT_WIDTH, h_pers=OUT_HEIGHT,  #NOTE: align the input_size with the out resultions of WM server
            fov_x=self.obs_hfov,
        )
        # rgb_paths = prepare_saved_imgs_nwm(perspective_views, save_dirs)  #

        # * 2. Generate frames for the all actions with batch
        self.connect_to_WM_server()

        # * 3. Send the batch to the server and get response)
        output_dict = self.send_batch_to_server(batch_actions, save_dirs, perspective_views)
        return output_dict

    def look_ahead_explore_camera(self, rgb, depth, init_turn_degrees, prior_action_ids, save_dir, rotate_type):
        rgb: Float[Tensor, "C H W"] = rgb
        depth: Float[Tensor, "1 H W"] = torch.einsum("hwc->chw", depth)

        vis_path, action_ids_list, batch_actions, image_tensors, save_dirs = self._prepare_common_data(
            rgb, init_turn_degrees, prior_action_ids, save_dir, rotate_type
        )

        depth_tensors = prepare_init_panos(depth, init_turn_degrees, rotate_type)

        batch_rgbs: Float[Tensor, "b C H W"] = torch.cat(image_tensors, dim=0)
        batch_depths: Float[Tensor, "b C H W"] = torch.cat(depth_tensors, dim=0)
        rgb_paths, depth_paths = prepare_saved_imgs(batch_rgbs, batch_depths, save_dirs)

        init_degrees_list = list(init_turn_degrees.values())
        final_paths = []
        for i in range(len(save_dirs)):
            success = self.save_gt_rgb(
                init_degrees_list[i], save_dirs[i], forward_dist=2.4,
            )
            if success:
                final_paths.append((rgb_paths[i], depth_paths[i]))
        self.temp.extend(final_paths)

        # * 2. Generate frames for the all actions with batch
        self.connect_to_WM_server()

        # * 3. Send the batch to the server and get response (only support local mode)
        output_dict = self.send_batch_to_server(batch_actions, save_dirs)
        return output_dict

    def look_ahead_explore_gt(self, rgb, init_turn_degrees, prior_action_ids, origin_action_ids, save_dir, rotate_type):
        """
        Ground-truth variant of look_ahead_explore: renders future frames by querying the simulator.
        Input/outputs mirror look_ahead_explore; only RGB is saved.
        """
        # Common prep: action tensorization + save dir layout (kept identical to WM path)
        vis_path, action_ids_list, batch_actions, _image_tensors, save_dirs = self._prepare_common_data(
            rgb, init_turn_degrees, origin_action_ids, save_dir, rotate_type
        )
        init_degrees_list = list(init_turn_degrees.values())
        pred_rgbs_all: list[list[str]] = []

        # Map action IDs to degrees
        id_to_action = self.action_space_map

        def _get_rgb_from_obs(pred_dir: str, step_idx: int):
            """Grab pano obs at current agent state and save the RGB frame."""
            agent_state = agent.get_state()
            obs, pano_obs = self.get_observations(
                self.sim, agent_state.position, agent_state.rotation.components
            )
            if self.task == "AEQA":
                pred_obs = pano_obs["rgb"]    #pano (C, H, W)
                pred_obs = torch.einsum("chw->hwc", pred_obs).numpy() # (C, H, W) -> (H, W, C)
            else:
                pred_obs = obs["rgb_front"]   #perspective (H, W, C)
            return pred_obs

        agent = self.sim.get_agent(0)
        origin_agent_state = agent.get_state()

        # Roll out each candidate sequence
        batch_actions = batch_actions.tolist()
        for i, (action_seq, pred_dir) in enumerate(zip(batch_actions, save_dirs)):
            rgb_init = _get_rgb_from_obs(pred_dir, step_idx=0)
            pred_rgbs: list[UInt8[NDArray, "H W C"]] = [rgb_init]

            # Set starting pose = origin pose rotated by the candidate's initial yaw
            if len(origin_action_ids[i]) == 0:
                print(f"WARNING: empty future action_seq in {pred_dir}, auto filled with 'forward'")

                theta0 = np.deg2rad(init_degrees_list[i])
                start_pos, start_rot = rotate_and_forward_agent(
                    origin_agent_state, theta0, magnitude=0.0
                )
                agent.set_state(habitat_sim.AgentState(position=start_pos, rotation=start_rot))
                rgb_init_2 = _get_rgb_from_obs(pred_dir, step_idx=0)
                action_seq = action_seq[2:]
                pred_rgbs = [rgb_init, rgb_init_2]

            for t, action_id in enumerate(action_seq):
                if IGENEX_ACTION_IDS["stop"] == action_id:
                    continue    # skip "stop" actions in GT rollout

                # 1) step simulator according to action
                action_name = id_to_action[action_id]
                position, rotation = Solver.perform_agent_move(
                    self, self.sim, action_name,
                )

                # 2) record current frame
                pred_rgb = _get_rgb_from_obs(pred_dir, step_idx=t)
                pred_rgbs.append(pred_rgb)

            # Restore for the next candidate and collect paths
            agent.set_state(origin_agent_state)
            assert len(pred_rgbs) == self.igenex_n_frame
            pred_rgbs_all.append(pred_rgbs)

        # Stack and transpose to match expected format: [b, T, C, H, W]
        pred_frames = np.stack([np.stack(frames, axis=0) for frames in pred_rgbs_all], axis=0)
        pred_frames = np.transpose(pred_frames, (0, 1, 4, 2, 3))  # [b, T, H, W, C]->[b, T, C, H, W]
        return {
            "save_dirs": save_dirs,
            "pred_frames": pred_frames,
        }

    def send_batch_to_server(self, batch_actions, save_dirs, batch_images=None):
        """
        return:
            output_dict = {
                "save_dirs": save_dirs,	    # same with input
                "pred_frames": pred_frames	# pred_frames is UInt8[NDArray, "b 14 C H W"], 14 is the predicted video length
            }
        """
        save_dirs = [os.path.abspath(f) for f in save_dirs]
        input_dict = {
            "b_action": batch_actions.numpy(),
            "save_dirs": save_dirs,
            "request_model_name": self.args.world_model_name
        }
        if batch_images is not None:
            input_dict["b_image"] = batch_images.numpy()
        if self.WM_remote_mode:
            input_dict["return_objects"] = [True]*len(save_dirs)

        check_inputdict(input_dict)
        write_framed(self.igenex_sock, input_dict)
        output_dict = read_framed(self.igenex_sock)
        # save_dirs = output_dict["save_dirs"]
        # if self.WM_remote_mode:
        #     pred_frames: Uint8[NDarray, "b 14 C H W"] = output_dict["pred_frames"]
        #     save_dirs = output_dict["save_dirs"]
        #     # save the frames on local when remote_mode to make format consistant:
        #     save_video_frames(pred_frames, save_dirs)
        return output_dict

    def connect_to_WM_server(self):
        if not hasattr(self, "igenex_sock"):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            port = int(self.WM_host.split(":")[1])
            if port not in [-1]:        # NOTE: the default ports for local machine
                self.WM_remote_mode = True
                print(f"[Client] Enable WM_remote_mode with port: {port}")
            else:
                self.WM_remote_mode = False
            sock.connect(
                (self.WM_host.split(":")[0], port)
            )
            self.igenex_sock = sock
            print(f"[Client] Connected WM_socket manager at {self.WM_host}")


    def imagine_by_model_type(
            self, datum, st,
            ith_action,
            init_turn_degrees, prior_action_ids, origin_action_ids,
            init_rotate_type,   # "by_degrees" | "by_shift"
        ):
        kwargs = dict(
            init_turn_degrees=init_turn_degrees,
            prior_action_ids=prior_action_ids,
            save_dir=self.saver.get_action_path_pref(datum, ith_action),
            rotate_type=init_rotate_type,
        )
        if self.args.world_model_type in ["action", "FTtext"]:
            output_dict = self.look_ahead_explore(
                st.fetch_current_state_obs("rgb"),
                **kwargs,
            ) #List[str]
            coord_type = "pano"

        elif self.args.world_model_type in ["GTsim"]:
            output_dict = self.look_ahead_explore_gt(
                st.fetch_current_state_obs("rgb"),
                origin_action_ids=origin_action_ids,    #only used in GTsim
                **kwargs,
            ) #List[str]
            if self.task == 'AEQA':
                coord_type = "pano"
            else:
                coord_type = "non_pano"

        elif self.args.world_model_type == "camera":
            output_dict = self.look_ahead_explore_camera(
                st.fetch_current_state_obs("rgb"),
                st.fetch_current_state_obs("depth"),
                **kwargs,
            )
            coord_type = "pano"

        elif self.args.world_model_type == "text":
            output_dict = self.look_ahead_explore_non_pano(
                st.fetch_current_state_obs("rgb"),
                **kwargs,
            )
            coord_type = "non_pano"

        print(f"[Worker {self.parallel_ith}] Finished imagining")
        output_dict["coord_type"] = coord_type
        return output_dict

    def clean_cache(self, st):
        if self.use_WM:
            st.clean_up_history(key=self.imagine_obs_key)
            st.clean_up_history(key=self.imagine_action_key)
        return st

    def save_gt_rgb(self, init_turn_degree, save_dir, forward_dist):
        """
        Fetch the ground truth observation for the given initial turn degrees and forward distance.
        """
        agent = self.sim.get_agent(0)

        theta = init_turn_degree / 180 * np.pi
        origin_agent_state = agent.get_state()
        tar_pos, tar_rot = rotate_and_forward_agent(
            origin_agent_state, theta, forward_dist
        )
        if not self.sim.pathfinder.is_navigable(tar_pos):
            return False

        obs, state_imgs = self.get_observations(self.sim, tar_pos, tar_rot.components)
        # set_agent_coordinates(self.sim, origin_pos, origin_rot)
        agent.set_state(origin_agent_state)     #reset the pos changed in get_observations

        for k, v in state_imgs.items():
            if "rgb" in k:
                save_image(v / 255.0, osp.join(save_dir, f"target_pano_{k}.png"))
                print(f"\tSaved ground truth image to: {save_dir}/target_pano_{k}.png")
        return True


# ================== Heuristic Policy Helpers ==================
    def heur_sample_next_action_seqs(self, action_seq, query_num):
        """
        Given past action_traj, sample `query_num` next action_seqs.
        Returns:
            action_seqs_u: List[List[str]]          # actions (text)
            action_seqs_idxs_u: List[List[str]]     # their indices (strings) in the same order
        """
        seen = set()
        action_seqs_u, action_seqs_idxs_u = [], []

        for ith_query in range(1000 * query_num):
            # * break when enough
            if len(action_seqs_u) == query_num:
                break

            new_action_idx_seq = []
            new_action_seq = copy.deepcopy(action_seq)
            for ith_action in range(self.look_ahead_action_num):
                seed = ith_query * self.look_ahead_action_num + ith_action
                next_action, next_action_idx = self.heur_sample_next_action(
                    new_action_seq, seed
                )
                new_action_seq.append(next_action)
                new_action_idx_seq.append(next_action_idx)

            tup = tuple(new_action_seq[len(action_seq) :])
            if tup not in seen:
                seen.add(tup)
                action_seqs_u.append(new_action_seq[len(action_seq) :])
                action_seqs_idxs_u.append(new_action_idx_seq)

        return action_seqs_u, action_seqs_idxs_u

    def heur_sample_next_action(self, action_seq, seed, max_rep_turns=4):
        """
        Given past action_traj, sample single next action
        """
        action_space_idxs, action_space_text = self.retrieve_action_space()

        action_space = list(action_space_text)
        if len(action_seq) > 0:
            last_move = action_seq[-1]

            # * 1. left -!> right && right -!> left
            if "left" in last_move:
                action_space = filter(lambda x: "right" not in x, action_space)
            elif "right" in last_move:
                action_space = filter(lambda x: "left" not in x, action_space)

            # * 2. max repetitive turns
            if (("left" in last_move or "right" in last_move)
                and action_seq[-max_rep_turns:] == [last_move] * max_rep_turns
            ):
                action_space = filter(lambda x: x != last_move, action_space)

        next_action = random.Random(seed).choice(list(action_space))

        # get the idx of the next_action from action_space_idxs
        idx = action_space_text.index(next_action)
        next_action_idx = action_space_idxs[idx]
        return next_action, next_action_idx

    def retrieve_action_space(self):
        action_space_all = construct_action_space_text(
            choice_format=self.planner_N_action.choice_format,
            include_stop=False,
        )
        action_space_all = action_space_all.split("\n")

        action_space_idxs, action_space_text = [], []
        for action in action_space_all:
            action_idx, action_text = action.split(". ")[0], action.split(". ")[1]
            action_space_idxs.append(action_idx)
            action_space_text.append(action_text)
        return action_space_idxs, action_space_text


# ================= Common CLI & Launch Utilities =================
def build_common_arg_parser(defaults={}):
    """Create an ArgumentParser pre-populated with common solver args.

    Args:
        world_model_choices: Optional[List[str]] of allowed world model types.
        defaults: Optional[Dict[str, Any]] to override per-solver defaults.

    Returns:
        argparse.ArgumentParser with common args added.
    """
    parser = argparse.ArgumentParser()

    # Experiment and logging
    parser.add_argument("--exp_id", type=str, default=defaults.get("exp_id", "03.15_exp"))
    parser.add_argument("--log_output_dir", type=str, default=defaults.get("log_output_dir", "downstream/logs"))

    # Models
    parser.add_argument("--answerer_model", type=str, default=defaults.get("answerer_model", "gpt-4o-mini-2024-07-18"))
    parser.add_argument("--planner_model", type=str, default=defaults.get("planner_model", "gpt-4o-mini-2024-07-18"))
    parser.add_argument(
        "--world_model_type",
        type=str,
        default=defaults.get("world_model_type", None),
        choices=["action", "text", "camera", "FTtext"],
    )

    # Infrastructure
    parser.add_argument("--WM_host", type=str, default=defaults.get("WM_host", "127.0.0.1:6000"))
    parser.add_argument("--sam2_host", type=str, default=defaults.get("sam2_host", "127.0.0.1:6001"))
    parser.add_argument("--vllm_host", type=str, nargs="+", default=defaults.get("vllm_host", ["localhost:8000"]))
    parser.add_argument("--worker_num", type=int, default=defaults.get("worker_num", 4))
    parser.add_argument("--api_key", type=str, default=defaults.get("api_key", "XXXX"), help="API key (openai)")
    parser.add_argument("--use_WM", action="store_true", help="Use WM or not")

    return parser


def launch_multiprocessing(args, target_fn):
    """Launch workers with a consistent spawn strategy.

    Args:
        args: Parsed argparse.Namespace with at least worker_num and api_key.
        target_fn: Callable(parallel_ith, args, api_key) -> Any
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")

    if args.worker_num == 1:
        target_fn(0, args, args.api_key)
    else:
        print(f"Using multiprocessing with {args.worker_num} workers.")
        processes = []
        for i in range(args.worker_num):
            p = ctx.Process(
                target=target_fn,
                args=(i, args, args.api_key),
            )
            processes.append(p)
            p.start()
            time.sleep(11)

        for p in processes:
            p.join()

    print("#" * 100 + "\n" + "#" * 100)
    print("All processes completed")
    print("#" * 100 + "\n" + "#" * 100)
