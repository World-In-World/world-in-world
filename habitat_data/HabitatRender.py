import habitat_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import habitat
from habitat.config.read_write import read_write
import os
import numpy as np
from collections import defaultdict
import torch
import json
from copy import deepcopy
from accelerate.logging import get_logger
import re
import random
import time

os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

logger = get_logger(__name__, log_level="INFO")

from .habitat_util import (
    configure_cubemap_sensors,
    configure_equirect_tfm,
    set_agent_heading,
    nearest_neighbor_tsp, best_middle_neighbor,
    cal_img_near_black_ratio,
    save_episodes,
    find_leaf_candidates,
)
from .config_utils import hm3d_config, mp3d_config, hssd_config, hm3d_pn_config, mp3d_pn_config


class HabitatRenderer:
    def __init__(self, dataset_name, fix_seed, output_dir, black_ratio_thr,
                 enable_depth, height, width, traj_num, gpu_id, worker_id,
                 forward_step_size=0.2, turn_angle=22.5, use_presaved_ep=True):

        self.dataset_name = dataset_name
        self.fix_seed = fix_seed
        self.output_dir = output_dir
        self.black_ratio_thr = black_ratio_thr   # Correctly assign the passed parameter
        self.enable_depth = enable_depth         # Assign new parameters
        self.height = height
        self.width  = width
        self.traj_num = traj_num
        self.gpu_id = gpu_id
        self.worker_info = f"[Render worker_id: {worker_id}]:"
        self.use_presaved_ep = use_presaved_ep
        if worker_id != -1:
            self.verbose=True
        else:
            self.verbose=False

        self.recorder = None
        self.planner = None
        self.all_data = defaultdict(lambda: defaultdict(dict))  # Nested dictionary
        self.all_steps = 0
        self.traj_idx = 0
        self.forward_step_size = forward_step_size
        self.turn_angle = turn_angle
        self.sensor_height = 1.5  # Sensor height in meters
        self.pending_episodes = {}

        # For cluster_sampling:
        self.depth_scale = 20.0
        self.explore_radius = 0.5
        self.goal_radius = 0.25
        self.max_step_limit = 7500
        self.sampled_point_density = 4
        self.select_leaf_ratio = 0.32 / self.sampled_point_density
        assert self.explore_radius > self.goal_radius, "Explore radius should be larger than goal radius."
        self.all_sample_points, self.all_dist_matrix = {}, {}
        self.adjust_curr_position = False

        self.init_habitat_env()

    def init_habitat_env(self):
        """
        Initializes the Habitat-Sim environment.
        """
        # Load Habitat configuration
        dataset_name, split = self.dataset_name.split('-')
        if dataset_name == 'hm3d':
            if split == 'train' and self.use_presaved_ep:
                self.ep_presaved_path = 'data/datasets/episodes-hm3d-train_unique.pth'    #Prev: data/datasets_/XXX
                split = 'val'
            habitat_config = hm3d_pn_config(stage=split, episodes=-1)   #800 for train
        elif dataset_name == 'hssd':
            habitat_config = hssd_config(stage=split, episodes=-1)
        elif dataset_name == 'mp3d':
            if split == 'train' and self.use_presaved_ep:
                self.ep_presaved_path = 'data/datasets/episodes-mp3d-train_unique.pth'
                split = 'val_mini'
            # habitat_config = mp3d_pn_config(stage=split, episodes=-1)
            habitat_config = mp3d_config(stage=split, episodes=-1)

        with read_write(habitat_config):
            # Modify simulator settings if needed
            habitat_config.habitat.simulator.forward_step_size = self.forward_step_size
            habitat_config.habitat.simulator.turn_angle = self.turn_angle
            habitat_config.habitat.environment.max_episode_steps = self.max_step_limit*100  # to make sure not exceed the limit
            habitat_config.habitat.simulator.habitat_sim_v0['gpu_device_id'] = self.gpu_id

        # Setup cubemap cameras and equirect transformers
        sensor_uuids_dict = configure_cubemap_sensors(
            habitat_config.habitat.simulator,
            self.enable_depth,
            self.depth_scale,
            sensor_height=self.sensor_height,
            enable_semantic=False,
        )
        cube2equirect_tfms = configure_equirect_tfm(
            self.height,
            self.width,
            self.enable_depth,
            sensor_uuids_dict,
            enable_semantic=False,
        )

        # Create Habitat environment
        self.habitat_env = habitat.Env(habitat_config)
        if self.fix_seed:
            self.habitat_env.sim.seed(42)
            random.seed(42)
        else:
            current_time_ns = time.time_ns()
            seed_value = current_time_ns % (2**32)
            random.seed(seed_value)
            self.habitat_env.sim.seed(seed_value)
        print(self.worker_info+f"Successfully initialized the Habitat env for dataset <{self.dataset_name}>.")
        print(self.worker_info+f'Num of current episodes = {len(self.habitat_env.episodes)}')

        self.sensor_uuids = sensor_uuids_dict
        self.cube2equirect = cube2equirect_tfms
        # self.all_env_episodes = deepcopy(self.habitat_env.episodes)
        # random.shuffle(self.habitat_env.episodes)
        if hasattr(self, 'ep_presaved_path'):
            self.all_episodes = torch.load(self.ep_presaved_path)
            print(self.worker_info+f'Using presaved episodes for {self.dataset_name}...')
            print(self.worker_info+f'Num of current episodes after loaded = {len(self.all_episodes)}')
        else:
            self.all_episodes = self.extract_unique_ep(self.habitat_env.episodes)   #is a dict
        # self.extract_presaved_eps(self.habitat_env.episodes)
        # self.validate_episodes(self.all_episodes)

        items = list(self.all_episodes.items())
        random.shuffle(items)
        self.pending_episodes = dict(items)

    def extract_presaved_eps(self, all_episodes, only_unique_eps=True):
        if only_unique_eps:
            all_episodes = self.extract_unique_ep(all_episodes)
            path = f'data/datasets/episodes-{self.dataset_name}_unique_PN.pth'
        else:
            path = f'data/datasets/episodes-{self.dataset_name}.pth'
        # for scene_id, ep in all_episodes.items():
        # if ep.episode_id in [8413]:   #episode id: 8413, scene id: data/scene_datasets/hm3d/train/00710-DGXRxHddGAW/DGXRxHddGAW.basis.glb
        #     self.validate_episodes({scene_id: ep})

        print(self.worker_info+f'Number of saved episodes = {len(all_episodes)} from: {path}')
        torch.save(all_episodes, path)
        exit(0)

    def validate_episodes(self, episodes):
        """Validates the episodes by checking if they can be load into habitat-sim correctly."""
        # episodes is a dict
        items = list(episodes.items())
        random.shuffle(items)
        episodes = dict(items)
        print(self.worker_info+f'==> To be validated episodes: {episodes.keys()}')

        for scene_id, ep in episodes.items():
            try:
                os.makedirs(os.path.join(self.output_dir, scene_id), exist_ok=False)
                next_ep = ep
            except FileExistsError:
                print(self.worker_info+f"Data already collected for scene <{scene_id}>. Skipping...")
                continue

            self.habitat_env.episodes = [next_ep]
            self.habitat_env.current_episode = next_ep
            print(self.worker_info+f"==> Current episode id: {self.habitat_env.current_episode.episode_id}, scene id: {self.habitat_env.current_episode.scene_id}")
            self._reset_sim()
        print(self.worker_info+f"==> Episodes validation completed successfully!")
        exit(0)

    def sample_nav_points(self, num_max_iter, num_random_nav_point, st_point):
        points = []
        for _ in range(num_random_nav_point-1):
            found_path = False
            while not found_path:
                # Create a path request
                p = self.habitat_env.sim.pathfinder.get_random_navigable_point()
                # p_ = self.habitat_env.sim.pathfinder.get_random_navigable_point_near(circle_center=p, radius=0.5)
                # validate the path:
                found_path, path_points = self.get_path_topoint(st_point, p)
                if found_path:
                    dist = self.habitat_env.sim.geodesic_distance(st_point, p)
                    found_path = found_path and np.isfinite(dist)

                if num_max_iter <= 0: break
                num_max_iter -= 1
            if found_path:
                points.append(p)
            else:
                print(self.worker_info+f"WARNING: When sampling nav points, it reached the max iteration limit.") if self.verbose else None
                break
        return points

    def cal_dist_matrix_sym(self, cluster_centers):
        N = len(cluster_centers)
        dist_matrix = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in range(i, N):  # Start j from i to compute only the upper triangle
                if i == j:
                    dist_matrix[i, j] = 0.0
                else:
                    dist = self.habitat_env.sim.geodesic_distance(
                            cluster_centers[i], cluster_centers[j],
                        )
                    if not np.isfinite(dist):
                        print(self.worker_info+f"Serious WARNING (precompute): The distance between two points {cluster_centers[i]} and {cluster_centers[j]} is infinite for scene {self.habitat_env.current_episode.scene_id}.")
                        dist_matrix[i, j] = 15; dist_matrix[j, i] = 15        # assign a temp value
                    else:
                        dist_matrix[i, j] = dist; dist_matrix[j, i] = dist
        return dist_matrix

    def cal_dist_matrix(self, points_a, points_b):
        """
        Compute the distance matrix between two lists of 3D points.
        points_a: List of np.array([x, y, z]) with shape (3,). Length N.
        points_b: List of np.array([x, y, z]) with shape (3,). Length M.
        Returns:
            dist_matrix: A numpy array of shape (N, M) where
                dist_matrix[i, j] is the geodesic distance between points_a[i] and points_b[j].
                If unreachable, self.habitat_env.sim.geodesic_distance might return inf.
        """
        N = len(points_a); M = len(points_b)
        dist_matrix = np.zeros((N, M), dtype=np.float32)

        for i in range(N):
            for j in range(M):
                # Compute the geodesic distance between the two points.
                dist = self.habitat_env.sim.geodesic_distance(points_a[i], points_b[j])
                if not np.isfinite(dist):
                    print(self.worker_info+f"Serious WARNING: The distance between two points {points_a[i]} and {points_b[j]} is infinite for scene {self.habitat_env.current_episode.scene_id}.")
                    dist_matrix[i, j] = 15        # assign a temp value
                else:
                    dist_matrix[i, j] = dist

        return dist_matrix

    def sample_trajs(self,
                     trajs_num=1,
                     num_max_iter=70000):
        """
        Samples valid trajectories for the given scene, but instead of
        just start->end, we:
         1) Sample 'num_random_nav_point' points on the navmesh
         2) Build a distance matrix for them
         3) Use a simple Nearest Neighbor TSP approach to order them
         4) Return that route as a single trajectory (or multiple if trajs_num>1)
        """
        trajs = []; left_points = []
        # For each trajectory we want to create:
        scene_area = self.habitat_env.sim.pathfinder.navigable_area     # assuem the area in unit m^2
        n_samples = int(self.sampled_point_density * scene_area)        # avg 1 points per m^2
        n_samples = min(n_samples, self.max_step_limit)
        if n_samples > 1400:
            print(self.worker_info+f"WARNING: The number of random nav points is too large: {n_samples}.") if self.verbose else None
            n_samples = 1400
        print(self.worker_info+f"Sampling <{n_samples}> points for each trajectory.") if self.verbose else None

        while len(trajs) < trajs_num:
            # 1) Sample N random navigable points
            st_point = self.habitat_env.sim.agents[0].state.position
            points = self.sample_nav_points(num_max_iter, n_samples, st_point)
            if len(points) < n_samples-1:
                print(self.worker_info+f"WARNING: Max iterations reached. this scene can not generate enough <{n_samples}> points. Current points: <{len(points)}>.") if self.verbose else None

            leaf_cands_idxs, left_node_idxs = self.generate_leafnode_idxs(points, init_store=True)

            if len(leaf_cands_idxs) <= 2:
                print(self.worker_info+"WARNING: Not enough leaf candidates found. The scene may be too small. Skip it!") if self.verbose else None
                break

            selected_points = [st_point] + [points[i] for i in leaf_cands_idxs]
            trajs.append(selected_points)
            left_points.append([points[i] for i in left_node_idxs])
            print(self.worker_info+f"Created route for trajectory <{len(trajs)}> with num of leaf points <{len(selected_points)}>") if self.verbose else None
        return trajs, left_points

    def generate_leafnode_idxs(self, points, filter_radius=3.0, init_store=False):
        # 2) Build NxN distance matrix of geodesic distance
        if init_store:
            dist_matrix = self.cal_dist_matrix_sym(points)  #generation the distance symmetric dist matrix
            self.all_dist_matrix[self.traj_idx] = dist_matrix
            self.all_sample_points[self.traj_idx] = points
            self.leaf_node_num = int(self.select_leaf_ratio * len(points))
        else:
            assert self.traj_idx < len(self.all_dist_matrix), "The traj_idx should be less than the length of all_dist_matrix."
            # Retrieve the stored full points and distance matrix for the current trajectory.
            stored_points = np.array(self.all_sample_points[self.traj_idx])  # shape: (N, 3)
            full_dist_matrix = self.all_dist_matrix[self.traj_idx]
            # For each point in the current list, use np.where to find the matching index.
            indices = []
            for p in points:
                # p is assumed to be exactly equal to one of the stored points.
                res = np.where((stored_points == p).all(axis=1))[0]
                if res.size == 0:
                    raise ValueError("A provided point was not found in the stored sample points.")
                indices.append(res[0])
            # Extract the submatrix corresponding to the current points.
            dist_matrix = full_dist_matrix[np.ix_(indices, indices)]

        leaf_scores = find_leaf_candidates(dist_matrix)        # retun a seq in decreasing order of the leaf score

        leaf_node_num = self.leaf_node_num
        leaf_scores_ = deepcopy(leaf_scores)  # a list of (node_idx, score), sorted by descending leaf score
        leaf_cands_idxs = []

        while len(leaf_cands_idxs) < leaf_node_num and len(leaf_scores_) > 0:
            node_idx, score = leaf_scores_.pop(0)
            leaf_cands_idxs.append(node_idx)
            # Remove from leaf_scores_ any node within 'filter_radius' of node_idx
            leaf_scores_ = [
                (idx, score) for idx, score in leaf_scores_
                if dist_matrix[node_idx, idx] >= filter_radius
            ]
            # leaf_node_num = min(leaf_node_num, len(leaf_scores_))
        # collect all the left idxs in leaf_scores_
        left_node_idxs = [i for i, s in leaf_scores_]
        return leaf_cands_idxs, left_node_idxs

    def get_path_topoint(self, st_point, end_point):
        """Get the shortest path from start point to end point."""
        # validate the path:
        path = habitat_sim.ShortestPath()
        path.requested_start = st_point
        path.requested_end = end_point
        found_path = self.habitat_env.sim.pathfinder.find_path(path)
        return found_path, path.points


    def generate_rand_actions(self, length: int) -> list[int]:
        """
        Return 2·length elements:   A₀, 0, A₁, 0, …, A_{length−1}, 0
        where each Aᵢ is drawn with probabilities
            turn-left (2)  : 1/6
            turn-right (3) : 1/6
            forward (1)    : 4/6
        The fixed seed keeps the sequence reproducible across runs.
        """
        primary = random.choices([2, 3, 1], weights=[1, 1, 5], k=length)
        # add 0 as stop at the end
        primary.append(0)
        return primary


    def navigate_to_waypoint_w_rand_actions(self, waypoint, waypoint_list):
        """Navigates the agent to the given waypoint."""
        prev_action_is0 = False
        pending_actions = self.generate_rand_actions(25)

        while True:
            curr_position = self.habitat_env.sim.agents[0].state.position
            action = pending_actions.pop(0)  # Get the next action from the list

            if action == 0:
                if prev_action_is0:
                    print(self.worker_info+f"Serious WARNING: The agent is stuck at the same position."+self.get_point_warrning(waypoint, curr_position)) if self.verbose else None
                prev_action_is0 = True

                print(self.worker_info+f"Jump to the next waypoint <{waypoint}> with random actions.") if self.verbose else None
                self.remove_bug_points(waypoint, waypoint_list)
                self.adjust_curr_position = True
                self.tried_position = waypoint
                break
            else:
                prev_action_is0 = False
                obs = self.habitat_env.step(action)
                yield obs, action

    def navigate_to_waypoint(self, waypoint, waypoint_list):
        """Navigates the agent to the given waypoint."""
        prev_action_is0 = False
        while True:
            curr_position = self.habitat_env.sim.agents[0].state.position
            action = self.planner.get_next_action(waypoint)
            dist = self.habitat_env.sim.geodesic_distance(
                curr_position, waypoint
            )
            if action == 0:  # Goal reached
                if prev_action_is0:
                    print(self.worker_info+f"Serious WARNING: The agent is stuck at the same position."+self.get_point_warrning(waypoint, curr_position)) if self.verbose else None
                    self.remove_bug_points(waypoint, waypoint_list)
                prev_action_is0 = True
                if dist > self.goal_radius:
                    print(self.worker_info+f"Serious WARNING: The agent is not close enough to the goal with dist {dist}."+self.get_point_warrning(waypoint, curr_position)) if self.verbose else None
                    self.remove_bug_points(waypoint, waypoint_list)
                    self.adjust_curr_position = True
                    self.tried_position = waypoint
                break
            else:
                prev_action_is0 = False
            obs = self.habitat_env.step(action)
            yield obs, action

    def get_point_warrning(self, waypoint, curr_position):
        return f"\n\tCurrent waypoint is {waypoint}, current position is {curr_position}, current scene is {self.habitat_env.current_episode.scene_id}, current steps is {self.all_steps}."

    def process_pano_obs(self, obs):
        """
        Processes the observation by converting cube maps to equirectangular format.
        Args:
            obs (dict): A dictionary containing cube map observations. The keys are sensor names (e.g., 'rgb', 'depth')
                        and the values are numpy arrays [0~255] representing the images captured by the sensors.
        Returns:
            tuple: A tuple containing:
                - rgb (torch.Tensor): The equirectangular RGB image tensor of shape [1, 3, H, W].
                - depth (torch.Tensor or None): The equirectangular depth image tensor of shape [1, 1, H, W]
                                                if depth is enabled, otherwise None.
        """
        obs_tensor = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}      #obs_tensor is shape of [1, H, W, 3]
        obs_eq_r = self.cube2equirect['rgb'](obs_tensor)[self.sensor_uuids['rgb'][0]].squeeze(0)        #Tensor, range: [0, 255]
        if self.enable_depth:
            obs_eq_d = self.cube2equirect['depth'](obs_tensor)[self.sensor_uuids['depth'][0]].squeeze(0)    #Tensor
            obs_eq_d = (obs_eq_d / self.depth_scale * 255.0)      #.to(dtype=torch.uint8)
            depth = obs_eq_d.unsqueeze(0)
        else:
            depth = None

        rgb = obs_eq_r.permute(2, 0, 1).unsqueeze(0)        #(H, W, 3), rgb -> (1, 3, H, W), rgb
        return rgb, depth

    def check_black_ratio(self, eq_rgb, scene_id, traj_id):
        """Updates trajectory data and saves observations."""
        ratio_black = cal_img_near_black_ratio(eq_rgb)
        self.all_data[f"scene-{scene_id}"][f"traj-{traj_id}"]["steps"][f"step-{self.all_steps}"]["blackRatio"] = ratio_black

        return ratio_black > self.black_ratio_thr

    def store_metadata(self, scene_id, traj_id, traj, action, agent_coord, camera_coord, waypoint_idx, steps_id):
        """Stores metadata into the tree-structured data dictionary."""
        if f"traj-{traj_id}" not in self.all_data[f"scene-{scene_id}"]:
            traj = [item.tolist() for item in traj]
            self.all_data[f"scene-{scene_id}"][f"traj-{traj_id}"] = {"trajPoints": traj, "steps": defaultdict(lambda: defaultdict(dict))}

        self.all_data[f"scene-{scene_id}"][f"traj-{traj_id}"]["steps"][f"waypoint-{waypoint_idx}"][f"step-{steps_id}"] = {
            "action": action,
            "coord": agent_coord,
            "habitat_camera_coord": camera_coord,
        }


    def reset_uni_episodes(self, ep_list_all):
        """Resets to the unique episodes for the current habitat_env."""
        scene_episodes_ = self.extract_unique_ep(ep_list_all)

        scene_episodes = {k: v for k, v in scene_episodes_.items() if 'qvNra81N8BU.basis.glb' not in k}
        if len(scene_episodes) < len(scene_episodes_):
            print(self.worker_info+f"WARNING: Removing the broken scene <qvNra81N8BU.basis.glb>.") if self.verbose else None

        # scene_episodes = {k: v for k, v in scene_episodes.items() if '107734188_176000034' in k or '105515505_173104584' in k}
        ep_list = list(scene_episodes.values())
        print(self.worker_info+f"all episodes/Number of unique scenes = {len(ep_list_all)}/{len(ep_list)}") if self.verbose else None
        self.habitat_env.episodes = ep_list

    def reset_agent(self, start_point):
        """Resets the agent to the start point."""
        assert self.habitat_env._episode_from_iter_on_reset is True
        self.habitat_env._episode_from_iter_on_reset = False
        obs_ = self.habitat_env.reset()
        self.habitat_env._episode_from_iter_on_reset = True

        # Set the new state for agent_state in the scene:
        agent_state = self.habitat_env.sim.agents[0].state
        agent_state.position = start_point
        # agent_state.rotation = set_agent_heading(start_point[0], start_point[2], traj[1][0], traj[1][2])
        self.habitat_env.sim.agents[0].set_state(agent_state)

        # Set the necessary vars to store the history info:
        self.planner = ShortestPathFollower(self.habitat_env.sim, goal_radius=self.goal_radius, return_one_hot=False)
        # self.recorder = Recoder(self.habitat_env)
        self.all_data = defaultdict(lambda: defaultdict(dict))
        self.all_steps = 0

    def reset_heading(self, traj):
        """Resets the agent to the start point."""
        # Set the new state for agent_state in the scene:
        agent_state = self.habitat_env.sim.agents[0].state
        # agent_state.position = traj[0]
        agent_state.rotation = set_agent_heading(traj[0][0], traj[0][2], traj[1][0], traj[1][2])
        self.habitat_env.sim.agents[0].set_state(agent_state)

        # Set the necessary vars to store the history info:
        self.all_data = defaultdict(lambda: defaultdict(dict))

    def reset_position(self, waypoint):
        """Resets the agent to the start point."""
        # Set the new state for agent_state in the scene:
        agent_state = self.habitat_env.sim.agents[0].state
        agent_state.position = waypoint
        # agent_state.rotation = set_agent_heading(traj[0][0], traj[0][2], traj[1][0], traj[1][2])
        self.habitat_env.sim.agents[0].set_state(agent_state)

    def _reset_sim(self):
        """Resets the episode from self.pending_episodes"""
        # assert self.habitat_env._episode_from_iter_on_reset is True
        self.habitat_env._episode_from_iter_on_reset = False
        # try:
        obs_ = self.habitat_env.reset()
        success = True
        # except Exception as e:
        #     print(self.worker_info+f"Error: Current episode id: {self.habitat_env.current_episode.episode_id}, scene id: {self.habitat_env.current_episode.scene_id}")
        #     print(self.worker_info+f"Error: Encountered an error while resetting the habitat env: {e}. Trying another episode...")
        #     success = False
        self.habitat_env._episode_from_iter_on_reset = True
        return success

    def update_pending_episodes(self):
        """walk through the output_dir and chekc the scene_id folder, if it is already exist, then remove it from the pending_episodes"""
        # while len(self.pending_episodes) == 0:
        next_ep = None
        while next_ep is None:
            new_complete_eps = []
            for scene_id, ep in self.pending_episodes.items():
                # Skip if data already exists for the scene:
                if os.path.exists(os.path.join(self.output_dir, scene_id)):
                    print(self.worker_info+f"Data already collected for scene <{scene_id}>. Skipping...") if self.verbose else None
                    new_complete_eps.append(scene_id)
                    continue
                else:
                    try:
                        os.makedirs(os.path.join(self.output_dir, scene_id), exist_ok=False)
                        next_ep = ep; next_scene_id = scene_id
                        break
                    except FileExistsError:
                        print(self.worker_info+f"Data already collected for scene <{scene_id}>. Skipping...") if self.verbose else None
                        new_complete_eps.append(scene_id)
                        continue
            self.pending_episodes = {k: v for k, v in self.pending_episodes.items() if k not in new_complete_eps}   # Q: what will happen if self.pending_episodes is {}? A:
            if len(self.pending_episodes) == 0:
                assert next_ep is None, "next_ep should be None when all episodes are completed."
                self.update_output_path()
                items = list(self.all_episodes.items())
                random.shuffle(items)
                self.pending_episodes = dict(items)
        return next_scene_id, next_ep

    def set_current_episodes(self):
        """Resets to the unique episodes for the current habitat_env."""
        next_scene_id, next_ep = self.update_pending_episodes()
        self.habitat_env.episodes = [next_ep]
        self.habitat_env.current_episode = next_ep
        return next_scene_id

    def extract_unique_ep(self, ep_list):
        scene_episodes = {}
        rnd = random.Random(42)
        rnd.shuffle(ep_list)
        for ep in ep_list:
            scene_id = os.path.basename(ep.scene_id)
            scene_episodes[scene_id] = ep

        # scene_episodes = self.get_certain_scene(scene_episodes)
        print(self.worker_info+f"all episodes/Number of unique scenes = {len(ep_list)}/{len(scene_episodes)}") if self.verbose else None
        return scene_episodes

    def get_certain_scene(self, scene_episodes):
        target_scene = ['8LLjiNrWzJ9.basis.glb', 'U3oQjwTuMX8.basis.glb', 'LcAd9dhvVwh.basis.glb', 'xvDx98avcwd.basis.glb', 'UuwwmrTsfBN.basis.glb', 'j2EJhFEQGCL.basis.glb', '4tdJ3qe1x7P.basis.glb', 'L5QEsaVqwrY.basis.glb', 'RHdkyzXFp1k.basis.glb', 'WpAGGyZFqQj.basis.glb', 'PPTLa8SkUfo.basis.glb', 'dTzYwo8Hppu.basis.glb', '1k479icNeHW.basis.glb', 'TQSiMZJawkS.basis.glb', 'as8Y8AYx6yW.basis.glb', 'CFVBbU9Rsyb.basis.glb', 'ZB8o8rMmPdB.basis.glb', 'yVbpFay8gTU.basis.glb', 'kdw2Uapns3b.basis.glb', 'HfMobPm86Xn.basis.glb', 'wrq3kiEU4VR.basis.glb', 'b2e31HFFizw.basis.glb', '1Rg1SS1dRpG.basis.glb', 'XiJhRLvpKpX.basis.glb', 'q28T9C3q2dv.basis.glb', 'ochRmQAHtkF.basis.glb', '27cQLjQ5CjV.basis.glb', 'JNiWU5TZLtt.basis.glb', 'rrjjmoZhZCo.basis.glb', '1W61QJVDBqe.basis.glb', 'wDDRygUCLMm.basis.glb', 'v3tsKAPVLJS.basis.glb', '5RtSdesLuHt.basis.glb', 'PyZonHqd5gy.basis.glb', '7GvCP12M9fi.basis.glb', 'FXuXGH9YQTW.basis.glb', 'FgXPKxNp5kK.basis.glb', 'DACaFbApXUe.basis.glb', 'UQ5EhY5wve1.basis.glb', 'u9LiqMn6kA6.basis.glb', 'yqNxxJnA3iL.basis.glb', '37c5w29pYm3.basis.glb', '9DnDAhJ7qcj.basis.glb', 'WeyCwVzL53K.basis.glb', '812QqCky3T7.basis.glb', 'b3WpMbPFB6q.basis.glb', '16tymPtM7uS.basis.glb', 'JFgrz9MNz4b.basis.glb', 'S3BfyR31Wc9.basis.glb', 'gxttMtT5ZGK.basis.glb', 'ECStCRoCNWM.basis.glb', 'g7hUFVNac26.basis.glb', 'gDDiZeyaVc2.basis.glb', 'gUqgeUmUagL.basis.glb', 'bB6nKqfsb1z.basis.glb', 'JptJPosx1Z6.basis.glb', 'EN7GiDgxdQ2.basis.glb', '1sM6KvYg3J5.basis.glb', 'zwzTbNq7xoW.basis.glb', 'VYnUX657cVo.basis.glb', 'EbiLVt7CHc1.basis.glb', 'QKfBMSSy7Hy.basis.glb', 'PXAfUkZGMdU.basis.glb', 'qQgcM8T4hiD.basis.glb', 'bHKTDQFJxTw.basis.glb', 'mt9H8KcxRKD.basis.glb', 'gmuS7Wgsbrx.basis.glb', 'GMwtBqNLGBs.basis.glb', 'nYYcLpSzihC.basis.glb']
        scene_episodes_new = {}
        for scene_id, ep in scene_episodes.items():
            if scene_id in target_scene:
                scene_episodes_new[scene_id] = ep
        return scene_episodes_new

    def collect_data(self):
        """Main data collection loop."""
        # self.reset_uni_episodes(self.habitat_env.episodes)
        while True:
            scene_id = self.set_current_episodes()
            success_flag = self._reset_sim()
            if success_flag is False:
                continue
            # for i in range(len(self.habitat_env.episodes)):
            # self.reset_episode()
            assert scene_id == self.habitat_env.current_episode.scene_id.split("/")[-1]
            self.traj_idx = 0
            sampled_trajs, left_points_all = self.sample_trajs(trajs_num=self.traj_num)
            # sampled_trajs = self.sampled_traj_straight()
            for traj, left_points in zip(sampled_trajs, left_points_all):
                self.reset_agent(traj[0])
                waypoint_list = traj[1:]
                print(self.worker_info+f"Start collecting data for scene <{scene_id}> with trajectory <{self.traj_idx}>") if self.verbose else None
                traj_dir = os.path.join(self.output_dir, scene_id, f'traj-{str(self.traj_idx)}')

                waypoint_idx = 0
                while len(waypoint_list) > 0:
                    waypoint_dir = os.path.join(traj_dir, f'waypoint-{str(waypoint_idx)}')
                    if self.adjust_curr_position:
                        self.reset_position(self.tried_position)
                        self.adjust_curr_position = False
                    curr_position = self.habitat_env.sim.agents[0].state.position
                    waypoint_list, left_points = self.filter_explored_points(curr_position, [waypoint_list,left_points])
                    if len(waypoint_list) == 0: break
                    curr_waypoint = self.select_next_waypoint(curr_position, waypoint_list)

                    found_path, path_points = self.get_path_topoint(curr_position, curr_waypoint)
                    if found_path:
                        self.reset_heading(path_points)
                    else:
                        print(self.worker_info+f"Serious WARNING: The agent can not find the path to the next waypoint."+self.get_point_warrning(curr_waypoint, curr_position)) if self.verbose else None

                    warrning_count, waypoint_step = 0, 0
                    # for obs, action in self.navigate_to_waypoint_w_rand_actions(curr_waypoint, waypoint_list):
                    for obs, action in self.navigate_to_waypoint(curr_waypoint, waypoint_list):
                        # Record coordinates
                        agent_state = self.habitat_env.sim.agents[0].state
                        curr_position = agent_state.position
                        quat = agent_state.rotation
                        camera_position = agent_state.sensor_states['rgb_front'].position
                        camera_quat = agent_state.sensor_states['rgb_front'].rotation
                        agent_coord = [curr_position.tolist(), (quat.w, quat.x, quat.y, quat.z)]
                        camera_coord = [camera_position.tolist(), (camera_quat.w, camera_quat.x, camera_quat.y, camera_quat.z)]
                        # Update the explored points in the waypoint_list
                        waypoint_list, left_points = self.filter_explored_points(curr_position, [waypoint_list,left_points])

                        # Process and record panorama (equirectangular) observations
                        rgb_eq, depth_eq = self.process_pano_obs(obs)
                        obs_dict = {'rgb_eq': rgb_eq, 'depth_eq': depth_eq, 'action': action,
                                    'scene_id': scene_id, 'traj_id': self.traj_idx,
                                    'worker_id': re.findall(r'\d+', self.worker_info)[0],
                                    'traj_dir': traj_dir, 'all_step_id': self.all_steps,
                                    'waypoint_dir': waypoint_dir, 'waypoint_idx': waypoint_idx, 'step_id': waypoint_step,
                                    'waypoint_list': waypoint_list, 'left_points': left_points}
                        yield obs_dict

                        # Store metadata
                        self.store_metadata(scene_id, self.traj_idx, traj, action,
                                            agent_coord, camera_coord, waypoint_idx, waypoint_step)
                        # should_skip = self.check_black_ratio(obs_dict['rgb_eq'], scene_id, self.traj_idx)

                        # if should_skip:
                        #     warrning_count += 1
                        # else:
                        #     warrning_count = 0
                        if waypoint_step > 750:     # or warrning_count > 10
                            print(self.worker_info+f"WARNING: Jumping to next waypoint, because of too many steps "
                                f"on the current waypoint: {waypoint_step} or ratio_black > {self.black_ratio_thr} with warrning count: {warrning_count}.") if self.verbose else None
                            # remove curr_waypoint from list and jump to next waypoint
                            waypoint_list = [point for point in waypoint_list if not np.allclose(point, curr_waypoint)]
                            break
                        waypoint_step += 1
                        self.all_steps += 1

                    waypoint_list, left_points = self.update_waypoint_list(waypoint_list, left_points)
                    self.save_meta_data(waypoint_dir)
                    if self.all_steps > self.max_step_limit:
                        print(self.worker_info+f"Error: Jumping to next waypoint, because of too many steps "
                                f"on the current trajectory: {self.all_steps}.") if self.verbose else None
                        break
                    yield 'cancel_current-waypoint_finished'
                    waypoint_idx += 1

                # self.recorder.draw_waypoints(dir, traj)
                # self.recorder.save_trajectory(dir, save_video=False)
                print(self.worker_info+f"Finished collecting data for scene <{scene_id}> with trajectory <{self.traj_idx}>") if self.verbose else None
                yield 'cancel_current-scene_finished'          #which means we should cancel the current return section because of the goal is reached
                self.traj_idx += 1
            self.pending_episodes = {k: v for k, v in self.pending_episodes.items() if k != scene_id}

    def update_waypoint_list(self, waypoint_list, left_points):
        """Updates the waypoint list after reaching the current waypoint."""
        all_points = waypoint_list + left_points
        if len(all_points) > 2:
            leaf_cands_idxs, left_node_idxs = self.generate_leafnode_idxs(all_points)
            new_selected_points = [all_points[i] for i in leaf_cands_idxs]
            new_left_points = [all_points[i] for i in left_node_idxs]
        else:
            new_selected_points, new_left_points = waypoint_list, left_points
        return new_selected_points, new_left_points

    def select_next_waypoint(self, current_position, point_list):
        """Selects the next waypoint to navigate to."""
        dist_row = self.cal_dist_matrix([current_position], point_list)[0]
        next_waypoint = best_middle_neighbor(point_list, dist_row)
        return next_waypoint

    def filter_explored_points(self, curr_position, points_list):
        filtered_groups  = []
        for points in points_list:
            if len(points) > 0:
                dist_row = self.cal_dist_matrix([curr_position], points)[0]
                # fast implementation to cal 3D coord euclidean distance:
                # dist_row = np.linalg.norm(np.array(points) - curr_position, axis=1)
                points = [pt for pt, dist in zip(points, dist_row) if dist > self.explore_radius]
            filtered_groups.append(points)
        return filtered_groups

    def remove_bug_points(self, point, points_list):
        points_list_ = np.array(points_list)
        res = np.where((points_list_ == point).all(axis=1))[0]
        if res.size == 0:
            print(self.worker_info+f"Serious WARNING: The point is not found in the list.") if self.verbose else None
        else:
            points_list.pop(res[0])

    def update_output_path(self):
        print(self.worker_info + f"Finished collecting for <{os.path.basename(self.output_dir)}>!") if self.verbose else None
        # Extract the current iteration and increment it
        base_dir, current_iter = os.path.split(self.output_dir)
        iter_prefix, iter_num = current_iter.split('-')
        next_iter = f"{iter_prefix}-{int(iter_num) + 1}"
        self.output_dir = os.path.join(base_dir, next_iter)

    def save_meta_data(self, dir):
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, "metadata.json"), "w") as f:
            json.dump(self.all_data, f, indent=4)
        # print(f"Metadata successfully saved for {num_traj_id} trajectories!")

    def collect_data_batch(self, num_frames):
        """Collects data in batches without a try/except for StopIteration."""
        # Convert the collect_data generator to an iterator allowing a default value.
        data_iter = iter(self.collect_data())

        while True:
            data_batch = defaultdict(list)
            pending_frames = num_frames
            while pending_frames > 0:
                # next(...) returns 'None' instead of raising StopIteration when exhausted
                obs_dict = next(data_iter, None)

                if isinstance(obs_dict, dict):
                    for k, v in obs_dict.items():
                        data_batch[k].append(v)
                    pending_frames = num_frames - len(data_batch[k])
                elif obs_dict == 'cancel_current-waypoint_finished':
                    data_batch = defaultdict(list)
                    pending_frames = num_frames
                elif obs_dict == 'cancel_current-scene_finished':
                    continue
            yield data_batch


def render_and_record_worker(
    habitat_dataset,
    fix_seed,
    enable_depth,
    width,
    height,
    black_ratio_thr,
    output_dir,
    device_id,
    worker_id,
    trajs_num_per_scene,
    use_presaved_ep,
    log_output_dir,
    num_recoder=4,
):
    """
    A dedicated worker function that:
      1. Initializes a renderer on `device_id`.
      2. Creates TWO Recoders.
      3. Runs one producer thread (rendering) and two consumer threads (each recoder),
         all in parallel, using two queues.
    """
    import os
    import threading
    from queue import Queue

    from utils.logger import setup_logger
    from .recoder import Recoder

    # Optionally set up a per-worker logger
    setup_logger(os.path.join(log_output_dir, f"worker_{worker_id}.log"))
    print(f"[Worker {worker_id}] Starting on GPU {device_id} ...", flush=True)

    # 2) Create n Recoders
    recoders = [Recoder() for _ in range(num_recoder-1)]            # 1
    # We will store the same data into two separate queues
    data_queues = [Queue(maxsize=40) for _ in range(num_recoder-1)]

    def render_producer():
        """
        Collects data (frames, steps, etc.) from the renderer and
        puts each piece of data into BOTH queues.
        """
        # 1) Create your habitat renderer
        renderer = HabitatRenderer(
            dataset_name=habitat_dataset,
            fix_seed=fix_seed,
            output_dir=output_dir,
            black_ratio_thr=black_ratio_thr,
            enable_depth=enable_depth,
            height=height,
            width=width,
            traj_num=trajs_num_per_scene,  # number of trajectories per scene
            gpu_id=device_id,
            worker_id=worker_id,
            use_presaved_ep=use_presaved_ep,
        )
        meta_recoder = Recoder(env=renderer.habitat_env)            # 2

        data_iter = renderer.collect_data()  # or collect_data_batch(), etc.
        traj_dir = None
        for i, data in enumerate(data_iter):
            # If the renderer signals a 'cancel_current' scenario:
            if isinstance(data, dict):
                img_dir = data["waypoint_dir"]
                traj_dir = data["traj_dir"]
                save_path = os.path.join(
                    img_dir,
                    f"step-{data['step_id']}_type-{{}}.png"
                )
                # 1. Save images:
                data_imgs = {k: v for k, v in data.items() if k in ["rgb_eq", "depth_eq"]}
                data_imgs.update({'save_path': save_path})
                data_queues[i % (num_recoder-1)].put(data_imgs)

                # 2. record metadata for visualization
                # data_other = {k: v for k, v in data.items() if k not in ["rgb_eq", "depth_eq"]}
                # meta_recoder.update_trajectory(data_other, save_imgs=False, dir=save_path)

            elif data == 'cancel_current-scene_finished':
                # If you want to save the partially-complete trajectory
                if traj_dir:
                    meta_recoder.save_trajectory(output_dir=traj_dir)
                    # scene_folder = os.path.dirname(traj_dir)    # get the upper folder
                    # scene_folder = scene_folder.replace('03.02_hm3d-wp-missing', '01.26_indoor_dataset-wp', )
                    # meta_recoder.del_mapped_scene_folder(dir_path=scene_folder)
                meta_recoder.reset()
            elif data == 'cancel_current-waypoint_finished':
                continue

        # Put a sentinel to signal that we are done producing
        for q in data_queues:
            q.put("DONE")


    def record_consumer(recoder, data_queue, tag=""):
        """
        Consumes data from 'data_queue' and updates the given 'recoder'.
        The 'tag' can be used to distinguish recoder1 vs. recoder2 logs.
        """
        while True:
            data = data_queue.get()
            if data == "DONE":
                # Rendering has finished; exit the consumer loop.
                break
            # Save images:
            save_path = data['save_path']
            recoder.update_trajectory(data, save_imgs=True, dir=save_path)

        print(f"[Worker {worker_id}] Consumer {tag} done. Saving any last data if needed.")

    # Create and start the producer thread
    producer_thread = threading.Thread(target=render_producer)
    producer_thread.start()

    # Create and start the consumer threads, one per recoder
    consumer_threads = []
    for i, rec in enumerate(recoders):
        t = threading.Thread(target=record_consumer, args=(rec, data_queues[i], f"R{i}"))
        t.start()
        consumer_threads.append(t)

    # 4) Wait for the producer and all consumers to complete
    print(f"[Worker {worker_id}] Waiting for producer and consumers to finish ...")
    producer_thread.join()
    for t in consumer_threads:
        t.join()
    print(f"[Worker {worker_id}] Finished collecting & recording data on GPU {device_id}.")


if __name__ == "__main__":
    import argparse
    import os
    import multiprocessing as mp

    def multiprocess_init(args):
        """
        Spawns multiple worker processes. Each worker is fully responsible for
        rendering and recording, so no data is returned to the main process.
        """
        ctx = mp.get_context("spawn")

        # Figure out how many GPUs are visible
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            num_gpu = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        else:
            num_gpu = 1

        # Ensure num_processes is divisible by num_gpu
        assert args.num_processes % num_gpu == 0, (
            f"num_processes ({args.num_processes}) must be a multiple of number of GPUs ({num_gpu})."
        )
        process_per_gpu = args.num_processes // num_gpu

        # Example: if num_gpu=2 and num_processes=4 => [0,1,0,1]
        worker_gpus = list(range(num_gpu)) * process_per_gpu
        worker_ids = [0] * args.num_processes
        log_output_dir = os.path.join(args.output_dir, args.exp_id)
        os.makedirs(log_output_dir, exist_ok=True)

        # Spawn each worker
        processes = []
        for i in range(args.num_processes):
            p = ctx.Process(
                target=render_and_record_worker,
                args=(
                    args.dataset_name,
                    False,  # fix_seed
                    args.enable_depth,
                    args.width,
                    args.height,
                    args.black_ratio_threshold,
                    args.output_dir,
                    worker_gpus[i],           # device_id
                    worker_ids[i],            # worker_id
                    args.trajs_num_per_scene,
                    args.use_presaved_ep,
                    log_output_dir,
                ),
            )
            p.start()
            time.sleep(20)
            processes.append(p)

        # Optionally, join them if you want the main process to wait for completion:
        for p in processes:
            p.join()
        print("[Main] All worker processes completed.")


    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--output_dir", type=str,
                            default="/data/datasets/username/read+write/username1/igenex/datasets/igenex_data/debug/")
        parser.add_argument("--black_ratio_threshold", type=float, default=0.65,
                            help="The threshold to filter out images that are mostly black (not used).")

        parser.add_argument("--enable_depth", action="store_true", help="Whether to render and save depth images.")
        parser.add_argument("--height", type=int, default=1000, help="Height of the rendered images (panorama height)")
        parser.add_argument("--width", type=int, default=2000, help="Width of the rendered images. (panorama width)")
        parser.add_argument("--dataset_name", type=str, default="hm3d-train", choices=["hm3d-train", "hm3d-val", "mp3d-train", "mp3d-val"])
        parser.add_argument("--trajs_num_per_scene", type=int, default=1, help="Number of trajectories per scene, 1 means one loop across the whole scene.")

        parser.add_argument("--num_processes", type=int, default=8,
                            help="Number of GPU-render processes to spawn.")
        parser.add_argument("--use_presaved_ep", action="store_true",
                            help="Whether to use the presaved episodes for the dataset.")
        parser.add_argument("--exp_id", type=str,
                            help="The current experiment id for the logging directory.")
        return parser.parse_args()

    # Parse args and setup logging
    args = get_args()
    # if "igenex" not in args.output_dir:
    #     args.output_dir = os.path.join(
    #         "/data/datasets/username/read+write/username1/igenex/datasets/igenex_data",
    #         args.output_dir
    #     )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving data to {args.output_dir}")
    print("Args:", args)

    # Initialize the multiprocess workflow
    multiprocess_init(args)
