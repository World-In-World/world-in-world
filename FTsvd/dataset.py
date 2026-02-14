import os
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
import glob
import torch.nn.functional as F

from utils.dataset_utils import (
    glob_all_imgleaf_folders,
    gen_frame_idxs, get_actions,
    get_pixel_values,
    get_pixel_values_,
    get_sorted_frame_paths,
    check_metadata,
    action_reverse_convert,
    action_flip_convert,

)
from utils.util import get_generator, seed_worker
from utils.svd_utils import rotate_by_degrees
from data_filtering.filter_util import (
    get_all_trajs_voidratios,
    assign_sample_weights,
    glob_all_overlap_json,
)



def init_dataloader(args, enable_filter=True, only_straight_data=False):
    if enable_filter:
        val_dataset = WeightedDataset(
            base_folder=args.base_folder,
            num_past_obs=args.num_past_obs,
            num_samples=args.test_num if getattr(args, "test_num", None) else 100000,
            width=args.width,
            height=args.height,
            sample_frames=args.num_frames,
            fix_seed=getattr(args, "fix_seed", False),
            weighted_method=args.data_weighted_method,
            cutoff_thr=args.data_cutoff_thr,
            reverse_aug=getattr(args, "enable_reverse_aug", False),
        )
    else:
        if only_straight_data:
            val_dataset = DummyDataset_Straight(
                base_folder=args.base_folder,
                num_past_obs=args.num_past_obs,
                num_samples=args.test_num if getattr(args, "test_num", None) else 100000,
                width=args.width,
                height=args.height,
                sample_frames=args.num_frames,
                fix_seed=getattr(args, "fix_seed", False),
                reverse_aug=getattr(args, "enable_reverse_aug", False),
            )
        else:
            val_dataset = DummyDataset(
                base_folder=args.base_folder,
                num_past_obs=args.num_past_obs,
                num_samples=args.test_num if getattr(args, "test_num", None) else 100000,
                width=args.width,
                height=args.height,
                sample_frames=args.num_frames,
                fix_seed=getattr(args, "fix_seed", False),
                reverse_aug=getattr(args, "enable_reverse_aug", False),
            )

    sampler = RandomSampler(val_dataset, generator=get_generator())
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler,
        batch_size=args.per_gpu_batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=True,
    )
    return val_dataloader


class DummyDataset(Dataset):
    def __init__(
        self,
        base_folder: str, # or list of str
        num_past_obs: int,
        num_samples=100000,
        width=1024,
        height=576,
        sample_frames=25,
        fix_seed=False,
        reverse_aug=False,
    ):
        """
        Args:
            base_folder (list of str): Path to the dataset root.
            num_past_obs (int): Number of past frames.
            num_samples (int): Number of samples in the dataset.
            width (int): The width to which frames are resized.
            height (int): The height to which frames are resized.
            sample_frames (int): Number of frames to sample in each sequence.
            seed (int or None): Random seed for deterministic folder selection
                                and start index selection. If None, random.
        """
        # 1. Basic assignments
        self.num_samples   = num_samples
        self.base_folder   = base_folder
        self.width         = width
        self.height        = height
        self.sample_frames = sample_frames
        self.num_past_obs  = num_past_obs
        self.channels      = 3
        self.enable_aug   = reverse_aug

        # 2. Create an RNG if a seed is provided
        if fix_seed:
            # Create a local random number generator so we don't affect global random
            self.rng = random.Random(42)
        else:
            self.rng = None

        # 3. Compute stats and create the folder generator
        self.prepare_available_data(base_folder)

    def prepare_available_data(self, base_folder):
        metadata_paths_all = []
        for folder_p in self.base_folder:
            metadata_paths = glob_all_imgleaf_folders(folder_p)
            metadata_paths_all.extend(metadata_paths)
        # extract the folder paths from the metadata path:
        self.folders = [os.path.dirname(p) for p in metadata_paths_all]
        self.stats_leaf_folders = self.count_png_files(self.folders)
        self.folder_generator = self.select_folder_by_count() # Use the method that yields folder paths


    def count_png_files(self, folders):
        """
        Count the number of .png files in each folder under the given path
        and update counts recursively for parent folders.
        Returns:
            dict: stats_all (including counts from parent folders)
            dict: stats_leaf_folders (only for leaf folders with valid metadata and enough frames)
        """
        stats_leaf_folders = {}

        for folder_path in folders:
            # Count .png files that have 'type-rgb' in their name
            num_png = len(glob.glob(os.path.join(folder_path, '*rgb.png')))
            # Update the count for this folder in stats_all and stats_leaf_folders
            stats_leaf_folders[folder_path] = num_png

        # we do not need to check metadata because the path of avi folders are from the the metadata
        # leaf_folders = list(stats_leaf_folders.keys())
        # leaf_folders = check_metadata(leaf_folders)
        # stats_leaf_folders = {folder: stats_leaf_folders[folder] for folder in leaf_folders}
        # ...and by having enough frames (using the counts stored in stats_leaf_folders)
        stats_leaf_folders = self.check_num_frames(stats_leaf_folders)

        return stats_leaf_folders


    def check_num_frames(self, folder_dict: dict) -> dict:
        """
        Checks that each folder in folder_dict has at least self.sample_frames number of frames.
        Args:
            folder_dict (dict): Keys are folder paths and values are the number of .png files in that folder.
        Returns:
            dict: A filtered dictionary (sorted by folder path) that only contains folders with enough frames.
        """
        num_deleted_few_frames = 0
        filtered_dict = {}

        for folder_path, count in folder_dict.items():
            if count < self.sample_frames:
                print(f"<{folder_path}> has less than {self.sample_frames} frames. Skipping.")
                num_deleted_few_frames += 1
            else:
                filtered_dict[folder_path] = count

        print(f"Warning: Exclude <{num_deleted_few_frames}> folders from training with less than {self.sample_frames} frames.")
        return dict(sorted(filtered_dict.items()))


    def select_folder_by_count(self):
        """
        A generator that yields folders (randomly if no seed is set, or reproducibly
        if a seed is provided), with probability proportional to the number of .png files
        they contain.
        """
        folders = list(self.stats_leaf_folders.keys())
        # folders = list(constraint_folders.intersection(set(folders_)))
        folders = sorted(folders)
        weights = [self.stats_leaf_folders[f] for f in folders]

        # Check sum of weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("No .png files found in the directory or constraint set is empty.")

        while True:
            # If seed is set, use self.rng. Otherwise, use global random
            if self.rng is not None:
                yield self.rng.choices(folders, weights=weights, k=1)[0]
            else:
                yield random.choices(folders, weights=weights, k=1)[0]


    def __len__(self):
        return self.num_samples

    def select_frames(self):
        folder_path = next(self.folder_generator)

        # 2. Generate indexes and load frames (pass in self.rng if we have one)
        frame_paths, start_idx = gen_frame_idxs(
            folder_path,
            num_frames=self.sample_frames,
            rng=self.rng,
        )
        selected_idxs = list(range(start_idx, start_idx + self.sample_frames))
        return folder_path, frame_paths, start_idx, selected_idxs

    def __getitem__(self, idx):
        """
        Returns:
            dict: Containing:
                - 'pixel_values'    : Tensor of shape [sample_frames, C, H, W]
                - 'past_obs'        : Tensor of shape [num_past_obs, C, H, W]
                - 'actions'         : Tensor of shape [sample_frames] or similar
                - 'frame_paths'     : List of full paths for future frames
                - 'frame_paths_past': List of full paths for past frames
        """
        # 1. Get a folder via the generator
        folder_path, frame_paths, start_idx, selected_idxs = self.select_frames()

        # 3. Get actions
        sceneID = folder_path.split('/')[-3]
        trajID  = folder_path.split('/')[-2].split('-')[-1]
        waypointID = folder_path.split('/')[-1].split('-')[-1]
        actions = get_actions(sceneID, trajID, waypointID, folder_path, frame_idxs=selected_idxs)
        actions = torch.tensor(actions)

        # 4. Load future frames
        selected_frames = [frame_paths[i] for i in selected_idxs]
        pixel_values = get_pixel_values(
            folder_path, selected_frames,
            channels=self.channels, width=self.width, height=self.height,
        )
        frame_paths_all = [os.path.join(folder_path, f) for f in selected_frames]

        # 4.5. do reverse augmentation if needed
        if self.enable_aug:
            # 50% chance to reverse the frames:
            do_reverse = False
            if do_reverse:
                assert pixel_values.shape == (self.sample_frames, self.channels, self.height, self.width)
                pixel_values_re = rotate_by_degrees(pixel_values, 180)
                pixel_values = torch.flip(pixel_values_re, dims=[0])
                # save_video_from_tensor((pixel_values+1)/2, f"data_temp/reverse/retest_512_aug.mp4", action_ids=actions)
                actions = action_reverse_convert(actions)
                frame_paths_all = frame_paths_all[::-1]

            do_flip = random.choice([True, False])
            # Flip the future frames horizontally
            if do_flip:
                pixel_values = torch.flip(pixel_values, dims=[3])
                # If your actions are direction dependent (like steering angles), you may need to invert them:
                actions = action_flip_convert(actions)

        # 5. Generate & load past frames
        # selected_frames_idxs = list(range(start_idx - (self.num_past_obs - 1), start_idx + 1))
        # selected_frames_idxs = [i for i in selected_frames_idxs if i >= 0]
        # selected_frames_past = [frame_paths[i] for i in selected_frames_idxs]
        # past_obs = get_pixel_values(
        #     folder_path, selected_frames_past,
        #     channels=self.channels, width=self.width, height=self.height
        # )
        past_obs = pixel_values[0]

        return {
            'pixel_values': pixel_values,
            'past_obs': past_obs,
            'actions': actions,
            'frame_paths': frame_paths_all,
            # 'frame_paths_past': [os.path.join(folder_path, f) for f in selected_frames_past],
            'folder_path': folder_path,
            'start_idx': start_idx,
            'reverse_aug': do_reverse if self.enable_aug else False,
            'flip_aug': do_flip if self.enable_aug else False,
        }


class WeightedDataset(DummyDataset):
    def __init__(
        self,
        base_folder: str, # or list of str
        num_past_obs: int,
        num_samples=100000,
        width=1024,
        height=576,
        sample_frames=25,
        fix_seed=False,
        weighted_method='exponential',
        cutoff_thr=0.45,
        reverse_aug=False,
    ):
        self.weighted_method = weighted_method
        self.cutoff_thr = cutoff_thr

        super().__init__(
            base_folder=base_folder,
            num_past_obs=num_past_obs,
            num_samples=num_samples,
            width=width,
            height=height,
            sample_frames=sample_frames,
            fix_seed=fix_seed,
            reverse_aug=reverse_aug,
        )

    def prepare_available_data(self, base_folder):
        json_files_all = []
        for folder_p in self.base_folder:
            json_files = glob_all_overlap_json(folder_p, self.sample_frames)
            json_files_all.extend(json_files)

        paths = []
        for f in json_files_all:
            # get the last level folder path:
            paths.append(os.path.dirname(f))
        paths_ = check_metadata(paths)
        if len(paths_) != len(paths):
            raise ValueError("Some folders do not have metadata.json files, but they are have overlap json files.")

        all_trajs_voidratios = get_all_trajs_voidratios(json_files_all)

        self.traj_entries, self.sample_weights = assign_sample_weights(
            all_trajs_voidratios,
            method=self.weighted_method,
            cutoff=self.cutoff_thr,
            # alpha=2.0,
        )
        self.traj_generator = self.select_traj_entry()


    def select_traj_entry(self):
        """
        A generator that yields folders (randomly if no seed is set, or reproducibly
        if a seed is provided), with probability proportional to the number of .png files
        they contain.
        """
        while True:
            # If seed is set, use self.rng. Otherwise, use global random
            if self.rng is not None:
                yield self.rng.choices(self.traj_entries, weights=self.sample_weights, k=1)[0]
            else:
                yield random.choices(self.traj_entries, weights=self.sample_weights, k=1)[0]


    def select_frames(self):
        folder_path, start_step = next(self.traj_generator)
        start_idx = int(start_step.split('-')[-1])
        frame_paths = get_sorted_frame_paths(folder_path, self.sample_frames)
        selected_idxs = list(range(start_idx, start_idx + self.sample_frames))
        return folder_path, frame_paths, start_idx, selected_idxs


class DummyDataset_Straight(DummyDataset):
    def __init__(
        self,
        base_folder: str, # or list of str
        num_past_obs: int,
        num_samples=100000,
        width=1024,
        height=576,
        sample_frames=25,
        fix_seed=False,
        reverse_aug=False,
    ):
        super().__init__(
            base_folder=base_folder,
            num_past_obs=num_past_obs,
            num_samples=num_samples,
            width=width,
            height=height,
            sample_frames=sample_frames,
            fix_seed=fix_seed,
            reverse_aug=reverse_aug,
        )

    def __getitem__(self, idx):
        """
        Returns:
            dict: Containing:
                - 'pixel_values'    : Tensor of shape [sample_frames, C, H, W]
                - 'past_obs'        : Tensor of shape [num_past_obs, C, H, W]
                - 'actions'         : Tensor of shape [sample_frames] or similar
                - 'frame_paths'     : List of full paths for future frames
                - 'frame_paths_past': List of full paths for past frames
        """
        # if actions [1:] is not all 1, then resample the and start_idx:
        while True:
            # 1. Get a folder via the generator
            folder_path, frame_paths, start_idx, selected_idxs = self.select_frames()
            # 3. Get actions
            sceneID = folder_path.split('/')[-3]
            trajID  = folder_path.split('/')[-2].split('-')[-1]
            waypointID = folder_path.split('/')[-1].split('-')[-1]
            actions = get_actions(sceneID, trajID, waypointID, folder_path, frame_idxs=selected_idxs)
            actions = torch.tensor(actions)
            if all(actions[1:] == 1):
                break

        # 4. Load future frames
        selected_frames = [frame_paths[i] for i in selected_idxs]
        pixel_values = get_pixel_values(
            folder_path, selected_frames,
            channels=self.channels, width=self.width, height=self.height,
        )
        frame_paths_all = [os.path.join(folder_path, f) for f in selected_frames]

        # 4.5. do reverse augmentation if needed
        if self.enable_aug:
            # 50% chance to reverse the frames:
            do_reverse = random.choice([True, False])
            if do_reverse:
                assert pixel_values.shape == (self.sample_frames, self.channels, self.height, self.width)
                pixel_values_re = rotate_by_degrees(pixel_values, 180)
                pixel_values = torch.flip(pixel_values_re, dims=[0])
                # save_video_from_tensor((pixel_values+1)/2, f"data_temp/reverse/retest_512_aug.mp4", action_ids=actions)
                actions = action_reverse_convert(actions)
                frame_paths_all = frame_paths_all[::-1]

            do_flip = random.choice([True, False])
            # Flip the future frames horizontally
            if do_flip:
                pixel_values = torch.flip(pixel_values, dims=[3])
                # If your actions are direction dependent (like steering angles), you may need to invert them:
                actions = action_flip_convert(actions)

        # 5. Generate & load past frames
        # selected_frames_idxs = list(range(start_idx - (self.num_past_obs - 1), start_idx + 1))
        # selected_frames_idxs = [i for i in selected_frames_idxs if i >= 0]
        # selected_frames_past = [frame_paths[i] for i in selected_frames_idxs]
        # past_obs = get_pixel_values(
        #     folder_path, selected_frames_past,
        #     channels=self.channels, width=self.width, height=self.height
        # )
        past_obs = pixel_values[0]

        return {
            'pixel_values': pixel_values,
            'past_obs': past_obs,
            'actions': actions,
            'frame_paths': frame_paths_all,
            # 'frame_paths_past': [os.path.join(folder_path, f) for f in selected_frames_past],
            'folder_path': folder_path,
            'start_idx': start_idx,
            'reverse_aug': do_reverse if self.enable_aug else False,
            'flip_aug': do_flip if self.enable_aug else False,
        }
