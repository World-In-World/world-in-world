
import numpy as np
import torch
import os
from PIL import Image
import psutil
import subprocess
import math, random


def count_parameters(module):
    total     = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    pct       = 100.0 * trainable / total if total > 0 else 0.0

    print(f"Total parameters     : {total:,d}  ({total/1e9:.2f} B)")
    print(f"Trainable parameters : {trainable:,d}  ({trainable/1e9:.2f} B)")
    print(f"Fraction trainable   : {pct:.2f}%")
    return total, trainable


def is_empty(var):
    """
    Check if variable is empty, explicitly handling None, common containers,
    NumPy arrays, PyTorch tensors, and avoiding accidental handling of scalars/booleans.
    """
    if var is None:
        return True

    # Explicitly handle NumPy arrays
    if 'numpy' in str(type(var)):
        return var.size == 0

    # Explicitly handle PyTorch tensors
    if 'torch' in str(type(var)):
        return var.numel() == 0

    # Container types that support len()
    try:
        return len(var) == 0
    except TypeError:
        # Non-container, scalar, numeric, bool types
        return False  # Numbers, False, and scalars are explicitly considered not empty



def gen_dataset_seq(num_process, train_datasets, gpu_num, curr_process_idx, base_seed=42):
    """
    Generate a list of dataset assignments with length equal to `num_process` such that:
      1. Each dataset is assigned a number of workers proportional to its number of scenes.
      2. The rounding is done in a greedy way to ensure that the total number of worker assignments equals num_process.
      3. The overall assignment list is then deterministically shuffled (using the base_seed).
      4. The function returns the sub-list for the current process (just like gen_GPU_ids_seq does).
    Args:
        num_process (int): Total number of worker processes.
        train_datasets (str): String with datasets in the format 'hm3d-train+mp3d-train', etc.
        curr_process_idx (int): Distributed rank (index) for the current process.
        base_seed (int, optional): Base seed for deterministic shuffling. Defaults to 42.
    Returns:
        list[str]: A list with the dataset assignment for the current process.
                   (Each process gets exactly one dataset from the overall assignment list.)
    """
    # Parse the input: expected format 'hm3d-train+mp3d-train'
    d_scene_num = parser_datasets_info(train_datasets)

    # Total number of scenes
    scene_sum = sum(d_scene_num.values())
    # Compute the ideal (float) worker allocation per dataset
    d_worker_frac = {k: (v / scene_sum) * num_process for k, v in d_scene_num.items()}

    # First, assign the floor value to each dataset.
    d_worker_int = {k: math.floor(v) for k, v in d_worker_frac.items()}
    current_total = sum(d_worker_int.values())
    remainder = num_process - current_total

    # Prepare a sorted list of datasets by the size of their fractional remainder.
    frac_list = [(k, d_worker_frac[k] - d_worker_int[k]) for k in d_worker_frac]
    frac_list.sort(key=lambda x: x[1], reverse=True)

    # Distribute the remaining workers using a greedy (largest remainder) approach.
    for i in range(remainder):
        dataset_to_increment = frac_list[i % len(frac_list)][0]
        d_worker_int[dataset_to_increment] += 1

    # Build the global assignment list based on the computed worker numbers.
    # Since we guarantee sum(d_worker_int.values()) == num_process, each worker will receive one assignment.
    dataset_assignments = []
    for dataset, worker_count in d_worker_int.items():
        dataset_assignments.extend([dataset] * worker_count)

    # Deterministically shuffle the global assignment list.
    num_per_gpu = num_process // gpu_num
    rng = random.Random(base_seed)
    rng.shuffle(dataset_assignments)

    return dataset_assignments[curr_process_idx*num_per_gpu : (curr_process_idx+1)*num_per_gpu]


def parser_datasets_info(train_datasets):
    d_list = train_datasets.split('+')

    # For each dataset, assign a number of scenes.
    d_scene_num = {}
    for d in d_list:
        # d is expected in the format 'datasetName-split'
        d_name, split = d.split('-')
        if d_name == 'hm3d':
            if split == 'train':
                d_scene_num[d] = 800
            elif split == 'val':
                d_scene_num[d] = 100
        elif d_name == 'mp3d':
            if split == 'train':
                d_scene_num[d] = 61
            elif split == 'val':
                d_scene_num[d] = 20
        elif d_name == 'hssd':
            if split == 'train':
                d_scene_num[d] = 122
            elif split == 'val':
                d_scene_num[d] = 42
        else:
            raise ValueError(f"Unknown dataset name: {d_name}")
    return d_scene_num


def gen_GPU_ids_seq(num_process, gpu_num, curr_process_idx, base_seed=42):
    """
    Generate a list of GPU device IDs of length `num_process` such that:
      1. The workload is balanced across `gpu_num` GPUs
         (since num_process is guaranteed to be a multiple of gpu_num).
      2. The order is randomized but *deterministic* for a given seed+rank.
      3. `curr_process_idx` (the distributed rank) is included in the seed
         so each rank can produce a stable, but rank-specific shuffle if desired.
    Args:
        num_process (int): Number of worker processes to spawn.
        gpu_num (int): Number of GPUs available. Must divide `num_process`.
        curr_process_idx (int, optional): Distributed rank for the current training process.
        base_seed (int, optional): Base random seed. Defaults to 42.
    Returns:
        list[int]: A shuffled list of GPU IDs of length `num_process`.
    """
    assert num_process % gpu_num == 0, (
        f"num_process={num_process} must be a multiple of gpu_num={gpu_num}!"
    )

    rng = random.Random(base_seed)

    # 3. Balanced distribution: each GPU gets exactly num_process // gpu_num
    num_per_gpu = num_process // gpu_num
    gpu_ids = []
    for g in range(gpu_num):
        gpu_ids.extend([g] * num_per_gpu)

    # 4. Shuffle to randomize the assignment sequence
    rng.shuffle(gpu_ids)
    return gpu_ids[curr_process_idx*num_per_gpu : (curr_process_idx+1)*num_per_gpu]


def print_slurm_job_id(logger=None):
    # Retrieve the SLURM_JOB_ID from environment variables
    slurm_job_id = os.environ.get('SLURM_JOB_ID', None)
    if slurm_job_id:
        if logger is not None:
            logger.info(f"Current SLURM Job ID: <{slurm_job_id}>")
        else:
            print(f"Current SLURM Job ID: <{slurm_job_id}>")
    else:
        if logger is not None:
            logger.info("SLURM_JOB_ID environment variable not found. Ensure the script is running within a Slurm job.")
        else:
            print("SLURM_JOB_ID environment variable not found. Ensure the script is running within a Slurm job.")

def print_nvidia_smi():
    try:
        # Run the nvidia-smi command
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check if the command was successful
        if result.returncode == 0:
            # Print the output of nvidia-smi
            print(result.stdout)
        else:
            # Print the error if the command failed
            print("Error running nvidia-smi:")
            print(result.stderr)
    except FileNotFoundError:
        print("nvidia-smi command not found. Ensure NVIDIA drivers are installed and nvidia-smi is in your PATH.")


def get_memory_info():
    mem = psutil.virtual_memory()
    total = mem.total / (1024 ** 3)       # Convert bytes to GB
    available = mem.available / (1024 ** 3)
    used = mem.used / (1024 ** 3)
    free = mem.free / (1024 ** 3)
    percent_used = mem.percent      #range [0, 100]

    return {
        "total_gb": total,
        "available_gb": available,
        "used_gb": used,
        "free_gb": free,
        "percent_used": percent_used
    }


def export_to_gif(frames, output_gif_path, fps, actions=None):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [Image.fromarray(frame) if isinstance(
        frame, np.ndarray) else frame for frame in frames]

    pil_frames[0].save(output_gif_path.replace('.mp4', '.gif'),
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=500,
                       loop=0)
    assert actions.dim() == 2 and actions.size(0) == 1
    actions = actions.squeeze(0)
    if isinstance(actions[0], torch.Tensor):
        actions = [action.item() for action in actions]
    print(f"Actions for the gif <{output_gif_path}> is: <{actions}>")


def get_generator(seed=1):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def become_deterministic(seed=0, except_cuda=False):
    # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    if except_cuda:
        return
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Seed for cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # can set to false
    # torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # torch.autograd.set_detect_anomaly(True)

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)