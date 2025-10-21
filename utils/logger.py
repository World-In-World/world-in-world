import os
import sys
import time
import os.path as osp
from pathlib import Path
import random
import numpy as np
import torch
import warnings
import logging
import socket, re, subprocess

original_fn = warnings.showwarning
original_stdout = sys.stdout
original_stderr = sys.stderr

__all__ = ["setup_logger", "log_args_and_env", "become_deterministic"]


class Logger:
    """Write console output to external text file and logging module.
    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os.path as osp
       >>> save_dir = 'output/experiment-1'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(save_dir, log_name))
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            Path(osp.dirname(fpath)).mkdir(parents=True, exist_ok=True)
            self.file = open(fpath, "w")

    def write(self, msg):
        self.console.write(msg)  # Write to console
        if self.file is not None and msg != "\n":
            logging.info(msg.strip())  # Forward stdout to logging as info
            # self.file.write(msg)  # Write to file

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # self.console.close()
        if self.file is not None:
            self.file.close()


def setup_logger(output=None):
    if output is None:
        output = f"./output/for_DEBUG1"

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = osp.join(output, "log.log")

    os.makedirs(osp.dirname(fpath), exist_ok=True)
    if osp.exists(fpath):
        # Ensure the existing log file is not overwritten
        postfix = fpath.split('.')[-1]  # File extension
        fpath = fpath.replace(f".{postfix}", f"{time.strftime('-%m-%d-%H-%M-%S')}.{postfix}")

    # Clear all existing handlers to avoid duplicate logs
    while logging.root.handlers:
        logging.root.removeHandler(logging.root.handlers[0])

    # Temporarily restore original stdout/stderr before closing the old Logger
    if isinstance(sys.stdout, Logger):
        sys.stdout.close()
        sys.stdout = original_stdout
    if isinstance(sys.stderr, Logger):
        sys.stderr.close()
        sys.stderr = original_stderr

    # Configure logging for warnings and external packages
    setup_warning_and_package_logging(fpath)

    # Redirect stdout and stderr to the Logger
    sys.stdout = Logger(fpath)
    sys.stderr = sys.stdout


def setup_warning_and_package_logging(fpath):
    # Configure logging to capture only warnings and above
    logging.basicConfig(
        filename=fpath,
        level=logging.INFO,  # Captures info, warnings, errors, and critical logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Redirect Python warnings to logging with file and line info, and print to console
    def log_warning_handler(message, category, filename, lineno, file=None, line=None):
        # Print warning to console
        original_fn(message, category, filename, lineno, file, line)
        logging.warning(message)
        # Log warning to file
        # logger = logging.getLogger("package_warning")
        # logger.warning("Category: %s",message,)

    warnings.showwarning = log_warning_handler


def log_args_and_env(cfg):
    print('************')
    print('** Config **')
    print('************')
    print(cfg)
    # print('Collecting env info ...')
    # print('** System info **\n{}\n'.format(collect_env_info()))


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


def log_worker_identity(device_str: str) -> None:
    """
    Log where this worker is running and which GPU it uses.
    - device_str: e.g., "cuda:0"
    Prints/logs: hostname, logical CUDA idx, physical GPU idx (via CUDA_VISIBLE_DEVICES),
                 device name, UUID, PCI bus id, and SLURM info if present.
    """
    host_short = socket.gethostname()
    try:
        host_fqdn = subprocess.check_output(["hostname", "-f"], text=True).strip()
    except Exception:
        host_fqdn = host_short

    slurm_job = os.environ.get("SLURM_JOB_ID", "")
    slurm_nodes = os.environ.get("SLURM_NODELIST", "")

    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()  # e.g., "3" or "3,7"
    logical_idx = None
    physical_idx = None
    device_name = "cpu"
    uuid = ""
    bus_id = ""

    try:
        if torch.cuda.is_available() and device_str.startswith("cuda"):
            dev = torch.device(device_str)
            torch.cuda.set_device(dev)  # ensure current device aligns with args.device
            logical_idx = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(logical_idx)

            # Map logical -> physical via CUDA_VISIBLE_DEVICES if it is set.
            if cuda_vis:
                mapping = [int(x) for x in re.findall(r"\d+", cuda_vis)]
                if logical_idx is not None and logical_idx < len(mapping):
                    physical_idx = mapping[logical_idx]

            # Ask nvidia-smi for UUID and PCI bus id of the physical GPU (if known).
            if physical_idx is not None:
                try:
                    line = subprocess.check_output(
                        [
                            "bash", "-lc",
                            "nvidia-smi --query-gpu=index,uuid,pci.bus_id,name "
                            "--format=csv,noheader,nounits"
                        ],
                        text=True
                    ).strip()
                    # Find the line for physical_idx
                    for row in line.splitlines():
                        cols = [c.strip() for c in row.split(",")]
                        if cols and cols[0] == str(physical_idx):
                            # cols: [index, uuid, pci.bus_id, name]
                            uuid, bus_id = cols[1], cols[2]
                            break
                except Exception:
                    pass
    except Exception:
        pass

    msg = (
        f"[worker_identity] host={host_fqdn} (short={host_short}) "
        f"SLURM_JOB_ID={slurm_job} SLURM_NODELIST={slurm_nodes} "
        f"CUDA_VISIBLE_DEVICES='{cuda_vis or '(unset)'}' device_arg='{device_str}' "
        f"logical_cuda_idx={logical_idx} physical_gpu_idx={physical_idx} "
        f"name='{device_name}' uuid='{uuid}' bus_id='{bus_id}'"
    )
    logging.info(msg)
    print(msg)
