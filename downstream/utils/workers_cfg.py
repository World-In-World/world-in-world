from downstream.utils.worker import init_workers, close_all
import os
import socket

if "Host1" in socket.gethostname():
    PLATFORM = "Host1"
elif "Host2" in socket.gethostname():
    PLATFORM = "Host2"
else:
    PLATFORM = "Host3"
print(f"hostname is {socket.gethostname()}, Current Platform: {PLATFORM}")


# default output size (not generation size) for video generation workers
OUT_WIDTH = 480
OUT_HEIGHT = 480
# for wan22:
# OUT_WIDTH = 640
# OUT_HEIGHT = 302

COMMON_ARGS = dict(
    Host1={
        "genex_worker": [
            "/data/username1/software/miniconda3/envs/instructnav1/bin/python",
            "eval_inference.py",
            "--unet_path=outputs/07.22_0.1D_uni_full_noFPSid_3acm_aug_NoiseA/seed_1_0722_213129/checkpoint-6000",
            "--svd_path=checkpoints/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e",
            "--action_input_channel=14",
            "--task_type=navigation",
        ],
        "cosmos_worker": [
            "/data/username2/conda/miniconda3/envs/cosmos-predict2/bin/python",
            "downstream/api_models/cosmos_model.py",
        ],
        "FTcosmos_worker": [
            "/data/username2/conda/miniconda3/envs/cosmos-predict2/bin/python",
            "downstream/api_models/cosmos_model.py",
            "--width=1024",
            "--height=576",
            "--num_frames=14",
            "--ft_dir=outputs/cosmos/converted_ft_cosmos2_2b_video2world_navigation",
        ],
        "FTltx_worker": [
            "/data/username2/conda/miniconda3/envs/LTXvideo/bin/python",
            "downstream/api_models/ltx_model.py",
            "--width=1024",
            "--height=576",
            "--num_frames=17",
            "--ft_dir=outputs/test_ltx_pipeline",
            "--ckpt_path=outputs/test_ltx_pipeline",
        ],
        "ltx_worker": [
            "/data/username2/conda/miniconda3/envs/LTXvideo/bin/python",
             "downstream/api_models/ltx_model.py",
        ],
        "sam2_worker": [
            "/data/username1/software/anaconda/envs/sam2/bin/python",
            "downstream/detection/sam2_model.py",
            "--ckpt_path=/data/username1/visual_navigation/username6/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
            "--cfg_path=/home/username8/igenex_code/configs/sam2.1/sam2.1_hiera_b+.yaml",
        ],
        "grounding_sam2_worker": [
            "/data/username1/software/anaconda/envs/sam2/bin/python",
            "downstream/detection/grounding_sam2_model.py",
            "--ckpt_path=/data/username1/visual_navigation/username6/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
            "--cfg_path=/home/username8/igenex_code/configs/sam2.1/sam2.1_hiera_b+.yaml",
        ],
        "genex_manip_worker": [
            "/data/username1/software/miniconda3/envs/instructnav1/bin/python",
            "eval_inference.py",
            "--unet_path=outputs/07.13_manip3D_14f_10-50_absAct_noExBound-35frame/checkpoint-26000/unet",     # 08.19: new 5 tasks on RLbench
            "--svd_path=outputs/checkpoints/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e",
            "--action_input_channel=10",
            "--task_type=manipulation",
            "--width=448",
            "--height=448",
        ],
        "gen4tur_worker": [
            "/data/username2/conda/miniconda3/envs/cosmos-predict2/bin/python",
            "downstream/api_models/runway_model.py",
            "--width=480",
            "--height=480",
            "--saaspit_key", "WnrT3xAxuwFo1b9r9Vg58h78E1"
        ]
    },
    Host2={
        "hunyuan_worker": [
            "/data/username3/anaconda3/envs/hunyuan/bin/python",
            "downstream/api_models/hunyuan_model.py",
        ],
        "ltx_worker": [
            "/data/username3/anaconda3/envs/LTXvideo/bin/python",
            "downstream/api_models/ltx_model.py",
        ],
        "FTltx_worker": [
            "/data/username3/anaconda3/envs/LTXvideo/bin/python",
            "downstream/api_models/ltx_model.py",
            "--width=1024",
            "--height=576",
            "--num_frames=17",
            "--ft_dir=outputs/test_ltx_pipeline",
            "--ckpt_path=outputs/test_ltx_pipeline",
        ],
        "wan21_worker": [
            "/data/username2/conda/miniconda3/envs/wan/bin/python",
            "downstream/api_models/wan_model.py",
        ],
        "cosmos_worker": [
            "/data/username2/conda/miniconda3/envs/cosmos-predict2/bin/python",
            "downstream/api_models/cosmos_model.py",
        ],
        "FTcosmos_worker": [
            "/data/username2/conda/miniconda3/envs/cosmos-predict2/bin/python",
            "downstream/api_models/cosmos_model.py",
            "--width=1024",
            "--height=576",
            "--num_frames=14",
            "--ft_dir=outputs/cosmos/converted_ft_cosmos2_2b_video2world_navigation",
        ],
        "genex_worker": [
            "/data/username2/conda/miniconda3/envs/habitat/bin/python",
            "eval_inference.py",
            "--unet_path=outputs/07.22_0.1D_uni_full_noFPSid_3acm_aug_NoiseA/seed_1_0722_213129/checkpoint-6000",
            "--svd_path=outputs/checkpoints/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e",
            "--action_input_channel=14",
            "--task_type=navigation",
        ],
    },
    Host3={
        "genex_worker": [
            "/weka/scratch/username4/username5/anaconda3/envs/habitat/bin/python",
            "eval_inference.py",
            "--unet_path=outputs/07.22_0.1D_uni_full_noFPSid_3acm_aug_NoiseA/seed_1_0722_213129/checkpoint-6000",
            "--svd_path=checkpoints/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e",
            "--action_input_channel=14",
            "--task_type=navigation",
        ],
        "svd_worker": [
            "/weka/scratch/username4/username5/anaconda3/envs/cosmos-predict2/bin/python",
            "downstream/api_models/svd_model.py",
        ],
        "cosmos_worker": [
            "/weka/scratch/username4/username5/anaconda3/envs/cosmos-predict2/bin/python",
            "downstream/api_models/cosmos_model.py",
        ],
        "FTcosmos_worker": [
            "/weka/scratch/username4/username5/anaconda3/envs/cosmos-predict2/bin/python",
            "downstream/api_models/cosmos_model.py",
            "--width=1024",
            "--height=576",
            "--num_frames=14",
            "--ft_dir=outputs/cosmos/converted_ft_cosmos2_2b_video2world_navigation",
        ],
        "genex_manip_worker": [
            "/scratch/username7/username8/anaconda3/envs/habitat/bin/python",
            "eval_inference.py",
            "--unet_path=outputs/07.03_manip3D_14f_10-50_absAct_noExBound/seed_1/checkpoint-22000/unet",
            "--svd_path=checkpoints/models--stabilityai--stable-video-diffusion-img2vid/snapshots/9cf024d5bfa8f56622af86c884f26a52f6676f2e",
            "--action_input_channel=10",
            "--task_type=manipulation",
        ],
        "se3ds_worker": [
            "LD_LIBRARY_PATH='/weka/scratch/username7/username6/software/miniconda3/envs/se3ds/lib:$LD_LIBRARY_PATH'",
            "/scratch/username7/username6/software/miniconda3/envs/se3ds/bin/python",
            "downstream/api_models/se3ds_model.py",
            "--ckpt_path=/weka/scratch/username7/username6/project_code/se3ds/se3ds/data/se3ds_ckpt",
        ],
        "pathdreamer_worker": [
            "LD_LIBRARY_PATH='/weka/scratch/username7/username6/software/miniconda3/envs/se3ds/lib:$LD_LIBRARY_PATH'",
            "/scratch/username7/username6/software/miniconda3/envs/se3ds/bin/python",
            "downstream/api_models/pathdreamer_model.py",
            "--ckpt_path=/weka/scratch/username7/username6/project_code/pathdreamer/pathdreamer/data/ckpt",
        ],
        "nwm_worker": [
            "/scratch/username7/username6/software/miniconda3/envs/nwm/bin/python",
            "downstream/api_models/nwm_model.py",
            "--ckpt_path=checkpoints/models--facebook--nwm/snapshots/bd92e0cdf7f3bc64cb2009fcd8882e96195e6150/0200000.pth.tar",
        ],
        "hunyuan_worker": [
            "/scratch/username7/username6/software/miniconda3/envs/hunyuan/bin/python",
            "downstream/api_models/hunyuan_model.py",
        ],
        "ltx_worker": [
            "/home/username6/.conda/envs/LTXvideo/bin/python",
            "downstream/api_models/ltx_model.py",
        ],
        "FTltx_worker": [
            "/home/username6/.conda/envs/LTXvideo/bin/python",
            "downstream/api_models/ltx_model.py",
            "--width=1024",
            "--height=576",
            "--num_frames=17",
            "--ft_dir=outputs/test_ltx_pipeline",
            "--ckpt_path=outputs/test_ltx_pipeline",
        ],
        "wan21_worker": [
            "/scratch/username7/username8/anaconda3/envs/wan/bin/python",
            "downstream/api_models/wan_model.py",
        ],
        "wan22_worker": [
            "/weka/scratch/username4/username5/anaconda3/envs/wan22/bin/python",
            "downstream/api_models/wan22_ti2v_model.py",
            "--width=1280",
            "--height=704",
            "--out_width=1280",
            "--out_height=704",
        ],
        "FTwan21_worker": [
            "/weka/scratch/username4/username9/scratch/username4/username9/envs/wan21/bin/python",
            "downstream/api_models/wan_model_diffsynth.py",
            "--model_id=Wan-AI/Wan2.1-I2V-14B-720P",
            "--width=1024",
            "--height=576",
            "--ft_method=lora",
            "--lora_path=/home/username5/scratchusername4/username9/shared/username6/models/wan_models/navigation_medium/Wan2.1-I2V-14B-720P_lora_navigation/epoch-0.safetensors",
        ],
        "FTwan22_worker": [
            "/weka/scratch/username4/username9/scratch/username4/username9/envs/wan21/bin/python",
            "downstream/api_models/wan_model_diffsynth.py",
            "--model_id=Wan-AI/Wan2.2-TI2V-5B",
            "--width=1024",
            "--height=576",
            "--ft_method=lora",
            "--lora_path=/home/username5/scratchusername4/username9/shared/username6/models/wan_models/navigation_medium/Wan2.2-TI2V-5B_lora_navigation/epoch-0.safetensors",
        ],
        "sam2_worker": [
            "/scratch/username7/username8/anaconda3/envs/sam2/bin/python",
            "downstream/detection/sam2_model.py",
            "--ckpt_path=/weka/scratch/username7/username6/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
            "--cfg_path=/weka/scratch/username7/username6/sam2/checkpoints/sam2.1_hiera_b+.yaml",
        ],
        "grounding_sam2_worker": [
            "/scratch/username7/username8/anaconda3/envs/sam2/bin/python",
            "downstream/detection/grounding_sam2_model.py",
            "--ckpt_path=/weka/scratch/username7/username6/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
            "--cfg_path=/weka/scratch/username7/username6/sam2/checkpoints/sam2.1_hiera_b+.yaml",
        ]
    }
)


def set_cuda_devices(num_workers, devices):
    """
    Set the CUDA devices for the workers.
    Args:
        num_workers (int): Number of workers.
        devices (str or None): Comma-separated string of device indices or None to use all available devices.
    Returns:
        list: List of device indices for the worker, e.g., [0, 1, 2].
    """
    if devices is None:
        devices = os.getenv("CUDA_VISIBLE_DEVICES")
        devices = devices.split(",")
        devices = [int(i) for i in devices]
        num_devices = len(devices)
        # available_devices = [f"{(num_devices-1) - i%num_devices}" for i in range(num_workers)]
        available_devices = [devices[i % num_devices] for i in range(num_workers)]
    else:
        # available_devices = devices.split(",")
        available_devices = [int(i) for i in devices.split(",")]
    return available_devices


def get_genex_workers_cmd(num_workers=3, devices=None, add_args={}):
    available_devices = set_cuda_devices(num_workers, devices)

    common_args = COMMON_ARGS[PLATFORM]["genex_worker"]
    script_args = [
        "--num_past_obs=1",
        "--action_strategy=micro_cond",
    ]
    added = [
        f"--{k}={v}" for k, v in add_args.items()
    ]
    worker_args = {}
    for i in range(num_workers):
        device = available_devices[i]
        worker_args[i] = [f'CUDA_VISIBLE_DEVICES="{device}"'] + common_args + script_args + added

    return worker_args

def get_genex_workers_manip_cmd(num_workers=3, devices=None, add_args={}):
    available_devices = set_cuda_devices(num_workers, devices)

    common_args = COMMON_ARGS[PLATFORM]["genex_manip_worker"]
    script_args = [
        "--num_past_obs=1",
        "--action_strategy=micro_cond",
    ]
    added = [
        f"--{k}={v}" for k, v in add_args.items()
    ]
    worker_args = {}
    for i in range(num_workers):
        device = available_devices[i]
        worker_args[i] = [f'CUDA_VISIBLE_DEVICES="{device}"'] + common_args + script_args + added

    return worker_args


def get_sam2_workers_cmd(num_workers=2, devices=None, add_args={}):
    available_devices = set_cuda_devices(num_workers, devices)

    common_args = COMMON_ARGS[PLATFORM]["sam2_worker"]
    script_args = []
    added = [
        f"--{k}={v}" for k, v in add_args.items()
    ]
    worker_args = {}
    for i in range(num_workers):
        device = available_devices[i]
        worker_args[i] = [f'CUDA_VISIBLE_DEVICES="{device}"'] + common_args + script_args + added

    return worker_args

def get_gd_sam2_workers_cmd(num_workers=2, devices=None, add_args={}):
    available_devices = set_cuda_devices(num_workers, devices)

    common_args = COMMON_ARGS[PLATFORM]["grounding_sam2_worker"]
    script_args = []
    added = [
        f"--{k}={v}" for k, v in add_args.items()
    ]
    worker_args = {}
    for i in range(num_workers):
        device = available_devices[i]
        worker_args[i] = [f'CUDA_VISIBLE_DEVICES="{device}"'] + common_args + script_args + added

    return worker_args

def get_worldmodel_workers_cmd(model_type, num_workers=2, devices=None, add_args={}):
    available_devices = set_cuda_devices(num_workers, devices)

    common_args = COMMON_ARGS[PLATFORM][model_type]
    script_args = []
    added = [
        f"--{k}={v}" for k, v in add_args.items()
    ]
    worker_args = {}
    for i in range(num_workers):
        device = available_devices[i]
        worker_args[i] = [f'CUDA_VISIBLE_DEVICES="{device}"'] + common_args + script_args + added

    return worker_args

