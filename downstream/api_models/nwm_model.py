# from distributed import init_distributed
# from datasets_nwm import EvalDataset
import sys
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import yaml
import argparse
import os
import numpy as np

from downstream.api_models.nwm.diffusion import create_diffusion
from diffusers.models import AutoencoderKL

from downstream.api_models.nwm import misc
from downstream.api_models.nwm.misc import angle_difference, get_data_path, get_delta_np, normalize_data, to_local_coords
import torch.distributed as dist

from downstream.api_models.nwm.diffusion.cdit import CDiT_models
from PIL import Image
import math
from downstream.utils.saver import save_predict
import os.path as osp
from utils.logger import setup_logger
from downstream.utils.worker_manager import (
    worker_main,
)
from downstream.api_models import process_output_dict, parse_input_data, prepare_image_list
from downstream.api_models import convert_actions_from_id_to_str
from tqdm import tqdm
import socket



def init_distributed(port=37124, rank_and_world_size=(None, None)):
    rank, world_size = rank_and_world_size
    # test if the port is available, if not +1 until it is available
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                break
        except OSError:
            port += 1
            print(f"Port {port} is already in use, trying next port...")
            continue

    dist_url='env://'
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(port))
    print("Using port", os.environ['MASTER_PORT'])

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            gpu = int(os.environ["LOCAL_RANK"])
        except Exception:
            print('torchrun env vars not sets')

    elif "SLURM_PROCID" in os.environ:
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            gpu = rank % torch.cuda.device_count()
            if 'HOSTNAME' in os.environ:
                os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
            else:
                os.environ['MASTER_ADDR'] = '127.0.0.1'
        except Exception:
            print('SLURM vars not set')

    else:
        rank = 0
        world_size = 1
        gpu = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'

    torch.cuda.set_device(gpu)

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        init_method=dist_url
    )

    # setup_for_distributed(rank == 0)
    return world_size, rank, gpu, True


class NavigationWM:
    def __init__(self, args, config, device):
        self.args = args
        self.device = device
        self.latent_size = config["image_size"] // 8
        # self.context_size = config["context_size"]
        self.context_size = args.context_size
        self.num_timesteps = int(config.get("num_diffusion_steps", 250))

        # Load model
        model = CDiT_models[config['model']](
            context_size=self.context_size,
            input_size=self.latent_size,
            in_channels=4,
        )

        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt["ema"], strict=True)

        model.eval()
        model.to(device)
        model = torch.compile(model)

        # Wrap in DDP if needed
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model, device_ids=[device], find_unused_parameters=False
        # )

        # Diffusion and VAE
        diffusion = create_diffusion(str(self.num_timesteps))
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)

        self.model = model
        self.diffusion = diffusion
        self.vae = vae

    @torch.no_grad()
    def inference(self, curr_obs, curr_delta, num_timesteps, num_goals=1, rel_t=None, progress=False):
        model, diffusion, vae = self.model, self.diffusion, self.vae
        x = curr_obs.to(self.device)
        y = curr_delta.to(self.device)

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            B, T = x.shape[:2]

            if rel_t is None:
                rel_t = torch.ones(B, device=self.device) * (1. / 128.) * num_timesteps

            x_flat = x.flatten(0, 1)
            z = vae.encode(x_flat).latent_dist.sample().mul_(0.18215)
            z = z.unflatten(0, (B, T))

            x_cond = z[:, :self.context_size]
            x_cond = x_cond.unsqueeze(1).expand(B, num_goals, self.context_size, *z.shape[2:]) # todo
            x_cond = x_cond.flatten(0, 1)

            z_noise = torch.randn(B * num_goals, 4, self.latent_size, self.latent_size, device=self.device)

            y = y.flatten(0, 1)
            model_kwargs = dict(y=y, x_cond=x_cond, rel_t=rel_t)

            samples = diffusion.p_sample_loop(
                model.forward, z_noise.shape, z_noise,
                clip_denoised=False, model_kwargs=model_kwargs,
                progress=progress, device=self.device
            )

            samples = vae.decode(samples / 0.18215).sample
            return torch.clip(samples, -1., 1.)

    def generate_rollout(self, output_dir, rollout_fps, idxs, obs_image, gt_image, delta, args):
        rollout_stride = args.input_fps // rollout_fps
        gt_image = gt_image[:, rollout_stride-1::rollout_stride]
        delta = delta.unflatten(1, (-1, rollout_stride)).sum(2)
        curr_obs = obs_image.clone().to(self.device)

        for i in range(gt_image.shape[1]):
            curr_delta = delta[:, i:i+1].to(self.device)
            if args.gt:
                x_pred = gt_image[:, i].clone().to(self.device)
            else:
                x_pred = self.inference(curr_obs, curr_delta, rollout_stride)
            curr_obs = torch.cat((curr_obs, x_pred.unsqueeze(1)), dim=1)[:, 1:]
            self.visualize_preds(output_dir, idxs, i, x_pred)

    def generate_time(self, idxs, obs_image, gt_output, delta, secs, args):
        '''
        idxs: (B, 1)
        obs_image: (B, 4, 3, H, W)
        gt_output: (B, 64, 3, H, W)
        delta: (B, 64, 3)
        secs: (5,)
        num_cond: 4
        '''
        assert delta.shape[1] == 14, f"delta.shape[1] ({delta.shape}) is not 14"
        eval_timesteps = [1] + [4*i for i in range(1, delta.shape[1])]
        # eval_timesteps = list(range(1, delta.shape[1] + 1))  # [1, 2, ..., 14]
        obs_image = obs_image.repeat(1, 4, 1, 1, 1) # (B, 4, 3, H, W)
        B = obs_image.shape[0]
        all_preds = []
        # import ipdb;ipdb.set_trace()

        for i, timestep in tqdm(enumerate(eval_timesteps), total=len(eval_timesteps)):
            curr_delta = delta[:, :i+1].sum(dim=1, keepdim=True)
            if args.gt:
                x_pred = gt_output[:, timestep-1].clone().to(self.device)
            else:
                x_pred = self.inference(obs_image, curr_delta, timestep) # (B, C, H, W)
            all_preds.append(x_pred)  # List of (B, C, H, W)
            # self.visualize_preds(output_dir, idxs, sec, x_pred)

        video_tensors = torch.stack(all_preds, dim=1) # (B, T, C, H, W)

        return video_tensors

    def generate_from_genframe(self, idxs, obs_image, gt_output, delta, secs, args):
        '''
        idxs: (B, 1)
        obs_image: (B, 4, 3, H, W)
        gt_output: (B, 64, 3, H, W)
        delta: (B, 64, 3)
        secs: (5,)
        num_cond: 4
        '''
        assert delta.shape[1] == 14, f"delta.shape[1] ({delta.shape}) is not 14"
        eval_timesteps = [1] + [4*i for i in range(1, delta.shape[1])]
        # eval_timesteps = list(range(1, delta.shape[1] + 1))  # [1, 2, ..., 14]
        obs_image = obs_image.repeat(1, 4, 1, 1, 1) # (B, 4, 3, H, W)
        B = obs_image.shape[0]
        all_preds = []

        pred_x_cond_pixels = obs_image
        for i, timestep in tqdm(enumerate(eval_timesteps), total=len(eval_timesteps)):
            y = delta[:, i]
            x_cond_pixels = pred_x_cond_pixels[:,-4:].to(device)
            rel_t = (torch.ones(1)*0.0078125).to(device)
            samples = self.inference(x_cond_pixels, y, None, rel_t=rel_t)
            x_cond_pixels = samples.unsqueeze(1) # torch.clip(samples, -1., 1.)
            pred_x_cond_pixels = torch.cat([pred_x_cond_pixels.to(x_cond_pixels), x_cond_pixels], dim=1)
            # samples = (samples * 127.5 + 127.5).permute(0, 2, 3, 1).clamp(0,255).to("cpu", dtype=torch.uint8).numpy()
            all_preds.append(samples)

        video_tensors = torch.stack(all_preds, dim=1) # (B, T, C, H, W)

        return video_tensors

    def visualize_preds(self, output_dir, idxs, tag, x_pred_pixels):
        for batch_idx, sample_idx in enumerate(idxs.squeeze()):
            sample_idx = int(sample_idx.item())
            sample_folder = os.path.join(output_dir, f'id_{sample_idx}')
            os.makedirs(sample_folder, exist_ok=True)
            image_file = os.path.join(sample_folder, f'{tag}.png')
            self.save_image(image_file, x_pred_pixels[batch_idx], unnormalize_img=True)


    def save_image(self, output_file, img, unnormalize_img):
        img = img.detach().cpu()
        if unnormalize_img:
            img = misc.unnormalize(img)

        img = img * 255
        img = img.byte()
        image = Image.fromarray(img.permute(1, 2, 0).numpy(), mode='RGB')

        image.save(output_file)


    # def convert_discrete_action_to_delta(self, action: str, heading_deg: float, step_size=1.0, turn_angle_deg=30.0):
    #     """
    #     Convert a discrete navigation action into continuous pose delta.
    #     """
    #     heading_rad = math.radians(heading_deg)
    #     if action == "forward":
    #         dx = step_size * math.cos(heading_rad)
    #         dy = step_size * math.sin(heading_rad)
    #         dtheta = 0.0
    #     elif action == "turn_left":
    #         dx = 0.0
    #         dy = 0.0
    #         dtheta = turn_angle_deg
    #     elif action == "turn_right":
    #         dx = 0.0
    #         dy = 0.0
    #         dtheta = -turn_angle_deg
    #     elif action == "stop":
    #         dx = 0.0
    #         dy = 0.0
    #         dtheta = 0.0
    #     else:
    #         raise ValueError(f"Unknown action: {action}")
    #     return dx, dy, math.radians(dtheta)  # output angle in radians

    def convert_discrete_action_to_delta(self, action, heading_rad, step_size=1, turn_deg=30.0):
        if action == "forward":
            dx = step_size * math.cos(heading_rad)
            dy = step_size * math.sin(heading_rad)
            dtheta = 0.0
        elif action == "turn_left":
            dx = 0.0
            dy = 0.0
            dtheta = math.radians(turn_deg)
        elif action == "turn_right":
            dx = 0.0
            dy = 0.0
            dtheta = -math.radians(turn_deg)
        elif action == "stop":
            dx = 0.0
            dy = 0.0
            dtheta = 0.0
        else:
            raise ValueError(f"Unknown action: {action}")
        return dx, dy, dtheta

    def discrete_actions_to_relative_trajectory_to_delta(self, action_list, step_size=0.2, turn_deg=22.5, normalize=True, spacing=0.25):
        positions = [np.array([0.0, 0.0])]
        yaws = [0.0]  # in radians

        for act in action_list:
            dx, dy, dtheta = self.convert_discrete_action_to_delta(act, yaws[-1], step_size, turn_deg)
            new_pos = positions[-1] + np.array([dx, dy])
            new_yaw = yaws[-1] + dtheta
            positions.append(new_pos)
            yaws.append(new_yaw)

        positions = np.array(positions)
        yaws = np.array(yaws)

        # Convert to local coords
        waypoints_pos = to_local_coords(positions, positions[0], yaws[0])
        waypoints_yaw = angle_difference(yaws[0], yaws)

        actions = np.concatenate([waypoints_pos, waypoints_yaw[:, None]], axis=1)
        actions = actions[1:]  # drop the initial state

        if normalize:
            actions[:, :2] /= spacing

        # import ipdb;ipdb.set_trace()
        self.ACTION_STATS = {'min': np.array([[-2.5, -4. ]]), 'max': np.array([[5, 4]])}
        actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
        deltas = get_delta_np(actions)

        return deltas  # shape: (T, 3)

    def action_list_to_delta(self, b_action, step_size=0.20, turn_deg=22.5):
        """
        Convert list of discrete actions into batched delta tensors using heading-aware updates.
        Args:
            b_action (List[List[str]]): A batch of action sequences (B, T).
        Returns:
            Tensor of shape (B, T, 3): Each delta is (dx, dy, dyaw) in radians.
        """
        all_deltas = []
        for action_seq in b_action:
            heading_deg = 0.0
            deltas = []
            for act in action_seq:
                dx, dy, dtheta = self.convert_discrete_action_to_delta(act, heading_deg, step_size, turn_deg)
                heading_deg += math.degrees(dtheta)  # update heading in degrees for next step
                deltas.append([dx, dy, dtheta])
            all_deltas.append(torch.tensor(deltas, dtype=torch.float32))
        return torch.stack(all_deltas, dim=0)  # shape: (B, T, 3)

    def action_list_to_delta_v2(self, actions_list, step_size=1.0, angle_scale=0.5):
        """
        Convert a list of actions (e.g., ['Forward', 'Rotate Left']) to a tensor of deltas.

        Args:
            actions_list (List[List[str]]): Batch of action sequences, shape (B, T)
            step_size (float): Distance moved when taking 'Forward'
            angle_scale (float): Rotation per step in radians (0.5 ≈ ~28.6 degrees)

        Returns:
            Tensor: shape (B, T, 3), with [dx, dy, dyaw]
        """
        commands = {
            'forward': [1, 0, 0],
            'turn_right': [0, 0, -angle_scale],
            'turn_left': [0, 0, angle_scale],
        }

        all_deltas = []
        for action_seq in actions_list:
            heading = 0.0  # in radians, facing +x
            deltas = []
            for act in action_seq:
                if act not in commands:
                    raise ValueError(f"Unknown action: {act}")
                dx, dy, dyaw = commands[act]
                # Rotate (dx, dy) into current heading
                world_dx = step_size * (dx * math.cos(heading) - dy * math.sin(heading))
                world_dy = step_size * (dx * math.sin(heading) + dy * math.cos(heading))
                heading += dyaw
                # deltas.append([world_dx, world_dy, dyaw])
                deltas.append([dx, dy, dyaw])

            all_deltas.append(torch.tensor(deltas, dtype=torch.float32))

        return torch.stack(all_deltas, dim=0)  # (B, T, 3)


    def prepare_nwm_input(self, b_action, img_list, device="cuda"):
        """
        Prepares input for NWM navigator inference_batch from given input_dict.
        Args:
            input_dict: {
                "b_action": List[List[str]],  # shape (B, T)
                "save_dirs": List[str],       # shape (B,)
            }
            image_size: (H, W) for resizing input image
            device: 'cuda' or 'cpu'
        Returns:
            idxs: (B, 1) tensor
            obs_image: (B, 1, 3, H, W) tensor
            delta: (B, T, 3) tensor
        """
        B = len(b_action)
        assert B == len(img_list), "Mismatch between b_action and save_dirs batch sizes"

        obs_list = []
        for img in img_list:
            image = img.resize((self.args.width, self.args.height))
            obs_tensor = misc.transform(image).unsqueeze(0)  # (1, C, H, W)
            obs_list.append(obs_tensor)

        obs_image = torch.stack(obs_list, dim=0).to(device)  # (B, 1, 3, H, W)
        idxs = torch.arange(B).unsqueeze(1).float().to(device)  # (B, 1)
        # delta = self.action_list_to_delta(b_action).to(device)  # (B, T, 3)
        # delta = self.action_list_to_delta_v2(b_action).to(device)  # (B, T, 3)

        all_deltas = []
        for action_seq in b_action:
            delta = self.discrete_actions_to_relative_trajectory_to_delta(action_seq)
            delta = torch.tensor(delta, dtype=torch.float32).to(device)  # (T, 3)
            all_deltas.append(delta)
        delta = torch.stack(all_deltas, dim=0) # (B, T, 3)

        # import ipdb;ipdb.set_trace()

        return idxs, obs_image, delta

    def inference_batch(self, input_dict):
        """
        Run batched inference over a batch of trajectories, and generate predictions
        either as a rollout or at specific time checkpoints.

        Args:
            idxs (Tensor): Batch of sample indices (B, 1), used for saving outputs.
            obs_image (Tensor): Observed input image sequence (B, T_1, C, H, W).
            gt_image (Tensor): Ground truth future image sequence (B, T_2, C, H, W).
            delta (Tensor): Corresponding delta pose sequence (B, T_2, 3).
            output_dir (str): Directory to save generated outputs.
        """
        b_action, save_dirs, b_image, return_objects = parse_input_data(input_dict)
        # 1. process the input data to pil imgs:
        img_list = prepare_image_list(save_dirs, b_image, return_as_paths=False)

        idxs, obs_image, delta = self.prepare_nwm_input(b_action, img_list)

        obs_image = obs_image[:, -self.context_size:].to(self.device)
        gt_image = None #gt_image.to(self.device)
        delta = delta.to(self.device)

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            # Default setting (eval_type=time): Perform one-shot predictions at fixed time intervals (2^i seconds)
            secs = np.array([2 ** i for i in range(self.args.num_sec_eval)])
            # video_tensors = self.generate_time(idxs, obs_image, gt_image, delta, secs, self.args) # (B, T, C, H, W)
            video_tensors = self.generate_from_genframe(idxs, obs_image, gt_image, delta, secs, self.args) # (B, T, C, H, W)
            # out_paths = save_predict(video_tensors, b_action, save_dirs, model_type='nwm')

        # return {'save_dirs': save_dirs, "pred_frames": video_tensors.cpu().numpy(), "out_paths": out_paths}
        return process_output_dict(b_action, save_dirs, return_objects, video_tensors, model_type='nwm')


def test_sample(args, device):
    # _, _, device, _ = init_distributed()
    print(args)
    device = torch.device(device)
    # num_tasks = dist.get_world_size()
    # global_rank = dist.get_rank()
    # exp_eval = args.exp_id

    # model & config setup
    # if args.gt:
    #     args.save_output_dir = os.path.join(args.output_dir, 'gt')
    # else:
    #     exp_name = os.path.basename(exp_eval).split('.')[0]
    #     args.save_output_dir = os.path.join(args.output_dir, exp_name)
    # os.makedirs(args.save_output_dir, exist_ok=True)

    config = load_and_merge_config()

    navigator = NavigationWM(
        args=args,
        config=config,
        device=device
    )
    # test case
    input_dict = {
        "b_action": [
            # ["stop", "turn_left", "forward", "turn_right"] + ["forward"]*10
            ["forward"]*14
        ],
        "save_dirs": [
            "data/nwm_debug1"
        ]
    }

    return_dict = navigator.inference_batch(input_dict)
    print(f"[nwm_worker] return_dict: {return_dict}")
    sys.exit(0)


def load_and_merge_config(exp_eval="downstream/api_models/nwm/config/nwm_cdit_xl.yaml"):
    with open("downstream/api_models/nwm/config/eval_config.yaml", "r") as f:
        default_config = yaml.safe_load(f)
    config = default_config

    with open(exp_eval, "r") as f:
        user_config = yaml.safe_load(f)
    config.update(user_config)
    return config


if __name__ == "__main__":
    # The last argument is the w_fd we pass from main
    pipe_fd = int(sys.argv[-1])
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--out_height", type=int, default=480)
    parser.add_argument("--out_width", type=int, default=480)
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="05.05_NWMdebug")
    # parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str,
                        default="checkpoints/models--facebook--nwm/snapshots/bd92e0cdf7f3bc64cb2009fcd8882e96195e6150/0100000.pth.tar")
    # NWM unique args:
    parser.add_argument("--num_sec_eval", type=int, default=5)
    parser.add_argument("--input_fps", type=int, default=4)
    parser.add_argument("--context_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--eval_type", type=str, default="time", help="type of evaluation has to be either 'time' or 'rollout'")
    # Rollout Evaluation Args
    parser.add_argument("--rollout_fps_values", type=str, default='1,4', help="")
    parser.add_argument("--gt", type=int, default=0, help="set to 1 to produce ground truth evaluation set")
    args = parser.parse_args(sys.argv[1:-1])
    # args = parser.parse_args()

    args.rollout_fps_values = [int(fps) for fps in args.rollout_fps_values.split(',')]

    log_path = osp.join(args.log_dir, f"{args.exp_id}", f"nwm_worker",f"worker{args.device}.log")
    setup_logger(log_path)
    print(f"[nwm_worker] All Args:\n {args}")
    # _, _, device, _ = init_distributed(port=37124)
    device = torch.device(args.device)
    # For debug:
    # -------------------------------- ---------------------------
    # test_sample(args, device)

    config = load_and_merge_config()

    navigator = NavigationWM(
        args=args,
        config=config,
        device=device
    )
    print(f"[nwm_worker] NWM model loaded successfully!")

    def do_some_tasks(input_dict):
        """
        Processes a task (mimicking ARSolver.b_genex) using global objects.
        Expects input_dict with keys 'b_image' and 'b_action'. Uses the global
        'navigator' and 'args' initialized in main.
        """
        action_seq_strs = []
        for action_seq in input_dict['b_action']:
            # assert (np.array(action_seq[1:]) == 1).all(), f"Only support b_action == forward, got {action_seq}"
            action_seq_str = convert_actions_from_id_to_str(action_seq, add_unitlen=False)
            action_seq_strs.append(action_seq_str)
        input_dict['b_action'] = action_seq_strs

        return_dict = navigator.inference_batch(input_dict)

        return return_dict


    worker_main(pipe_fd, do_some_tasks)
