#!/usr/bin/env python

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import logging
import math
import os
import cv2
import shutil
from pathlib import Path

import accelerate
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from copy import deepcopy
from einops import rearrange

import diffusers
from diffusers import StableVideoDiffusionPipeline
from diffusers.models.lora import LoRALinearLayer
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler, UNetSpatioTemporalConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available

import sys
# Ensure repository root is available for imports like `utils` and `downstream`
FILE_DIR = Path(__file__).resolve().parent
PARENT_DIR = FILE_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from utils.svd_utils import (
    get_action_ids, norm_image,
    apply_conditioning_dropout,
    apply_discrete_conditioning_dropout,
    rotate_by_degrees,
    sample_latent_noise,
)
from dataset import init_dataloader
# from data.vlmbench.manip_dataset_load_code.dataset_manipulation import init_dataloader_man
from eval_inference import collect_inference_frames, Navigator

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")

# os.environ["WANDB_PROJECT"] = "igenex_training"
import torch.distributed as dist
from datetime import timedelta
from datetime import datetime
import time
# dist.init_process_group(
#     timeout=timedelta(minutes=30)  # Set timeout to 120 minutes
# )


# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()



def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(
        output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def tensor_to_vae_latent(t, vae, weight_dtype):
    video_length = t.shape[1]
    t = t.to(vae.dtype)

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents.to(dtype=weight_dtype)


def reconstruct_from_latent(latents, vae):

    latents = latents / vae.config.scaling_factor
    batch_size, video_length, _, _, _ = latents.shape

    latents = rearrange(latents, "b f c h w -> (b f) c h w")
    latents = latents.to(vae.dtype)
    reconstructed_images = vae.decode(latents, num_frames=video_length).sample

    reconstructed_images = rearrange(reconstructed_images, "(b f) c h w -> b f c h w", f=video_length)

    return reconstructed_images


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    parser.add_argument(
        "--base_folder",
        required=True, type=str,
        nargs='+',  # Accepts one or more values
        help="Base dataset folder paths"
    )
    parser.add_argument(
        "--val_base_folder",
        required=True, type=str,
        nargs='+',  # Accepts one or more values
        help="Base validation dataset folder paths"
    )
    parser.add_argument(
        "--habitat_dataset",
        type=str, default='hm3d-train',
        help="type of habitat dataset",
    )
    parser.add_argument(
        "--exp_id",
        default=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        type=str,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=6,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1000,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the text/image prompt"
            " multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=0.1,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=10.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--num_past_obs",
        type=int,
        default=5,
        help="The number of past observations to condition on.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=2,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    parser.add_argument(
        "--data_weighted_method",
        type=str,
        default='exponential',
        help="The method to weight the data sampling according to the overlap ratio, ['exponential', 'linear', 'uniform']",
    )
    parser.add_argument(
        "--data_cutoff_thr",
        type=float,
        default=0.5,
        help="The threshold to cut off the data from sampling according to the overlap ratio",
    )
    parser.add_argument(
        "--train_param_type",
        type=str,
        help="Decide which parameters to train, ['full', 'new', 'new+temp_layer']",
    )
    parser.add_argument(
        "--action_strategy",
        type=str, default='action_block',
        help="The strategy to handle the action embedding, ['action_block', 'micro_cond', 'action_block_nocfg']",
    )
    parser.add_argument(
        "--action_input_channel",
        type=int, help="The number of input channels for action Embedder, 14 for 'navigation', 23 for 'manipulation'",
    )
    parser.add_argument(
        "--task_type",
        type=str, help="The type of task to perform, ['manipulation', 'navigation']",
    )
    parser.add_argument(
        "--enable_reverse_aug",
        action="store_true", default=False,
        help="Whether to enable the reverse augmentation of data for training.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def main():
    args = parse_args()

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    print(f"Number of visible GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # DataLoaders creation:
    args.process_idx = accelerator.process_index
    train_dataloader = init_dataloader(args, enable_filter=False)
    # train_dataloader = init_dataloader_man(args) #train for manipulation tasks

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # time.sleep(90)  #wait the GPU worker to be ready
    # Load img encoder, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor", revision=args.revision
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    config_dict = {
        'pretrained_model_name_or_path': args.pretrained_model_name_or_path if args.pretrain_unet is None else args.pretrain_unet,
        'subfolder': 'unet',
        'low_cpu_mem_usage': False,
        'device_map': None,
        'action_strategy': args.action_strategy,
        'task_type': args.task_type,
        'num_frames': args.num_frames,
        'action_input_channel': args.action_input_channel,
    }
    try:
        unet = UNetSpatioTemporalConditionModel.from_pretrained(**config_dict)
    except Exception as e:
        logger.info(f"Error when load Unetpretained fp32 version: {e}")
        logger.info(f"Trying to load Unetpretained fp16 version...")
        unet = UNetSpatioTemporalConditionModel.from_pretrained(**config_dict, variant="fp16")
    # unet.init_action_proj(num_frames=args.num_frames, action_embeds_dim=256)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    unet.accelerator = accelerator

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device)     #Note we keep vae in full precision

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(
        ), model_cls=UNetSpatioTemporalConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                # weights.pop()
                if weights: # Don't pop if empty
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), UNetSpatioTemporalConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNetSpatioTemporalConditionModel.from_pretrained(
                    input_dir,
                    subfolder="unet",
                    action_strategy=args.action_strategy,
                    task_type=args.task_type,
                    num_frames=args.num_frames,
                    action_input_channel=args.action_input_channel
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Customize the parameters that need to be trained; if necessary, you can uncomment them yourself.
    if args.train_param_type == 'full':
        cond = 'True'
    elif args.train_param_type == 'new':
        cond = "('action' in name) or ('noise' in name)"
    elif args.train_param_type == 'new+temp_layer':
        cond = "'temporal_transformer_block' in name or ('action' in name) or ('noise' in name)"

    parameters_list = get_train_params(cond, unet)

    # check parameters
    if accelerator.is_main_process:
        rec_txt1 = open('params_freeze.txt', 'w')
        rec_txt2 = open('params_train.txt', 'w')
        for name, para in unet.named_parameters():
            if para.requires_grad is False:
                rec_txt1.write(f'{name}\n')
            else:
                rec_txt2.write(f'{name}\n')
        rec_txt1.close()
        rec_txt2.close()

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    optimizer, lr_scheduler = configure_optimizer_scheduler(
        args, accelerator.num_processes,
        optimizer_cls, parameters_list,
    )

    # Prepare everything with our `accelerator`.
    train_dataloader_ = train_dataloader
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )
    # Create a GradScaler for mixed precision training
    if accelerator.mixed_precision == "fp16" and accelerator.scaler is not None:
        prev_scale = accelerator.scaler.get_scale()

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # attribute handling for models using DDP
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            "SVDXtend",
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": args.exp_id,
                    "dir": args.output_dir,
                }
            }
        )

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    def encode_image(pixel_values):
        # Normalize the image with for CLIP input
        pixel_values = feature_extractor(
            images=pixel_values,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        pixel_values = pixel_values.to(
            device=accelerator.device, dtype=weight_dtype)
        image_embeddings = image_encoder(pixel_values).image_embeds
        return image_embeddings

    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]
        if hasattr(unet, 'module'):
            passed_add_embed_dim = unet.module.config.addition_time_embed_dim * \
                len(add_time_ids)
        else:
            passed_add_embed_dim = unet.config.addition_time_embed_dim * \
                len(add_time_ids)
        expected_add_embed_dim = unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size, 1)
        return add_time_ids


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "None":
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps)

            # Optional: Override optimizer's learning rate:
            desired_lr = args.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = desired_lr
                param_group["initial_lr"] = desired_lr  # your new desired learning rate
            lr_scheduler.scheduler.base_lrs = [desired_lr for _ in lr_scheduler.scheduler.base_lrs]
            lr_scheduler.scheduler._last_lr = [desired_lr for _ in lr_scheduler.scheduler._last_lr]

    logger.info(f"  arguments: ")
    logger.info(args, main_process_only=True)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    print(f'Current accelerator.gradient_accumulation_steps is {accelerator.gradient_accumulation_steps}')

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue
            # count_parameters(unet)
            # sys.exit(0)

            with accelerator.accumulate(unet):
                # first, convert images to latent space.
                pixel_values = batch["pixel_values"].to(    #torch.Size([1, 8, 3, 256, 512])
                    accelerator.device, non_blocking=True)
                past_obs = batch["past_obs"].to(   #torch.Size([1, 5, 3, 256, 512])
                    accelerator.device, non_blocking=True)
                actions = batch["actions"].to(
                    accelerator.device, non_blocking=True)    #torch.Size([1, 25])
                conditional_pixel_values = pixel_values[:, 0:1, :, :, :]    #torch.Size([1, 1, 3, 256, 512])
                # conditional_pixel_values = pixel_values[:, 0:1, :, :, :].repeat(1, args.num_frames, 1, 1, 1)

                latents = tensor_to_vae_latent(pixel_values, vae, weight_dtype)   #torch.Size([1, 8,   4, 32, 64])
                # latents_ = rotate_by_degrees(latents, angle=22.5)
                # img_recon = reconstruct_from_latent(latents, vae)    #torch.Size([1, 8, 3, 256, 512])
                # img_recon  = (img_recon + 1)/2     #now range [0, 1]
                # # img_recon_per = img_recon[0, -3, :, :, :].cpu()
                # img_recon_per = img_recon[0, -4, :, :, :].cpu()
                # save_img(img_recon_per, os.path.join('data_temp/check_turn22.5', 'pano_22.5_collected_base.png'))   #original pano

                # Sample noise that we'll add to the latents
                # noise = torch.randn_like(latents)
                noise = sample_latent_noise(actions, latents.shape, latents.device, latents.dtype, args.task_type)
                bsz = latents.shape[0]

                cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(latents)
                noise_aug_strength = cond_sigmas[0]         # TODO: support batch > 1
                cond_sigmas = cond_sigmas[:, None, None, None, None]        #torch.Size([1, 1, 1, 1, 1])
                conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values     #do what: add noise to the conditional image
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae, weight_dtype)[:, 0, :, :, :]    #torch.Size([1,   4, 32, 64]), only the first frame
                conditional_latents = conditional_latents / vae.config.scaling_factor   #do what: get no norm latents

                # Sample a random timestep for each image
                # P_mean=0.7 P_std=1.6
                sigmas = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(latents)
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                sigmas = sigmas[:, None, None, None, None]
                noisy_latents = latents + noise * sigmas    #torch.Size([1, 8, 4, 32, 64]), noise is the same as conditional_latents, magnitude is different
                timesteps = torch.Tensor(
                    [0.25 * sigma.log() for sigma in sigmas]).to(accelerator.device)

                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Get the text embedding for conditioning:
                # past_obs is shape of torch.Size([1, num_past_obs, 3, 256, 512])
                # past_obs_norm = ((past_obs + 1.0) / 2.0).squeeze(0)     # NOTE squeeze(0) for batch_size==1
                # cubes = convert_equi2cube(past_obs_norm)[0]             # return Size([num_past_obs, 6, 3, 224, 224])
                # cubes_unnorm = cubes * 2.0 - 1.0
                past_obs_pixel = norm_image(past_obs)                   # torch.Size([num_past_obs, 3, 224, 224])
                encoder_hidden_states = encode_image(past_obs_pixel)    # torch.Size([num_past_obs, 1024])

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    7, # fixed
                    127, # motion_bucket_id = 127, fixed
                    noise_aug_strength, # noise_aug_strength == cond_sigmas
                    encoder_hidden_states.dtype,
                    bsz,
                )
                added_time_ids = added_time_ids.to(latents.device)
                action_ids = get_action_ids(bsz, actions, args.action_strategy, weight_dtype)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.action_strategy == 'micro_cond' or args.action_strategy == 'action_block_nocfg':
                    cond_fn = apply_conditioning_dropout
                elif args.action_strategy == 'action_block':
                    cond_fn = apply_discrete_conditioning_dropout
                encoder_hidden_states, conditional_latents, action_ids = cond_fn(
                    encoder_hidden_states=encoder_hidden_states,
                    conditional_latents=conditional_latents,
                    action_conditioning=action_ids,
                    bsz=bsz,
                    conditioning_dropout_prob=args.conditioning_dropout_prob,
                    generator=generator  # can be None if you don't need reproducibility
                )

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                conditional_latents = conditional_latents.unsqueeze(    #torch.Size([1, **8**, 4, 32, 64]), add a dimension **8** represent frames
                    1).repeat(1, noisy_latents.shape[1], 1, 1, 1)
                inp_noisy_latents = torch.cat(                          #torch.Size([1, 8, 8, 32, 64])
                    [inp_noisy_latents, conditional_latents], dim=2)

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = latents    #torch.Size([1, 8, 4, 32, 64])
                # encoder_hidden_states = encoder_hidden_states.reshape(-1, 1, 1024)   #if multi condition: encoder_hidden_states.shape is torch.Size([1, 8, 1024])
                model_pred = unet(  #torch.Size([1, 8, 4, 32, 64])
                    inp_noisy_latents, timesteps, encoder_hidden_states,
                    added_time_ids=added_time_ids,
                    added_action_ids=action_ids,
                ).sample

                # Denoise the latents
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss = avg_loss.item()
                # check is train_loss is Nan or Inf
                if not torch.isfinite(loss).all(): # check if loss is finite
                    logger.error(f"ERROR: train_loss is not finite at step {global_step}")
                    logger.error(f"Current training data is: {batch['folder_path']} "
                                 f"and start_idx: {batch['start_idx']}")

                # Backpropagate
                accelerator.backward(loss)
                # if args.full_params:
                #     if accelerator.sync_gradients:
                #         accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if accelerator.mixed_precision == "fp16" and accelerator.scaler is not None:
                    current_scale = accelerator.scaler.get_scale()
                    if prev_scale != current_scale:
                        accelerator.print(f"Update current GradScaler scale: {current_scale}")
                        prev_scale = current_scale

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss, 'lr': lr_scheduler.get_lr()[0]}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        # for older version of diffusers, do it outside of main process
                        if 'DEEPSPEED' not in accelerator.distributed_type:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            os.makedirs(save_path, exist_ok=True)
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")
                    # sample images!
                    if (
                        (global_step % args.validation_steps == 0)
                        or (global_step == 1)
                    ):
                        logger.info(
                            f"Running validation... \n Generating {args.num_validation_images} videos."
                        )
                        # create pipeline
                        if args.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        # Before evaluation starts clears gradients to avoid OOM
                        optimizer.zero_grad(set_to_none=True)
                        # The models need unwrapping because for compatibility in distributed training mode.
                        torch.cuda.empty_cache()
                        pipeline = StableVideoDiffusionPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            image_encoder=accelerator.unwrap_model(image_encoder),
                            vae=accelerator.unwrap_model(vae),
                            revision=args.revision,
                            torch_dtype=weight_dtype,
                        )
                        pipeline = pipeline.to(accelerator.device)
                        pipeline.set_progress_bar_config(disable=True)

                        # run inference:
                        eval_inference(deepcopy(args), accelerator, unet,
                                       global_step, pipeline,
                                       test_num=args.num_validation_images,
                                       ema_unet=ema_unet if args.use_ema else None)
                        del pipeline
                        torch.cuda.empty_cache()

                # save checkpoints of DeepSpeed
                if global_step % args.checkpointing_steps == 0 and 'DEEPSPEED' in accelerator.distributed_type:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # After saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # After we save the new checkpoint, we need to have at most `checkpoints_total_limit` checkpoints
                            if len(checkpoints) > args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            image_encoder=accelerator.unwrap_model(image_encoder),
            vae=accelerator.unwrap_model(vae),
            unet=unet,
            revision=args.revision,
        )
        pipeline = pipeline.to(accelerator.device)

        # pipeline.save_pretrained(args.output_dir)
        metrics_dict = eval_inference(deepcopy(args), accelerator, unet,
                pipeline=pipeline, global_step=-1, test_num=20, num_workers=1,
                log_values=False,
                ema_unet=ema_unet if args.use_ema else None
        )
        log_dict = {}
        for key, value in metrics_dict.items():
            log_dict.update({key: value['value'][0]})
        log_dict = {f'{k}_final': v for k, v in log_dict.items()}
        accelerator.log(log_dict, step=global_step+1)
        del pipeline
        torch.cuda.empty_cache()

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
        # terminate_gpu_processes(train_dataloader_.processes)
    accelerator.end_training()

def get_train_params(cond, unet):
    """get the parameters that need to be trained from condition `cond`."""
    parameters_list = []

    # if 'temporal_transformer_block' in name or ('action' in name):  #or ('time_embedding' in name) or ('add_embedding' in name)
    for name, param in unet.named_parameters():
        if eval(cond):
            parameters_list.append(param)
            param.requires_grad = True
        else:
            param.requires_grad = False
    return parameters_list

def configure_optimizer_scheduler(args, num_processes, optimizer_cls, parameters_list):
    optimizer = optimizer_cls(
        parameters_list,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * num_processes,
        num_training_steps=args.max_train_steps * num_processes,
    )
    return optimizer, lr_scheduler


def eval_inference(args_val, accelerator, unet, global_step, pipeline, test_num=6, num_workers=1,
                   log_values=True, ema_unet=None):
    from evaluation import evaluate_video_metrics

    val_save_dir = os.path.join(args_val.output_dir, "val_images", f"step_{global_step}")
    if not os.path.exists(val_save_dir):
        os.makedirs(val_save_dir)

    args_val.test_num = test_num; args_val.num_workers = num_workers
    args_val.fix_seed = True
    args_val.base_folder = args_val.val_base_folder
    args_val.enable_reverse_aug = False
    # 1) Construct your dataset using args
    val_dataloader  = init_dataloader(args_val, enable_filter=False)
    # val_dataloader  = init_dataloader_man(args_val)
    navigator       = Navigator(args_val, accelerator.device)
    navigator.pipe  = pipeline
    device          = accelerator.device
    weight_dtype = torch.float16

    with torch.autocast(
        str(accelerator.device).replace(":0", ""), dtype=weight_dtype, enabled=True,
    ):
        frames = collect_inference_frames(device, weight_dtype, navigator, val_dataloader)

        # 3. Visualize and save the results
        for i in range(args_val.test_num):
            out_file = os.path.join(
                val_save_dir, f"val_{i}.mp4",
            )
            navigator.save_video_stitch(
                frames['gen_frames'][i],
                frames['gt_frames_np'][i],
                save_path=out_file)

        # 4. Evaluation metrics
        metrics_dict = evaluate_video_metrics(
            ground_truth_frames=frames['gt_frames'],
            generated_frames=frames['gen_frames'],
            only_final=True,         # or True, depending on your requirement
            fvd_method="styleganv",  # or "videogpt"
        )
        # Log results to wandb
        if log_values:
            log_dict = {}
            for key, value in metrics_dict.items():
                log_dict.update({key: value['value'][0]})
            accelerator.log(log_dict, step=global_step)

    if args_val.use_ema:
        # Switch back to the original UNet parameters.
        ema_unet.restore(unet.parameters())

    return metrics_dict



if __name__ == "__main__":
    main()
