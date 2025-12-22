"""
Cosmos Predict2.5 WM_server worker.

This worker implements the WM_server_API contract:
- Reads `input_dict` payloads from manager (see `downstream/utils/worker_manager.py`)
- Runs Cosmos Predict2.5 inference
- Returns an `output_dict` that includes `save_dirs` and optionally `pred_frames`

Notes:
- Cosmos Predict2.5 is vendored under `downstream/api_models/cosmos-predict2.5/`.
- This model is NOT integrated through the diffusers pipeline here; we call
  `cosmos_predict2.inference.Inference` directly (similar to
  `cosmos-predict2.5/examples/inference.py`) but without the CLI wrapper.

Env (typical):
  conda activate cosmos-predict25
  See `downstream/api_models/cosmos-predict2.5/my_env_setup.log`
"""
from __future__ import annotations

import sys
import os.path as osp
from dataclasses import dataclass
from typing import Optional, Literal

import torch
from torchvision.transforms import functional as TVF

# Allow running this file directly without needing PYTHONPATH="." (WM manager sets it).
_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger import setup_logger  # noqa: E402
from downstream.utils.worker_manager import worker_main  # noqa: E402
from downstream.api_models import process_input_dict, process_output_dict, images_to_tensor  # noqa: E402

@dataclass(frozen=True)
class Cosmos25GenArgs:
    inference_type: str  # text2world|image2world|video2world
    resolution: str  # "none" or "H,W"
    num_output_frames: int
    num_steps: int
    guidance: int
    seed: int
    negative_prompt: str
    enable_autoregressive: bool
    chunk_size: int
    chunk_overlap: int


class Cosmos25Engine:
    """
    Thin wrapper that holds a single loaded Cosmos Predict2.5 inference pipeline.
    """

    def __init__(self, setup_args) -> None:
        from cosmos_predict2.inference import Inference

        self.infer = Inference(setup_args)

    @torch.no_grad()
    def generate_one(self, *, name: str, prompt: str, input_path: Optional[str], gen: Cosmos25GenArgs) -> torch.Tensor:
        """
        Returns float tensor in [0,1], shape (T, C, H, W).
        """
        inference_type = str(gen.inference_type)
        if inference_type == "text2world":
            num_latent_conditional_frames = 0
        elif inference_type == "image2world":
            num_latent_conditional_frames = 1
        elif inference_type == "video2world":
            num_latent_conditional_frames = 2
        else:
            raise ValueError(f"Unknown inference_type: {inference_type}")

        common_kwargs = dict(
            prompt=prompt,
            input_path=str(input_path) if input_path is not None else None,
            guidance=int(gen.guidance),
            num_latent_conditional_frames=num_latent_conditional_frames,
            resolution=str(gen.resolution),
            seed=int(gen.seed),
            negative_prompt=str(gen.negative_prompt),
            num_steps=int(gen.num_steps),
        )

        if gen.enable_autoregressive:
            video_btchw = self.infer.pipe.generate_autoregressive_from_batch(
                **common_kwargs,
                chunk_size=int(gen.chunk_size),
                chunk_overlap=int(gen.chunk_overlap),
            )
        else:
            video_btchw = self.infer.pipe.generate_vid2world(
                **common_kwargs,
            )

        # Cosmos returns [-1,1] float, shape (B, C, T, H, W)
        video_cthw = (1.0 + video_btchw[0]) / 2.0   # torch.Size([1, 3, 93, 704, 1280])
        video_tchw = video_cthw.permute(1, 0, 2, 3).contiguous()
        return video_tchw.clamp(0.0, 1.0)


DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, "
    "shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, "
    "poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, "
    "unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, "
    "visual noise, and flickering. Overall, the video is of poor quality."
)


def _build_gen_from_args(args) -> Cosmos25GenArgs:
    """
    Build generation defaults from:
    - Cosmos-native `args.overrides` (if provided)
    - worker-level `default_*` fields (fallbacks)
    """
    ov = args.overrides.model_dump(exclude_none=True)
    inference_type = ov.get("inference_type", args.default_inference_type)
    return Cosmos25GenArgs(
        inference_type=inference_type,
        resolution=ov.get("resolution", args.default_resolution),
        num_output_frames=ov.get("num_output_frames", args.default_num_frames),
        num_steps=ov.get("num_steps", args.default_num_steps),
        guidance=ov.get("guidance", args.default_guidance),
        seed=ov.get("seed", args.default_seed),
        negative_prompt=ov.get("negative_prompt", args.default_negative_prompt),
        enable_autoregressive=ov.get("enable_autoregressive", args.default_enable_autoregressive),
        chunk_size=ov.get("chunk_size", args.default_chunk_size),
        chunk_overlap=ov.get("chunk_overlap", args.default_chunk_overlap),
    )

def _run_wm_batch(args, *, engine: Cosmos25Engine, gen: Cosmos25GenArgs, input_dict: dict) -> dict:
    """
    Shared execution path for both WM worker mode and local `--debug` runs.
    """
    b_action, save_dirs, return_objects, txt_list, img_paths = process_input_dict(
        input_dict, args.task_type, args.world_model_name, return_as_paths=True
    )

    videos = []
    for i, (save_dir, prompt, img_path) in enumerate(zip(save_dirs, txt_list, img_paths)):
        if isinstance(save_dir, str):
            name = osp.basename(save_dir.rstrip("/")) or f"sample_{i}"
        else:
            name = f"sample_{i}"
        in_path = None if gen.inference_type == "text2world" else str(img_path)
        video_tchw = engine.generate_one(name=name, prompt=prompt, input_path=in_path, gen=gen)
        # Convert video_tchw -> List[PIL.Image] then reuse shared `images_to_tensor` helper
        video_pil = [TVF.to_pil_image(frame_cpu) for frame_cpu in video_tchw.cpu()]
        video_btchw = images_to_tensor(
            [video_pil],  # outer list is batch
            uni_samp=gen.num_output_frames,
            save_size=(args.out_width, args.out_height),
        )
        videos.append(video_btchw[0])

    video_tensors = torch.stack(videos, dim=0)  # (B, T, C, H, W)
    return process_output_dict(b_action, save_dirs, return_objects, video_tensors)


def test_sample(args) -> None:
    """
    Local debug helper. Requires an existing `cond_rgb.png` under `debug_save_dir`.
    """
    engine = Cosmos25Engine(args.setup)
    gen = _build_gen_from_args(args)
    input_dict = {
        "b_action": [[1] * 14],
        "save_dirs": [args.debug_save_dir],
        "return_objects": [False],
    }
    out = _run_wm_batch(args, engine=engine, gen=gen, input_dict=input_dict)
    print(f"[cosmos25_worker] test_sample output keys: {list(out.keys())}")


if __name__ == "__main__":
    # Use the same arg parsing style as the original Cosmos inference entrypoint:
    # `tyro.cli(...)` + `handle_tyro_exception`.
    # _maybe_add_cosmos25_to_syspath(None)

    import pydantic
    import tyro
    from cosmos_oss.init import init_environment, init_output_dir
    from cosmos_predict2.config import (
        InferenceOverrides,
        SetupArguments,
        handle_tyro_exception,
        is_rank0,
    )

    class Args(pydantic.BaseModel):
        """
        WM worker arguments.

        - `setup` is Cosmos-native setup args (same as `examples/inference.py`).
        - `overrides` is Cosmos-native per-sample overrides; we apply them as defaults
          when creating per-request `InferenceArguments` from WM `input_dict`.
        """
        world_model_name = "cosmos25"

        # Cosmos-native setup + overrides (same types as official CLI)
        model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
        # Cosmos-native setup args (same as official CLI). Note: Cosmos requires
        # `--output-dir` (and `--model` due to a pre-validator) to be provided.
        setup: SetupArguments
        overrides: InferenceOverrides = pydantic.Field(default_factory=InferenceOverrides)

        # new added WM / logging
        log_dir: str = "downstream/logs"
        exp_id: str = "cosmos25_debug"
        device: str = "cuda:0"
        task_type: Literal["navigation", "manipulation", "freetext"] = "navigation"

        # Output formatting (what WM will return / save)
        out_width: int = 480
        out_height: int = 480

        # Cosmos vendored repo path (optional; helps local installs)
        cosmos_repo_dir: Optional[str] = None

        # Worker defaults (used if not specified in overrides)
        default_inference_type: Literal["text2world", "image2world", "video2world"] = "image2world"
        default_resolution: str = "576,576"   # default: 704, 1280
        default_num_frames: int = 14          # 93
        default_num_steps: int = 35
        default_guidance: int = 7
        default_seed: int = 0
        default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
        default_enable_autoregressive: bool = False
        default_chunk_size: int = 77
        default_chunk_overlap: int = 1

        # Debug helper
        debug: bool = False
        debug_save_dir: str = "downstream/api_models/test_sample/debug"

        # (No automatic output_dir syncing; pass `--output-dir` explicitly in workers_cfg.)

    init_environment()

    if "--debug" in sys.argv:
        argv = sys.argv[1:]
        pipe_fd = None
    else:
        argv = sys.argv[1:-1]
        pipe_fd = int(sys.argv[-1])

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
            args=argv,
        )
    except Exception as e:
        handle_tyro_exception(e)

    # Ensure vendored Cosmos packages are importable (use user-supplied repo dir if any).
    # _maybe_add_cosmos25_to_syspath(args.cosmos_repo_dir)

    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    log_path = osp.join(args.log_dir, f"{args.exp_id}", "cosmos25_worker", f"worker{args.device}.log")
    setup_logger(log_path)
    print(f"[cosmos25_worker] All Args:\n {args}")

    if args.debug:
        test_sample(args)
        # In debug runs, clean up Cosmos env similar to official script.
        from cosmos_oss.init import cleanup_environment

        cleanup_environment()
        sys.exit(0)

    if pipe_fd is None:
        raise ValueError("Missing pipe fd (worker mode expects manager to append it). Use --debug for local runs.")

    engine = Cosmos25Engine(args.setup)
    print("[cosmos25_worker] Cosmos25Engine loaded successfully!")

    gen = _build_gen_from_args(args)

    def do_some_tasks(input_dict: dict) -> dict:
        return _run_wm_batch(args, engine=engine, gen=gen, input_dict=input_dict)

    worker_main(pipe_fd, do_some_tasks)
