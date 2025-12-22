
IGENEX_ACTION_IDS = {'forward': 1, 'turn_left': 2, 'turn_right': 3, 'stop': 4, 'placeholder': 0}
IGENEX_ACTION_STRS = {
    v: k for k, v in IGENEX_ACTION_IDS.items()
}
from abc import ABC, abstractmethod
import os
import random
import numpy as np
import os.path as osp
from PIL import Image
import torch
from torchvision import transforms
from downstream.utils.saver import save_action_sequence, save_predict, prepare_saved_imgs_nwm
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import functional as F


def convert_actions_from_str_to_id(actions: list[str]) -> list[int]:
    """
    Convert a list of action strings to their corresponding IDs.
    """
    return [IGENEX_ACTION_IDS[action] for action in actions]


def convert_actions_from_id_to_str(actions: list[int], add_unitlen=True) -> list[str]:
    """
    Convert a list of action IDs to their corresponding strings.
    """
    str_list = []
    for action in actions:
        action_str = IGENEX_ACTION_STRS[action]
        if add_unitlen:
            if action_str == "forward":
                action_str = "forward 0.2m"
            elif "turn" in action_str:
                action_str = f"{action_str} 22.5°"
        str_list.append(action_str)
    print(f"Converted action_str into: {str_list}")
    return str_list

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


######For processing the input data of models from diffusers######
def process_b_action_nav(b_action):
    PROMPT_TEMPLATE = (
        "Follow this sequence of camera motions: {action}."
    )
    # PROMPT_TEMPLATE_ = (
    #     # "It is an indoor scene.",
    #     ""
    # )
    txt_list = []
    for actions in b_action:
        action_seq_str = convert_actions_from_id_to_str(actions[1:])
        txt_list.append(PROMPT_TEMPLATE.format(action=str(action_seq_str)))
        # txt_list.append(PROMPT_TEMPLATE_)
        # print(f"text_prompt: {txt_list}")
    return txt_list

def process_b_action_manip(b_action):
    PROMPT_TEMPLATE = (
        "Follow the instruction to move the robotic arm: {action}."
    )
    txt_list = []
    for instruction in b_action:
        txt_list.append(PROMPT_TEMPLATE.format(action=instruction))

    print(f"text_prompt: {txt_list}")
    return txt_list

def process_b_action_freetext(b_action):
    PROMPT_TEMPLATE = (
        "{action}."     #Follow the instruction in the freetext:
    )
    txt_list = []
    for instruction in b_action:
        txt_list.append(PROMPT_TEMPLATE.format(action=instruction))
    print(f"text_prompt: {txt_list}")
    return txt_list

def process_b_action(b_action, task_type="navigation"):
    if hasattr(b_action, "tolist"):
        b_action = b_action.tolist()
    if task_type == "navigation":
        return process_b_action_nav(b_action)
    elif task_type == "manipulation":
        return process_b_action_manip(b_action)
    elif task_type == "freetext":
        return process_b_action_freetext(b_action)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

def process_save_dirs(save_dirs):
    """
    Return a list of PIL images loaded from the save directories.
    """
    from diffusers.utils import load_image

    img_list = []
    paths = [osp.join(save_dir, "cond_rgb.png") for save_dir in save_dirs]
    for path in paths:
        img_list.append(load_image(path))
    return img_list, paths


def images_to_tensor(
    pipe_images: list[list[Image.Image]],
    uni_samp: int = None,
    save_size: tuple[int, int] = (480, 480),
    center_crop: bool = False,
) -> torch.Tensor:
    """
    Convert a batch of per‑frame PIL images to BxTxCxHxW float tensors.
    pipe_images : List[List[PIL.Image]]
        Outer list over batch, inner list over timesteps.
    uni_samp : int | None
        If given, uniformly subsample each sequence to this length.
    save_size : (int, int)
        Final (width, height).  Used for the crop size as well.
    center_crop : bool
        If True, **preserve aspect ratio**:
            1. Resize so that the *shorter* side equals `min(save_size)`.
            2. Center‑crop to exactly `save_size`.
        If False (default), simply resize to `save_size` ignoring aspect ratio.
    """
    w_out, h_out = save_size
    batch_tensors = []

    for image_list in pipe_images:
        # Optional uniform subsampling
        seq_len = len(image_list)
        if uni_samp is not None:
            if uni_samp <= seq_len:
                # Pure down-sampling (same behaviour as before)
                idx = np.linspace(0, seq_len - 1, uni_samp).astype(int)
            else:
                # Need to pad with the last frame
                idx_first = list(range(seq_len))                    # keep every frame once
                idx_pad   = [seq_len - 1] * (uni_samp - seq_len)    # repeat the last one
                idx = np.array(idx_first + idx_pad, dtype=int)
            image_list = [image_list[i] for i in idx]

        tensor_frames = []
        for img in image_list:
            if center_crop:
                # Step 1 – aspect‑ratio‑preserving resize on shorter side
                short_side = min(w_out, h_out)
                img = F.resize(img, short_side, interpolation=Image.BICUBIC, antialias=True)
                # Step 2 – center crop to the exact output shape
                img = F.center_crop(img, (h_out, w_out))
            else:
                img = img.resize(save_size, Image.BICUBIC)

            tensor_frames.append(F.to_tensor(img))          # C × H × W

        video_tensor = torch.stack(tensor_frames, dim=0)    # T × C × H × W
        batch_tensors.append(video_tensor)

    return torch.stack(batch_tensors, dim=0)                # B × T × C × H × W


def maybe_compile(mod: torch.nn.Module):
    """Wrap `mod` with torch.compile if possible; otherwise fall back."""
    if not hasattr(torch, "compile"):          # pre-2.0
        return mod
    try:
        return torch.compile(
            mod,
            mode="reduce-overhead",
            fullgraph=False,                   # safer for HF models
            # options={"triton": {"autotune": False}},  # synchronous build
        )
    except Exception as err:
        print(f"[compile disabled] {mod.__class__.__name__}: {err}")
        return mod

###### For processing the input_dict and output_dict of models ######
def process_input_dict(input_dict, task_type, world_model_name, return_as_paths=False):
    """
    Process the input_dict to extract b_action, save_dirs, return_objects, txt_list, img_list.
    """
    assert input_dict["request_model_name"] == world_model_name, f"request_model_name: {input_dict['request_model_name']} does not match deployed world_model_name: {world_model_name}"
    b_action, save_dirs, b_image, return_objects = parse_input_data(input_dict)

    # 1. process the input data - PIL images:
    img_list = prepare_image_list(save_dirs, b_image, return_as_paths)

    # 2. process the input data - actions:
    assert len(b_action) == len(save_dirs)
    txt_list = process_b_action(b_action, task_type)

    return b_action, save_dirs, return_objects, txt_list, img_list

def parse_input_data(input_dict):
    b_action = input_dict["b_action"]
    save_dirs = input_dict["save_dirs"]
    b_image = input_dict.get("b_image", None)
    return_objects = "return_objects" in input_dict and input_dict["return_objects"]
    return b_action,save_dirs,b_image,return_objects

def prepare_image_list(save_dirs, b_image, return_as_paths):
    if b_image is None:
        img_list, img_paths = process_save_dirs(save_dirs) # List[PIL.Image]
        if return_as_paths:
            # If return_paths is True, transform img_list into List[pathlib.Path]
            img_list = img_paths
    else:
        if return_as_paths:
            save_dirs_temp = [os.path.join(d, "temp") for d in save_dirs]
            img_paths = prepare_saved_imgs_nwm(b_image, save_dirs_temp)
            img_list = img_paths
        else:
            b_image = torch.from_numpy(b_image)
            assert b_image.dim() == 4, f"b_image should be B C H W, got {b_image.shape}"
            assert b_image.dtype == torch.uint8, f"b_image should be uint8, got {b_image.dtype}"
            img_list = [to_pil_image(b_image[i]) for i in range(b_image.shape[0])]
    return img_list

def process_output_dict(b_action, save_dirs, return_objects, video_tensors, model_type="default"):
    video_tensors = video_tensors.cpu().float()  # Convert to float tensor
    if return_objects:
        out = {
            "pred_frames": (np.clip(video_tensors.numpy(), 0, 1) * 255).astype(np.uint8),
            "save_dirs": save_dirs,
        }
    else:
        save_predict(video_tensors, b_action, save_dirs, model_type)
        out = {"save_dirs": save_dirs}
    print(f"video_tensors shape: {video_tensors.shape}")
    return out

# --------------------------------------------------------------------------- #
class DiffuserModel(ABC):
    def __init__(self, args):
        self.args = args
        self.pipe = self._load_pipe()
        if hasattr(args, "enable_compile") and args.enable_compile:
            self._compile_pipe()
        self.pipe_args = self._load_pipe_args()

    def _compile_pipe(self):
        """
        Replace selected pipeline components with their compiled versions.
        The list covers common blocks in most Diffusers pipelines; feel free to extend.
        """
        for name in ["transformer", "unet", "vae"]:
            mod = getattr(self.pipe, name, None)
            if isinstance(mod, torch.nn.Module):
                print(f"[torch.compile] Compiling {name}...")
                setattr(self.pipe, name, maybe_compile(mod))

    @abstractmethod
    def _load_pipe(self): ...

    @abstractmethod
    def _load_pipe_args(self):
        """Load the other necessary arguments for the pipeline when call pipe(*args)."""

    def inference_batch(self, input_dict):
        b_action, save_dirs, return_objects, txt_list, img_list = (
            process_input_dict(input_dict, self.args.task_type, self.args.world_model_name)
        )

        # 3. run the pipeline
        pipe_images = self.generate_pipe_images(txt_list, img_list)
        video_tensors = self.postprocess_frames(pipe_images)

        out = process_output_dict(b_action, save_dirs, return_objects, video_tensors)
        return out

    def postprocess_frames(self, pipe_images: list[list[Image.Image]]) -> torch.Tensor:
        video_tensors = images_to_tensor(
            pipe_images,
            uni_samp=self.args.num_output_frames,
            save_size=(self.args.width, self.args.height),
        )
        return video_tensors

    def generate_pipe_images(self, txt_list, img_list):
        pipe_images = self.pipe(
            # generator=generator,
            image=img_list,
            prompt=txt_list,
            **self.pipe_args,
        ).frames

        return pipe_images
