import tensorflow as tf
import collections
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import argparse, sys
import torch


from se3ds.models import model_config
from se3ds.models import models
from utils.logger import setup_logger
from downstream.utils.saver import save_predict
from downstream.utils.worker_manager import (
    worker_main,
)
from downstream.api_models import process_output_dict, parse_input_data, prepare_image_list
from typing import List
from utils.svd_utils import (
    rotate_by_shift,
)
UNIT_DISTANCE = 0.2
UNIT_DEGREE = 22.5



def count_parameters_tf(model,
                        *,
                        verbose: bool = False,
                        return_skipped: bool = False
                       ):
    """
    Count total and trainable parameters in a TensorFlow model.

    The function traverses every sub-module.  If a layer has not been
    built yet (so accessing `layer.variables` raises a *ValueError*),
    it is skipped and reported.

    Parameters
    ----------
    model : tf.Module | any
        A `tf.keras.Model`, `tf.Module`, or an arbitrary Python wrapper
        that (directly or indirectly) contains TensorFlow variables.
    verbose : bool, default False
        Print a summary if True.
    return_skipped : bool, default False
        If True, also return the list of layer names that were skipped.

    Returns
    -------
    total_params : int
        Number of parameters (trainable + non-trainable) that were counted.
    trainable_params : int
        Number of trainable parameters.
    skipped_layers : list[str], optional
        Only if *return_skipped* is True.  Names of layers whose weights
        were not yet created and therefore not counted.
    """
    visited_var_ids = set()
    total_params    = 0
    trainable_params = 0
    skipped_layers   = []

    # If the object is not a tf.Module, fall back to the previous
    # generic walker so the function also works on wrappers.
    if not isinstance(model, tf.Module):
        from collections import deque
        q = deque([model])
        while q:
            obj = q.popleft()
            if isinstance(obj, tf.Variable):
                vid = id(obj)
                if vid in visited_var_ids:
                    continue
                visited_var_ids.add(vid)
                n = int(tf.size(obj).numpy())
                total_params += n
                if obj.trainable:
                    trainable_params += n
            elif isinstance(obj, (list, tuple, set)):
                q.extend(obj)
            elif isinstance(obj, dict):
                q.extend(obj.values())
            else:
                for name in dir(obj):
                    if not name.startswith('_'):
                        try:
                            q.append(getattr(obj, name))
                        except Exception:
                            pass
    else:
        # -------- tf.Module / Keras path ------------------------------------
        for layer in model.submodules:       # includes `model` itself
            try:
                vars_ = layer.variables
            except ValueError:
                skipped_layers.append(layer.name or layer.__class__.__name__)
                continue

            for v in vars_:
                vid = id(v)
                if vid in visited_var_ids:
                    continue
                visited_var_ids.add(vid)

                n = int(tf.size(v).numpy())
                total_params += n
                if v.trainable:
                    trainable_params += n

    pct = 100.0 * trainable_params / total_params if total_params else 0.0

    if verbose:
        print(f"Total parameters     : {total_params:,d}  ({total_params/1e9:.2f} B)")
        print(f"Trainable parameters : {trainable_params:,d}  ({trainable_params/1e9:.2f} B)")
        print(f"Fraction trainable   : {pct:.2f}%")
        if skipped_layers:
            print("\nLayers **not** counted because they are not built yet:")
            for name in skipped_layers:
                print(f"  · {name}")

    if return_skipped:
        return total_params, trainable_params, skipped_layers
    return total_params, trainable_params


def split_actions(actions: List[int], max_len: int) -> List[List[int]]:
    """
    Split a sequence of action-ids into sub-sequences under two rules.
    Rule 1: whenever the current value ≠ 1, start a new sublist (the value
            that triggered the split becomes the first element of the new sublist).
    Rule 2: a sublist must never grow beyond max_len elements; if adding the
            next element would exceed max_len, start a new sublist.
    Parameters
    ----------
    actions : List[int]
        Full sequence of action-ids.
    max_len : int
        Maximum allowed length of each sublist (must be ≥ 1).
    Returns
    -------
    List[List[int]]
        List of sublists that satisfy both rules.
    """
    if max_len < 1:
        raise ValueError("max_len must be at least 1")

    result: List[List[int]] = []
    current: List[int] = []

    for val in actions:
        # --- decide whether to break before adding `val` ---
        need_split = (
            val != 1 or                      # rule 1
            len(current) >= max_len          # rule 2 (would overflow)
        )

        if need_split and current:
            result.append(current)
            current = []

        current.append(val)

    if current:                              # append the tail
        result.append(current)

    return result


def tfm_action_to_coords(forward_steps, forward_dist=2.4):
    """
    converts action IDs to start and end coordinates in the SE3DS coordinate system.
    """
    step_dist = forward_dist / forward_steps

    # Rebuild the coordinate system for SE3DS: (x, y, z) where y is along the travel direction.
    start_pos = tf.constant([[0, 0, 0]], tf.float32)
    positions = [start_pos]
    # end_pos_ = tf.constant([[0, dist_2d.numpy(), 0]], tf.float32)

    for i in range(forward_steps):
        curr_pos = positions[-1]
        end_pos = curr_pos + tf.constant([[0, step_dist, 0]], tf.float32)
        positions.append(end_pos)

    return positions  # shape (N, 3)



def load_example_pano(base_dir, image_size=512):
    """
    Loads an example equirectangular panorama (1 RGB and 1 depth) using the same
    preprocessing as the original code.

    Args:
      base_dir: Directory where the pano images are stored.
      image_size: The height of the image (the width will be image_size * 2).

    Returns:
      A tuple (rgb, depth) where:
        rgb is a tensor of shape (1, image_size, image_size*2, 3) with dtype uint8.
        depth is a tensor of shape (1, image_size, image_size*2) with values in [0, 1].
    """
    # Load RGB image (PNG format).
    rgb_path = os.path.join(base_dir, 'cond_pano_rgb.png')
    print(f"Loading RGB image from {rgb_path}")
    with tf.io.gfile.GFile(rgb_path, 'rb') as f:
        rgb_img = tf.image.decode_png(f.read(), channels=3)
        rgb_img = tf.image.resize(rgb_img, (image_size, image_size * 2), method='bilinear')
        rgb_img = tf.cast(rgb_img, tf.uint8)

    # Load depth image.
    depth_path = os.path.join(base_dir, 'cond_pano_depth.png')
    with tf.io.gfile.GFile(depth_path, 'rb') as f:
        depth_img = tf.image.decode_png(f.read(), dtype=tf.uint16)       #NOTE: dtype=tf.uint16
        depth_img = tf.image.convert_image_dtype(depth_img, tf.float32)
        depth_img = tf.image.resize(depth_img, (image_size, image_size * 2), method='nearest')
        # Unnormalize into [0, 1].
        depth_img_max = tf.reduce_max(depth_img)
        depth_img_min = tf.reduce_min(depth_img)
        depth_img = (depth_img - depth_img_min) / (depth_img_max - depth_img_min)
        depth_img = tf.clip_by_value(depth_img, 0, 1)
        # Remove extra channel dimension if present.
        if depth_img.shape.ndims == 3:
            depth_img = depth_img[..., 0]

    # Expand dimensions to match the batch format.
    rgb_img = tf.expand_dims(rgb_img, axis=0)      # (1, image_size, image_size*2, 3)
    depth_img = tf.expand_dims(depth_img, axis=0)    # (1, image_size, image_size*2)
    # convert into (1, 3, image_size, image_size*2)
    rgb_img = tf.transpose(rgb_img, perm=[0, 3, 1, 2])    # (1, 3, image_size, image_size*2)

    return rgb_img, depth_img


class SE3DSInferenceEngine:
    def __init__(self, ckpt_path='data/se3ds_ckpt'):
        """
        Initializes the SE3DS inference engine by loading the pretrained model.
        This is done once so that multiple inferences can be run without reloading weights.

        Args:
          ckpt_path: Path to the model checkpoint.
        """
        self.config = model_config.get_config()
        self.config.depth_scale = 20.0         #HACK: still not sure
        self.config.ckpt_path = ckpt_path
        self.stoch_model = models.SE3DSModel(self.config)

    def init_bef_inference(self, start_rgb, start_depth):
        """
        Initializes the inference engine with the starting RGB and depth images,
        and the starting position.

        Args:
          start_rgb: Tensor of shape (1, H, W, 3) for the starting RGB image.
          start_depth: Tensor of shape (1, H, W) for the starting depth image.
          start_pos: Tensor of shape (1, 3) for the starting (x, y, z) position.
        """
        assert len(start_rgb.shape) == 4, f"start_rgb shape: {start_rgb.shape}"
        if start_rgb.shape[-1] != 3:
            # tfm from (1, 3, H, W) to (1, H, W, 3)
            start_rgb = tf.transpose(start_rgb, perm=[0, 2, 3, 1])

        self.stoch_model.reset_memory()
        B, H, W, C = start_rgb.shape
        start_seg = tf.zeros((1, H, W, 1), tf.int32)
        start_pos = tf.constant([[0, 0, 0]], tf.float32)
        self.stoch_model.add_to_memory(start_rgb, start_seg, start_depth, start_pos)

    def run_inference(self, end_pos, add_to_mem):
        """
        Runs inference on the pretrained SE3DS model given a new position,
        and outputs the predicted RGB, depth, and projected RGB images.

        Args:
          end_pos: Tensor of shape (1, 3) for the new (x, y, z) position.

        Returns:
          end_rgb: Numpy array of the predicted RGB image at end_pos.
          end_depth: Numpy array of the predicted depth image at end_pos.
          proj_rgb: Numpy array of the projected RGB image at end_pos.
        """
        outputs = self.stoch_model(end_pos, add_preds_to_memory=add_to_mem, sample_noise=False)
        end_rgb = outputs.pred_rgb.numpy().squeeze(0)
        end_depth = outputs.pred_depth.numpy().squeeze(0)
        proj_rgb = outputs.proj_rgb.numpy().squeeze(0)
        return end_rgb, end_depth, proj_rgb

    def batch_inference(self, end_pos_list, return_full=False):
        """
        Runs SE3DS inference for a *sequence* of camera positions.

        Parameters
        end_pos_list : Union[list[np.ndarray], list[tf.Tensor], tf.Tensor]
            Sequence of positions with shape (N, 3).  Each entry may be
            - a length‑3 NumPy/TF 1‑D array, or
            - a length‑3 tensor, or
            - a rank‑2 tensor of shape (N, 3).
        return_full : bool, optional
            If True, returns a dict containing RGB / depth / proj_rgb /
            cumulative distance / position history (handy for visualization).
            If False (default), returns only the list of RGB predictions.

        Returns
        list[np.ndarray] | dict
            • If *return_full* is False: list of N RGB predictions
            • If *return_full* is True:  dictionary with keys
              {rgb, depth, proj_rgb, pos, distance}
        """
        # --- Normalise input -------------------------------------------------
        if isinstance(end_pos_list, tf.Tensor):
            end_pos_seq = tf.reshape(end_pos_list, (-1, 3))
        else:
            # Convert list‑like => (N, 3) tensor
            end_pos_seq = tf.convert_to_tensor(end_pos_list, dtype=tf.float32)
            end_pos_seq = tf.reshape(end_pos_seq, (-1, 3))       # ensure rank 2
        N = end_pos_seq.shape[0]

        # --- Containers ------------------------------------------------------
        results = collections.defaultdict(list)
        total_dist = 0.0
        prev_pos   = None

        # --- Batched “loop” inference ---------------------------------------
        for i in tqdm(range(N), desc="Batch inference"):
            pos = tf.reshape(end_pos_seq[i], (1, 3))          # shape (1, 3)
            add_to_mem = False                              # first step priming only
            rgb, depth, proj = self.run_inference(pos, add_to_mem=add_to_mem)

            # Save outputs ----------------------------------------------------
            # chaneg return shape from (1, H, W, 3) to (1 3 H W)
            results['rgb'].append(np.transpose(rgb, (2, 0, 1)))
            results['depth'].append(depth)
            results['proj_rgb'].append(np.transpose(proj, (2, 0, 1)))
            results['pos'].append(pos.numpy())

            # Cumulative distance (optional but handy) ------------------------
            if prev_pos is not None:
                step_dist   = tf.norm(pos - prev_pos).numpy()
                total_dist += step_dist
            results['distance'].append(total_dist)
            prev_pos = pos

        return results if return_full else {'rgb': results['rgb'], 'depth': results['depth']}



if __name__ == '__main__':
    # The last argument is the w_fd we pass from main
    pipe_fd = int(sys.argv[-1])

    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--out_height", type=int, default=576)
    parser.add_argument("--out_width", type=int, default=1024)
    parser.add_argument("--log_dir", type=str, default="downstream/logs")
    parser.add_argument("--exp_id", type=str, default="04.19_se3dsdebug")
    # parser.add_argument("--num_frames", type=int, default=14)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ckpt_path", type=str,
                        default="data/se3ds_ckpt")
    # parser.add_argument("--weight_dtype", type=torch.dtype, default=torch.float16)
    args = parser.parse_args(sys.argv[1:-1])
    # args = parser.parse_args()

    log_path = osp.join(args.log_dir, f"{args.exp_id}", f"se3ds_worker",f"worker{args.device}.log")
    setup_logger(log_path)
    print(f"[se3ds_worker] All Args:\n {args}")
    image_size = args.height

    # Initialize the inference engine.
    engine = SE3DSInferenceEngine(ckpt_path=args.ckpt_path)
    print(f"[se3ds_worker] SE3DS model loaded successfully!")

    def do_some_tasks(input_dict: dict) -> dict:
        """
        Run SE3DS on every (save_dir, action_ids) pair and save the predicted videos.
        Expected keys in *input_dict*:  'b_action', 'save_dirs'.
        """
        b_action, save_dirs, b_image, return_objects = parse_input_data(input_dict)
        # assert b_image == None, "b_image will not be used as inputs for SE3DS"

        video_tensors = []                         # output buffer

        for save_dir, act_ids in zip(save_dirs, b_action):
            rgb, depth = load_example_pano(save_dir, image_size=image_size) #size (1, 3, H, W)

            frames: list[np.ndarray] = []          # collected RGB frames
            frames.append(rgb.numpy()[0])  # app (3, H, W)

            for sub_seq in split_actions(act_ids[1:], max_len=14):   # skip dummy
                first = sub_seq[0]

                # -------- yaw actions (2 = left, 3 = right) ----------------
                if first != 1:
                    direction  = 1 if first == 2 else -1
                    shift_px   = direction * int(UNIT_DEGREE * args.width / 360)

                    rgb   = tf.convert_to_tensor(rotate_by_shift(rgb.numpy(),   shift_px))
                    depth = tf.convert_to_tensor(rotate_by_shift(depth.numpy(), shift_px))

                    # record rotated view(s)
                    frames.append(rgb[0].numpy())
                    sub_seq = sub_seq[1:]
                    if not sub_seq:                # only a yaw, no forward move
                        continue

                # -------- forward rollout ----------------------------------
                # fwd_dist  = UNIT_DISTANCE * len(sub_seq)
                positions = tfm_action_to_coords(forward_steps=len(sub_seq),
                                                 forward_dist=UNIT_DISTANCE * len(sub_seq))

                engine.init_bef_inference(rgb, depth)
                out = engine.batch_inference(positions)      # keys: 'rgb', 'depth'

                # accumulate frames
                frames.extend(out["rgb"][1:])

                # carry state to the next loop
                rgb   = tf.expand_dims(tf.convert_to_tensor(out["rgb"][-1],   tf.uint8), 0)
                depth = tf.expand_dims(tf.convert_to_tensor(out["depth"][-1], tf.float32), 0)

            # pack to (T, 3, H, W) torch tensor
            video_tensor = torch.from_numpy(np.asarray(frames, dtype=np.uint8))
            # resize into out_width and out_height
            video_tensor = torch.nn.functional.interpolate(
                video_tensor,
                size=(args.out_height, args.out_width),
                mode='bilinear',
                align_corners=False,
            )

            video_tensors.append(video_tensor)

        video_tensors = torch.stack(video_tensors, dim=0).float() / 255.0  # (B, T, 3, H, W) float tensor in [0, 1]
        return process_output_dict(b_action, save_dirs, return_objects, video_tensors)

    worker_main(pipe_fd, do_some_tasks)

    # For debug:
    # -----------------------------------------------------------
    # base_dir = 'downstream/states/AEQA_04.28_aeqa_recompare/c5eTyR3Rxyh/Q23fb241e-989a-4299-a3fb-8d41f7156397/A001/igenex/PredA-0'
    # base_dir = 'downstream/states/AEQA_04.28_aeqa_recompare/c5eTyR3Rxyh/Q23fb241e-989a-4299-a3fb-8d41f7156397/A001/igenex/PredA-0'
    # input_dict = {
    #     # 'b_action': [[1] * 14] * 2,  # Example action sequence
    #     'b_action': [[1]*14] * 1,  # Example action sequence
    #     'save_dirs': [base_dir] * 1,
    # }
    # save_dirs_ = do_some_tasks(input_dict)
    # print(f"save_dirs_: {save_dirs_}")
