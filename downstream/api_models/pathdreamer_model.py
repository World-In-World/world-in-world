import argparse, sys, os, os.path as osp, collections, math, pickle
import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

from pathdreamer.models import pathdreamer_config, pathdreamer_models
from pathdreamer.utils   import utils
from pathdreamer         import constants

from utils.logger              import setup_logger
from downstream.utils.saver    import save_predict
from downstream.utils.worker_manager import (
    read_pickled_data,
    write_pickled_data,
    read_pickled_data_non_blocking,
    receiver_for_worker,
    worker_main,
)
from downstream.api_models.se3ds_model import tfm_action_to_coords, count_parameters_tf

# Utility functions
_CMAP = utils.create_label_colormap()





def load_example_pano(base_dir: str, image_size: int = 512):
    """
    Load RGB / depth / seg frames saved by previous pipeline stages.
    File names must be:
        cond_pano_rgb.png   (uint8 PNG, H×W×3)
        cond_pano_depth.png (uint16 PNG, H×W×1, *un-scaled depth*)
        cond_pano_seg.png   (uint8 / indexed-color PNG, H×W×3)   ← optional
    """
    h, w = image_size, image_size * 2

    def _read_img(path, ch):
        if ch == 1:
            dtype = tf.uint16
        else:
            dtype = tf.uint8

        with tf.io.gfile.GFile(path, 'rb') as f:
            # changed to read the img by the extension
            if path.endswith('.jpg') or path.endswith('.jpeg'):
                img = tf.image.decode_jpeg(f.read(), channels=ch)
            elif path.endswith('.png'):
                img = tf.image.decode_png(f.read(), channels=ch, dtype=dtype)
        img = tf.image.resize(img, (h, w),
                              method='bilinear' if ch == 3 else 'nearest')
        return img

    # RGB
    rgb = tf.cast(_read_img(osp.join(base_dir, 'cond_pano_rgb.png'), 3), tf.uint8)

    # Depth → float32 in [0,1]
    d16 = _read_img(osp.join(base_dir, 'cond_pano_depth.png'), 1)
    d16 = tf.image.convert_image_dtype(d16, tf.float32)
    # get the max value of the depth
    d16 = (d16 - tf.reduce_min(d16)) / (tf.reduce_max(d16) - tf.reduce_min(d16))
    depth = tf.squeeze(d16, -1)

    # Seg
    seg_path = osp.join(base_dir, 'cond_pano_seg.png')
    # is not exits seg, input a zero seg
    if not tf.io.gfile.exists(seg_path):
        seg_rgb = tf.zeros((h, w, 3), dtype=tf.uint8)
        seg = tf.zeros((h, w), dtype=tf.int32)
    else:
        # Read and convert to label map
        seg_rgb = tf.cast(_read_img(seg_path, 3), tf.uint8)
        seg = utils.cmap_to_label(seg_rgb, _CMAP)
        seg = tf.cast(seg, tf.int32)

    # Add batch dim
    rgb   = rgb[None, ...]
    depth = depth[None, ...]              # (1, H, W)
    seg   = seg[None, ..., None]          # (1, H, W, 1)

    return rgb, seg, depth


# Inference engine
class PathdreamerEngine:
    def __init__(self, structure_ckpt: str, image_ckpt: str):
        cfg = pathdreamer_config.get_config()
        cfg.depth_scale = 7.0
        cfg.ckpt_path      = structure_ckpt
        cfg.spade_ckpt_path = image_ckpt
        self.model = pathdreamer_models.PathdreamerModel(cfg)

    # ──────────────────────────────────────────────────────────────────────
    def init_bef_inference(self, rgb, seg, depth, start_pos=None):
        """Prime Pathdreamer with the first observation."""
        if start_pos is None:
            start_pos = tf.constant([[0., 0., 0.]], tf.float32)
        self.model.reset_memory()
        self.model.add_to_memory(rgb, seg, depth, start_pos)

    # ──────────────────────────────────────────────────────────────────────
    def infer(self, end_pos, add_to_mem):
        """Single-step prediction -> RGB/Depth/Semantic."""
        outs = self.model(end_pos,
                          add_preds_to_memory=add_to_mem,
                          sample_noise=False)
        rgb   = outs.pred_rgb.numpy().squeeze(0)      # H×W×3
        depth = outs.pred_depth.numpy().squeeze(0)
        seg   = outs.pred_semantic.numpy().squeeze(0)
        return rgb, depth, seg

    # ──────────────────────────────────────────────────────────────────────
    def batch_inference(self, positions, return_full=False):
        """
        positions: (N,3) tensor / list. Returns list of RGB or full dict.
        """
        pos_seq = tf.convert_to_tensor(positions, tf.float32)
        pos_seq = tf.reshape(pos_seq, (-1, 3))
        N = pos_seq.shape[0]

        res = collections.defaultdict(list)
        total_dist, prev = 0., None

        for i in tqdm(range(N), desc="Pathdreamer inference"):
            pos = tf.reshape(pos_seq[i], (1, 3))
            rgb, depth, seg = self.infer(pos, add_to_mem=(i > 0))

            res['rgb'].append(np.transpose(rgb, (2,0,1)))   # C×H×W
            res['depth'].append(depth)
            res['seg'].append(seg)
            res['pos'].append(pos.numpy())

            if prev is not None:
                total_dist += tf.norm(pos - prev).numpy()
            res['distance'].append(total_dist)
            prev = pos

        return res if return_full else res['rgb']


# Worker entry-point
def build_argparser():
    p = argparse.ArgumentParser("Pathdreamer socket worker")
    p.add_argument("--width",  type=int, default=1024)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--out_width", type=int, default=1024)
    p.add_argument("--out_height", type=int, default=576)
    p.add_argument("--log_dir", type=str,  default="downstream/logs")
    p.add_argument("--exp_id", type=str,  default="04.28_pathdreamer_debug")
    p.add_argument("--num_frames", type=int, default=14)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ckpt_path", type=str,
                   default="data/ckpt")
    return p

if __name__ == "__main__":
   # Last CLI arg is the write FD
    pipe_fd = int(sys.argv[-1])
    args = build_argparser().parse_args(sys.argv[1:-1])
    # For debug:
    # args = build_argparser().parse_args()
    args.world_model_name = "pathdreamer"

    args.structure_ckpt = osp.join(args.ckpt_path, "structure_gen_ckpt")
    args.image_ckpt     = osp.join(args.ckpt_path, "image_gen_ckpt")
    log_path = osp.join(args.log_dir, args.exp_id,
                        "pathdreamer_worker", f"worker{args.device}.log")
    setup_logger(log_path)
    print(f"[pathdreamer_worker] Args:\n{args}")

    # Initialise engine
    engine = PathdreamerEngine(args.structure_ckpt, args.image_ckpt)
    print("[pathdreamer_worker] Pathdreamer loaded!")

    # ────────────────────────────────────────────────────────────────────
    def process_task(input_dict):
        """
        Expects keys:  • b_action : list[list[int]]
                       • save_dirs: list[str]
        Returns dict  {save_dirs: …} identical to SE3DS worker.
        """
        b_action  = input_dict["b_action"]
        save_dirs = input_dict["save_dirs"]

        vid_tensors = []
        img_h = args.height

        for path, acts in zip(save_dirs, b_action):
            # 1) first observation
            rgb0, seg0, dep0 = load_example_pano(path, image_size=img_h)

            # 2) coordinates from actions (same assumptions as SE3DS)
            assert (np.array(acts[1:]) == 1).all(), "Only ‘forward’ supported."
            coords = tfm_action_to_coords(forward_steps=1)

            # 3) inference
            engine.init_bef_inference(rgb0, seg0, dep0)
            vid = engine.batch_inference(coords)
            vid_tensors.append(torch.from_numpy(np.array(vid)))  # F×3×H×W
            # count_parameters_tf(engine.model.model, verbose=True)
            # sys.exit(0)
        out_paths = save_predict(vid_tensors, b_action, save_dirs)
        return {"save_dirs": out_paths}

    # ────────────────────────────────────────────────────────────────────
    worker_main(pipe_fd, process_task)

    # For debug:
    # -----------------------------------------------------------
    # export LD_LIBRARY_PATH="/weka/scratch/ayuille1/jzhan423/software/miniconda3/envs/se3ds/lib:${LD_LIBRARY_PATH}"
    # base_dir = 'downstream/states/AEQA_04.28_aeqa_recompare/c5eTyR3Rxyh/Q23fb241e-989a-4299-a3fb-8d41f7156397/A001/igenex/PredA-0'
    # input_dict = {
    #     'b_action': [[1] * 14] * 2,  # Example action sequence
    #     'save_dirs': [base_dir] * 2,
    # }
    # save_dirs_ = process_task(input_dict)
    # print(f"save_dirs_: {save_dirs_}")
