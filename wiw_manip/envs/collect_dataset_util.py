import os
import re
import time
import numpy as np
from pathlib import Path
from typing import Union, Dict, Tuple, List
from wiw_manip.planner.utils.saver import save_video
from amsolver.backend.const import *
from amsolver.backend import utils
from PIL import Image
import pickle


# ----------------------------------------------------------------------------- #
# Helper: create the next free  episodeN  directory under   .../variation0/episodes
# ----------------------------------------------------------------------------- #
_EP_RE = re.compile(r"^episode(\d+)$")
def create_unique_episode_dir(base_dir: Path, max_retry: int = 30) -> Path:
    """
    Atomically create e.g.  episode0 / episode1 / …  and return the full path.

    Parameters
    ----------
    base_dir : Path
        The .../variation0/episodes   folder.  Will be created if missing.
    max_retry : int
        Fails if we collide  `max_retry`  times in a row (very unlikely).
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    # Discover the highest existing index once; +1 is our first candidate.
    existing = [
        int(m.group(1))
        for d in base_dir.iterdir()
        if d.is_dir() and (m := _EP_RE.match(d.name))
    ]
    next_idx = max(existing) + 1 if existing else 0

    for _ in range(max_retry):
        target = base_dir / f"episode{next_idx}"
        try:
            target.mkdir(exist_ok=False)        # <-- atomic
            return target                       # success
        except FileExistsError:
            # Another process was faster; try the next integer
            next_idx += 1
            time.sleep(0.02)                    # tiny back-off

    raise RuntimeError(
        f"Could not create a unique episode directory in {base_dir} "
        f"after {max_retry} attempts."
    )


def save_demo(demo: List, example_path: str, also_save_video=False, save_mask_depth=False, interval=2) -> None:
    """
    Parameters
    demo
        List of Observation objects in temporal order.
    example_path
        Root directory for this demo run.
    also_save_video
        If True, writes front_rgb_video.mp4 at 7 fps.
    save_mask_depth
        If False, depth and mask images are not written to disk.
    interval
        Down-sampling factor in time: 1→save every frame, 2→every 2nd frame, …
    """
    # 1)  De-duplicate identical Observation objects                     #
    seen_ids, unique_demo, removed_idx = set(), [], []
    for i, obs in enumerate(demo):
        if id(obs) in seen_ids or obs.front_rgb is None:
            removed_idx.append(i)
            continue
        seen_ids.add(id(obs))
        unique_demo.append(obs)

    if removed_idx:
        print(
            f"[save_demo] Removed duplicates/None frames "
            f"({len(removed_idx)}/{len(demo)}).  Kept {len(unique_demo)}."
        )
    demo = unique_demo

    # 2)  Temporal sub-sampling                                          #
    demo = [obs for i, obs in enumerate(demo) if i % interval == 0]
    if not demo:
        raise RuntimeError("Sampling interval removed every frame – nothing to save")

    # 3)  Prepare directories                                            #
    front_rgb_path   = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path  = os.path.join(example_path, FRONT_MASK_FOLDER)

    os.makedirs(front_rgb_path, exist_ok=True)
    if save_mask_depth:
        os.makedirs(front_depth_path, exist_ok=True)
        os.makedirs(front_mask_path, exist_ok=True)

    # 4)  Save per-frame images & strip heavy tensors                    #
    all_rgbs = []
    for i, obs in enumerate(demo):
        # -- RGB --------------------------------------------------------
        front_rgb = Image.fromarray(obs.front_rgb)
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        all_rgbs.append(front_rgb)

        # -- Depth & mask ----------------------------------------------
        if save_mask_depth:
            front_depth = utils.float_array_to_rgb_image(
                obs.front_depth, scale_factor=DEPTH_SCALE
            )
            front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

            front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
            front_mask.save(os.path.join(front_mask_path,  IMAGE_FORMAT % i))

        # -- Strip heavyweight tensors ---------------------------------
        for cam in ("left_shoulder", "right_shoulder", "overhead",
                    "wrist", "front"):
            for suffix in ("rgb", "depth", "point_cloud", "mask"):
                setattr(obs, f"{cam}_{suffix}", None)

    # 5)  Pickle low-dimensional data                                    #
    with open(os.path.join(example_path, LOW_DIM_PICKLE), "wb") as f:
        pickle.dump(demo, f)

    # 6)  Optional stitched video                                        #
    if also_save_video:
        save_video(
            all_rgbs, fps=7,
            save_path=os.path.join(example_path, "front_rgb_video.mp4"),
        )
    print(f"Demo successfully saved to: {example_path}", flush=True)
