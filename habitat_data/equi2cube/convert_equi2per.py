#!/usr/bin/env python3

import argparse
import os.path as osp
import time
from typing import Union

import cv2

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from equilib import Equi2Pers
import torch
import torch.nn.functional as F

RESULT_PATH = "./"
DATA_PATH = "./"


def preprocess(
    img: Union[np.ndarray, Image.Image], is_cv2: bool = False
) -> np.ndarray:
    """Preprocesses image"""
    if isinstance(img, np.ndarray) and is_cv2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(img, Image.Image):
        # Sometimes images are RGBA
        img = img.convert("RGB")
        img = np.asarray(img)
    assert len(img.shape) == 3, "input must be dim=3"
    assert img.shape[-1] == 3, "input must be H W C"
    img = np.transpose(img, (2, 0, 1))
    return img


def test_video(
    path: str, h_pers: int = 480, w_pers: int = 640, fov_x: float = 90.0
) -> None:
    """Test video"""
    # Rotation:
    pi = np.pi
    inc = pi / 180
    roll = 0  # -pi/2 < a < pi/2
    pitch = 0  # -pi < b < pi
    yaw = 0

    # Initialize equi2pers
    equi2pers = Equi2Pers(
        height=h_pers, width=w_pers, fov_x=fov_x, mode="bilinear"
    )

    times = []
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()

        rot = {"roll": roll, "pitch": pitch, "yaw": yaw}

        if not ret:
            break

        s = time.time()
        equi_img = preprocess(frame, is_cv2=True)
        pers_img = equi2pers(equi=equi_img, rots=rot)
        pers_img = np.transpose(pers_img, (1, 2, 0))
        pers_img = cv2.cvtColor(pers_img, cv2.COLOR_RGB2BGR)
        e = time.time()
        times.append(e - s)

        # cv2.imshow("video", pers)

        # change direction `wasd` or exit with `q`
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        if k == ord("w"):
            roll -= inc
        if k == ord("s"):
            roll += inc
        if k == ord("a"):
            pitch += inc
        if k == ord("d"):
            pitch -= inc

    cap.release()
    cv2.destroyAllWindows()

    print(sum(times) / len(times))
    x_axis = list(range(len(times)))
    plt.plot(x_axis, times)
    save_path = osp.join(RESULT_PATH, "times_equi2pers_numpy_video.png")
    plt.savefig(save_path)

import time
def test_image(
    path: str, h_pers: int = 512, w_pers: int = 512, fov_x: float = 90.0
) -> None:
    """Test single image"""
    # Rotation:
    rot = {
        "roll": 0,  #
        "pitch": 0,  # vertical
        "yaw": 0,  # horizontal
    }

    # Initialize equi2pers
    equi2pers = Equi2Pers(
        height=h_pers, width=w_pers, fov_x=fov_x, mode="bilinear", clip_output=False,
    )

    # Open Image
    equi_img = Image.open(path)
    equi_img = preprocess(equi_img)

    pers_img = equi2pers(equi_img, rots=rot)    # equi_img is (c, h, w)

    pers_img = np.transpose(pers_img, (1, 2, 0))# pers_img is (h, w, c)
    pers_img = Image.fromarray(pers_img)

    # pers_path = osp.join(RESULT_PATH, "output_equi2pers_numpy_image.jpg")
    out_path = osp.join(RESULT_PATH, f"output_per-{time.time()}_.png")
    pers_img.save(out_path)


def convert_equi2per(
    equi_img, h_pers: int = 512, w_pers: int = 512, fov_x: float = 90.0
) -> np.ndarray:
    # assert equi_img is (b, c, h, w)
    assert len(equi_img.shape) == 4, f"input must be dim=4, but got shape {equi_img.shape}"

    B, C, H, W = equi_img.shape
    # to avoid the W != 2 * H for equi2cube:
    if W != 2 * H:
        if isinstance(equi_img, torch.Tensor):
            equi_img = F.interpolate(equi_img, size=(H, 2 * H), mode="bilinear")
        elif isinstance(equi_img, np.ndarray):
            resized_imgs = np.empty((B, C, H, 2 * H), dtype=equi_img.dtype)
            for b in range(B):
                for c in range(C):
                    # cv2.resize expects size as (width, height)
                    resized_imgs[b, c] = cv2.resize(
                        equi_img[b, c], (2 * H, H), interpolation=cv2.INTER_LINEAR
                    )
            equi_img = resized_imgs
        else:
            raise ValueError("W != 2 * H for equi2cube")

    rot = [{
        "roll": 0,  #
        "pitch": 0,  # vertical
        "yaw": 0,  # horizontal, + means rotate left, - means right
    }] * B

    equi2pers = Equi2Pers(
        height=h_pers, width=w_pers, fov_x=fov_x, mode="bilinear", clip_output=False,
    )

    pers_img = equi2pers(equi_img, rots=rot)    # equi_img is (b, c, h, w)
    return pers_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--data", nargs="?", default=None, type=str)
    args = parser.parse_args()

    # Variables:
    fov_x = 105
    h_pers = 576
    w_pers = int((fov_x / 90)*h_pers)

    data_path = args.data
    if args.video:
        if data_path is None:
            data_path = osp.join(DATA_PATH, "R0010028_er_30.MP4")
        assert osp.exists(data_path)
        test_video(data_path, h_pers, w_pers, fov_x)
    else:
        if data_path is None:
            data_path = osp.join(DATA_PATH, "/weka/scratch/ayuille1/jzhan423/igenex_code/data/datasets__/06.05_hm3d_collision_data_/812QqCky3T7.basis.glb/traj-0/waypoint-3/step-16_type-rgb.png")
        assert osp.exists(data_path)
        test_image(data_path, h_pers, w_pers, fov_x)


if __name__ == "__main__":
    main()
