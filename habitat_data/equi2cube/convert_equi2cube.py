import torch
import itertools
import os
import numpy as np
from equilib import equi2cube
import warnings
from PIL import Image
import cv2
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F



def _open_as_PIL(img_path: str) -> Image.Image:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    img = Image.open(img_path)
    assert img is not None
    if img.getbands() == tuple("RGBA"):
        # NOTE: Sometimes images are RGBA
        img = img.convert("RGB")
    return img

def _open_as_cv2(img_path: str) -> np.ndarray:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    # FIXME: shouldn't use `imread` since it won't auto detect color space
    warnings.warn("Cannot handle color spaces other than RGB")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    assert img is not None
    return img


def load2numpy(
    img_path: str, dtype: np.dtype, is_cv2: bool = False
) -> np.ndarray:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    if is_cv2:
        # FIXME: currently only supports RGB
        img = _open_as_cv2(img_path)
    else:
        img = _open_as_PIL(img_path)
        img = np.asarray(img)

    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img = np.transpose(img, (2, 0, 1))

    # NOTE: Convert dtypes
    # if uint8, keep 0-255
    # if float, convert to 0.0-1.0
    dist_dtype = np.dtype(dtype)
    if dist_dtype in (np.float32, np.float64):
        img = img / 255.0
    img = img.astype(dist_dtype)

    return img

def load2torch(
    img_path: str, dtype: torch.dtype, is_cv2: bool = False
) -> torch.Tensor:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    if is_cv2:
        # FIXME: currently only supports RGB
        img = _open_as_cv2(img_path)
    else:
        img = _open_as_PIL(img_path)

    # NOTE: Convert dtypes
    # if uint8, keep 0-255
    # if float, convert to 0.0-1.0 (ToTensor)
    if dtype in (torch.float16, torch.float32, torch.float64):
        img = to_tensor(img)
        # FIXME: force typing since I have no idea how to change types in
        # PIL; also it's easier to change type using `type`; might be slower
        img = img.type(dtype)
        # NOTE: automatically adds channel for grayscale
    elif dtype == torch.uint8:
        img = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True))
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        else:
            img = img.permute((2, 0, 1)).contiguous()
        assert img.dtype == torch.uint8

    return img      #torch.Size([3, 2000, 4000])

def get_numpy_img(dtype: np.dtype = np.float32):
    path = os.path.join(IMG_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img


def get_torch_img(dtype: torch.dtype = torch.float32):
    path = os.path.join(IMG_ROOT, IMG_NAME)
    img = load2torch(path, dtype=dtype, is_cv2=False)
    return img


#%%

def stitch_cube_faces(cube_dict, face_order=None):
    """
    Stitch cube face images into a 3x2 grid image.

    The desired grid order is:
        Row 1: Back, Down, Front
        Row 2: Right, Left, Up

    This corresponds to the dictionary keys:
        ['B', 'D', 'F', 'R', 'L', 'U']

    Args:
        cube_dict (dict): Dictionary containing cube face images.
                          Expected keys are 'F', 'R', 'B', 'L', 'U', 'D'.
                          Each value is assumed to be an image in (3, H, W)
                          format (either a torch.Tensor or a numpy array).
        face_order (list, optional): Custom order of keys to use. Defaults to ['B', 'D', 'F', 'R', 'L', 'U'].

    Returns:
        PIL.Image: The stitched image in a 3x2 grid.
    """
    if face_order is None:
        face_order = ['B', 'D', 'F', 'R', 'L', 'U']

    # Convert each face to a PIL Image
    images = []
    for key in face_order:
        face_img = cube_dict[key]
        # Convert torch tensor to numpy if needed
        if isinstance(face_img, torch.Tensor):
            face_img = face_img.detach().cpu().numpy()
        # If image is in CHW format, convert it to HWC
        if face_img.ndim == 3 and face_img.shape[0] == 3:
            face_img = face_img.transpose(1, 2, 0)
        # If image dtype is not uint8, assume it's in [0,1] and convert
        if face_img.dtype != np.uint8:
            face_img = np.clip(face_img, 0, 1)
            face_img = (face_img * 255).astype(np.uint8)
        pil_img = Image.fromarray(face_img)
        images.append(pil_img)

    # Assume all faces have the same dimensions
    img_width, img_height = images[0].size

    # Create a canvas for a 3x2 grid
    grid_cols, grid_rows = 3, 2
    canvas_width = grid_cols * img_width
    canvas_height = grid_rows * img_height
    canvas = Image.new('RGB', (canvas_width, canvas_height), color=(0, 0, 0))

    # Paste the images into the canvas following the order
    for idx, img in enumerate(images):
        row = idx // grid_cols
        col = idx % grid_cols
        x = col * img_width
        y = row * img_height
        canvas.paste(img, (x, y))

    return canvas

#%%
def test_torch_single(
    w_face: int,
    cube_format: str,
    z_down: bool,
    mode: str,
    dtype: torch.dtype,
) -> None:
    torch_single(
        w_face=w_face,
        cube_format=cube_format,
        z_down=z_down,
        mode=mode,
        dtype=dtype,
    )

def torch_single(
    w_face: int,
    cube_format: str,
    z_down: bool,
    mode: str,
    dtype: torch.dtype,
) -> None:
    # just a single image and rotation dictionary
    img = get_torch_img(dtype=dtype)    #torch.Size([3, 2000, 4000])
    rot = {
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }

    out = equi2cube(
        equi=img_.clone(),
        rots=rot,
        w_face=w_face,
        cube_format=cube_format,
        mode=mode,
        z_down=z_down,
    )

    if cube_format == "dice":
        assert out.shape == (3, w_face * 3, w_face * 4)
        assert out.dtype == dtype
    elif cube_format == "horizon":
        assert out.shape == (3, w_face, w_face * 6)
        assert out.dtype == dtype
    elif cube_format == "dict":
        assert isinstance(out, dict)
        for key in ["F", "R", "B", "L", "U", "D"]:
            assert out[key].shape == (3, w_face, w_face)
            assert out[key].dtype == dtype
        stitched_img = stitch_cube_faces(out)
        stitched_img.save('stitched_cube.png')
    elif cube_format == "list":
        assert isinstance(out, list)
        for cube in out:
            assert cube.shape == (3, w_face, w_face)
            assert cube.dtype == dtype



def convert_equi2cube(
    img: torch.Tensor,      # range [0, 1], shape (b, 3, h, w)
    cube_format='list',
    mode='bilinear',
    dtype=torch.float32,
    z_down=False,
    w_face=224,
    rot = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0,},
):
    """
    Input:
        - img: torch.Tensor, shape (b, 3, h, w)
    Return:
        - out_: torch.Tensor, shape (b, 6, 3, w_face, w_face)
    """
    assert len(img.shape) == 4, "input must be dim=4"

    B, C, H, W = img.shape
    # to avoid the W != 2 * H for equi2cube:
    if W != 2 * H:
        if isinstance(img, torch.Tensor):
            img = F.interpolate(img, size=(H, 2 * H), mode="bilinear")
        else:
            raise ValueError("W != 2 * H for equi2cube")

    out = equi2cube(
        equi=img.clone(),
        rots=[rot]*img.shape[0],
        w_face=w_face,
        cube_format=cube_format,
        mode=mode,
        z_down=z_down,
    )

    # assert cube.shape == (3, w_face, w_face)
    out_ = torch.empty((len(out), 6, 3, w_face, w_face), dtype=dtype)
    for i, cubes in enumerate(out):
        out_[i] = torch.stack(cubes, dim=0)

    return out_



def main():
    # Parameter sets (same as in the pytest parameterization)
    w_faces = [512]
    cube_formats = ["dict",] #"dice",   "list"
    z_downs = [False]
    modes = [ "bilinear", ]    #"nearest","bicubic"
    dtypes = [torch.float32]

    # Loop over all parameter combinations
    for w_face, cube_format, z_down, mode, dtype in itertools.product(
        w_faces, cube_formats, z_downs, modes, dtypes
    ):
        print("=" * 40)
        print(f"Running test with parameters: w_face={w_face}, cube_format={cube_format}, "
              f"z_down={z_down}, mode={mode}, dtype={dtype}")
        test_torch_single(w_face, cube_format, z_down, mode, dtype)

if __name__ == '__main__':
    IMG_ROOT = "/data/jieneng/visual_navigation/InstructNav/data_collect/test_images"
    SAVE_ROOT = "./"
    IMG_NAME = "test.png"
    main()
