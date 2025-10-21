import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from os import path as osp
import os


def spherical_to_equirectangular(spherical_coords, img_width, img_height):
    """
    Vectorized version of spherical_to_equirectangular that processes
    an Nx3 array of spherical coords: [r, theta, phi].
    Returns an Nx2 array of pixel coordinates in the equirectangular image.

    Args:
        spherical_coords: (N, 3) array of spherical coordinates [r, theta, phi].
        img_width: width of the equirectangular panorama
        img_height: height of the equirectangular panorama

    Returns:
        (N, 2) array of integer pixel coordinates (x, y).
    """
    # spherical_coords: [r, theta, phi] for each row.
    # We'll ignore 'r' in the projection to equirectangular (distance doesn't affect az/el).
    # Make sure we're dealing with a NumPy array:
    coords = np.array(spherical_coords, dtype=float)

    # Extract each component
    # r = coords[:, 0]  # "r" is not used directly in equirectangular projection
    theta = coords[:, 1]
    phi   = coords[:, 2]

    # Step 1: Shift theta by pi and wrap in [0, 2*pi)
    theta = theta + np.pi
    theta = np.mod(theta, 2 * np.pi)

    # Step 2: Adjust phi if < 0 or > pi
    #   If phi < 0  => phi = |phi|
    #   If phi > pi => phi = 2*pi - phi
    mask_neg = phi < 0
    phi[mask_neg] = -phi[mask_neg]

    mask_large = phi > np.pi
    phi[mask_large] = 2 * np.pi - phi[mask_large]

    # Step 3: Convert to pixel coordinates
    #   x = (theta / 2π) * width
    #   y = (phi   / π)  * height
    x = (theta / (2 * np.pi)) * img_width
    y = (phi   / np.pi)       * img_height

    # Step 4: Convert to integers (pixel coords) and stack
    x = x.astype(int)
    y = y.astype(int)

    return np.column_stack((x, y))


def draw_bbox_on_panorama(
    input_image_path=None,
    json_path='out.json',
    output_path='bbox_visualization.png',
    draw_connections=False
):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # corners is presumably a list of shape (N, 3), i.e. [ [r,theta,phi], [r,theta,phi], ... ]
    corners = data['corners_spherical_coord'][0]

    # Load or create a placeholder image
    if input_image_path:
        img = cv2.imread(input_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_height = 1000
        img_width = 2000
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    img_height, img_width = img.shape[:2]

    # --- NEW: Vectorized spherical->equirectangular ---
    corners_arr = np.array(corners, dtype=float)  # shape (N, 3)
    pixel_coords = spherical_to_equirectangular(
        corners_arr, img_width, img_height
    )
    # pixel_coords now is (N, 2)
    draw_bbox_from_spherical_coords(output_path, img, pixel_coords)
    return img


def draw_bbox_from_spherical_coords(output_path, img, pixel_coords, draw_connections=True):
    """
    Inputs:
        - img: The image on which to draw the bounding box, shape (H, W, 3), and color format (RGB).
        - pixel_coords: The pixel coordinates of the corners in the equirectangular image.
        - draw_connections: If True, draw lines between corners (if 8 corners).
    """
    img_height, img_width= img.shape[:2]
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255)
    ]
    # Optionally draw lines between corners (if 8 corners):
    if draw_connections and len(pixel_coords) == 8:
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for edge in edges:
            pt1 = tuple(pixel_coords[edge[0]])
            pt2 = tuple(pixel_coords[edge[1]])
            cv2.line(img, pt1, pt2, (255, 255, 255), 2)

    # Draw corners
    for i, (px, py) in enumerate(pixel_coords):
        # Guard in case outside image bounds
        if not (0 <= px < img_width and 0 <= py < img_height):
            continue
        cv2.circle(img, (px, py), 5, colors[i % len(colors)], -1)
        cv2.putText(
            img, str(i), (px - 5, py + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 1, cv2.LINE_AA
        )

    # Display / save
    cv2.imwrite(output_path, img)

    print(f"Visualization saved to {output_path}")
    return img


if __name__ == "__main__":
    # draw_bbox_on_panorama(input_image_path='your_panorama.jpg')
    # base_folder = "/home/jchen293/igenex_code/downstream/states/AR_03.24_bbox/EU6Fwq7SyZv/E134/A000"
    # base_folder = "/home/jchen293/igenex_code/downstream/states/AR_03.24_bbox/QUCTc6BB5sX/E002/A000"
    # base_folder = "/home/jchen293/igenex_code/downstream/states/AR_03.24_bbox/QUCTc6BB5sX/E014/A000"
    base_folder = "/home/jchen293/igenex_code/downstream/states/AR_03.24_bbox/X7HyMhZNoso/E028/A000"

    tfm_path = os.path.join(base_folder, 'bbox_tfm.json')
    # draw_bbox_on_panorama(json_path=tfm_path)
    draw_bbox_on_panorama(input_image_path=osp.join(osp.dirname(tfm_path), 'rgb.png'),
                        json_path=tfm_path)