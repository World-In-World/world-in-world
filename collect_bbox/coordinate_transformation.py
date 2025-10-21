import numpy as np
import json
import os
import cv2
from torchvision.ops import masks_to_boxes

def habitat_camera_intrinsic(width, height, hfov = 90):
    width = width; height = height
    xc = (width - 1.) / 2.  #x-coordinate of the center of an image
    zc = (height - 1.) / 2. #y-coordinate of the center of an image
    f = (width / 2.) / np.tan(np.deg2rad(hfov / 2.))
    intrinsic_matrix = np.array([[f,0,xc],
                                 [0,f,zc],
                                 [0,0,1]],np.float32)
    return intrinsic_matrix

# intrinsic =
BLUE = (255, 0, 0)
RED = (0, 0, 255)

def draw_bbox_on_perspective_image(img_path, uv_points,
                                   line_color=RED, line_thickness=2,
                                   point_color=BLUE, point_radius=2):
    """
    Draws the edges of a 3D bounding box on a perspective image.

    Parameters:
    -----------
    img_path : str or np.ndarray
         Path to the perspective image or the image itself.
    uv_points : np.ndarray
         An (8, 2) array of pixel coordinates representing the 8 corners of the 3D bounding box.
         The ordering is assumed to be:
         [front_top_left, front_top_right, front_bottom_right, front_bottom_left,
          back_top_left, back_top_right, back_bottom_right, back_bottom_left]
    line_color : tuple, optional
         BGR color for the box edges (default is green).
    line_thickness : int, optional
         Thickness of the drawn lines (default is 2).
    point_color : tuple, optional
         BGR color for the corner points (default is red).
    point_radius : int, optional
         Radius of the corner points (default is 5).

    Returns:
    --------
    img_with_bbox : np.ndarray
         The image with the drawn bounding box.
    """
    # Load image if img_path is a string path
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Could not load image from the provided path.")
    else:
        img = img_path.copy()

    img_with_bbox = img.copy()

    # Define edges of the cuboid bounding box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # front face
        (4, 5), (5, 6), (6, 7), (7, 4),  # back face
        (0, 4), (1, 5), (2, 6), (3, 7)   # connecting edges
    ]

    # Draw lines for each edge
    for start_idx, end_idx in edges:
        pt1 = (int(uv_points[start_idx][0]), int(uv_points[start_idx][1]))
        pt2 = (int(uv_points[end_idx][0]), int(uv_points[end_idx][1]))
        cv2.line(img_with_bbox, pt1, pt2, line_color, line_thickness)

    # Draw corner points
    for uv in uv_points:
        u, v = int(uv[0]), int(uv[1])
        # Check bounds before drawing
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img_with_bbox, (u, v), point_radius, point_color, -1)


    # Optionally, save the image if img_path was provided as a path
    if isinstance(img_path, str):
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(img_path)
        out_name = img_name + "_bbox.png"
        output_path = os.path.join(img_dir, out_name)
        cv2.imwrite(output_path, img_with_bbox)
        print(f"Saved image with bounding box to {output_path}")

    return img_with_bbox


def world_to_camera_coordinates(corners_world, camera_matrix, intrinsic=None):
    """
    Transform bounding box corners from world coordinates to camera coordinates.

    Parameters:
    -----------
    corners_world : list
        List of 3D points representing the corners of the bounding box in world coordinates.
    camera_matrix : np.ndarray
        4x4 camera transformation matrix (RT matrix).

    Returns:
    --------
    corners_camera : np.ndarray
        Array of 3D points representing the corners of the bounding box in camera coordinates.
    """
    # Convert camera matrix to numpy array
    camera_matrix = np.array(camera_matrix)

    # Calculate the inverse of the camera matrix to go from world to camera coordinates
    camera_matrix_inv = np.linalg.inv(camera_matrix)

    # Convert corners to homogeneous coordinates (add 1 as fourth coordinate)
    corners_homogeneous = np.ones((len(corners_world), 4))
    corners_homogeneous[:, :3] = corners_world

    # Transform corners to camera coordinates
    corners_camera_homogeneous = np.dot(
        corners_homogeneous, camera_matrix_inv.T)

    # Convert back to 3D coordinates
    corners_camera = corners_camera_homogeneous[:,:3] / corners_camera_homogeneous[:, 3:4]

    if False:   #for debug
        depth_values = -corners_camera[:, 2]
        # # Extract intrinsic parameters
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        # Project points onto the image plane (np.array)
        filter_x = (((corners_camera[:, 0] * fx / depth_values) + cx) + 0.5).astype(np.int32)  # (N,)
        filter_z = (((-corners_camera[:, 1] * fy / depth_values) - cy + (cy * 2) - 1) + 0.5).astype(np.int32)   # (N,)

        uv = np.stack([filter_x, filter_z], axis=1)
        img_path = os.path.join(base_folder, "rgb_front.png")
        draw_bbox_on_perspective_image(img_path, uv)

    return corners_camera   #, depth_values.unsqueeze(1)


def cartesian_to_spherical(points_cartesian):
    """
    Transform 3D Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi)
    with respect to the camera.

    In camera coordinates:
    - The camera is at the origin (0, 0, 0)
    - The camera looks along the positive z-axis
    - The x-axis points to the right
    - The y-axis points up

    Spherical coordinates:
    - r: distance from the origin (camera) to the point
    - theta: azimuthal angle in the x-z plane (horizontal angle, from -π to π)
           0 is directly in front of the camera (positive z-axis)
           π/2 is to the right (positive x-axis)
           -π/2 is to the left (negative x-axis)
    - phi: polar angle from the y-axis (vertical angle, from 0 to π)
           0 is directly above (positive y-axis)
           π/2 is horizontal (in the x-z plane)
           π is directly below (negative y-axis)

    Parameters:
    -----------
    points_cartesian : np.ndarray or list
        Array of 3D points in Cartesian coordinates (x, y, z)

    Returns:
    --------
    points_spherical : np.ndarray
        Array of 3D points in spherical coordinates (r, theta, phi)
    """
    # Convert to numpy array for vector operations
    points = np.array(points_cartesian)

    # Initialize spherical coordinates array
    spherical = np.zeros_like(points)

    # Calculate r (distance from origin to point)
    spherical[:, 0] = np.sqrt(np.sum(points**2, axis=1))

    # Calculate theta (azimuthal angle in x-z plane)
    # Note: arctan2 returns angles in range [-π, π]
    # We use arctan2(x, z) because z is the camera's forward direction
    spherical[:, 1] = np.arctan2(points[:, 0], -points[:, 2])

    # Calculate phi (polar angle from y-axis)
    # Ensure the denominator (r) is not zero to avoid division by zero
    with np.errstate(invalid='ignore'):
        spherical[:, 2] = np.arccos(np.divide(
            points[:, 1],
            spherical[:, 0],
            out=np.zeros_like(points[:, 1]),
            where=spherical[:, 0] != 0
        ))

    # Handle any NaN values that might occur from division by zero
    spherical = np.nan_to_num(spherical)

    return spherical


def world_to_spherical(corners_world, camera_matrix, intrinsic=None):
    """
    Complete pipeline to transform from world coordinates to spherical camera coordinates.

    Parameters:
    -----------
    corners_world : list
        List of 3D points representing corners in world coordinates.
    camera_matrix : list
        4x4 camera transformation matrix (RT matrix).

    Returns:
    --------
    corners_camera : np.ndarray
        Array of 3D points in camera cartesian coordinates.
    corners_spherical : np.ndarray
        Array of 3D points in spherical coordinates (r, theta, phi).
    """
    # Step 1: Transform from world to camera coordinates
    corners_camera = world_to_camera_coordinates(
        corners_world, camera_matrix, intrinsic
    )

    # Step 2: Transform from cartesian to spherical coordinates
    corners_spherical = cartesian_to_spherical(corners_camera)

    return corners_camera, corners_spherical


def process_json(input_path, output_path):
    """
    Read input JSON, process the data, and save the results to output JSON.

    Parameters:
    -----------
    input_path : str
        Path to the input JSON file.
    output_path : str
        Path to save the output JSON file.
    """
    # Read the input JSON file
    with open(input_path, 'r') as f:
        json_data = json.load(f)

    # Extract corners and camera matrix
    corners_world = json_data["corners_coord"][0]
    camera_matrix = json_data["camera_world_coord"][0]

    # Run the transformation pipeline
    intrinsic = habitat_camera_intrinsic(width=512, height=512)
    corners_camera, corners_spherical = world_to_spherical(
        corners_world, camera_matrix,
    )

    # Convert numpy arrays to lists for JSON serialization
    corners_camera_list = corners_camera.tolist()
    corners_spherical_list = corners_spherical.tolist()

    # Add the results to the JSON data
    json_data["corners_camera_coord"] = [corners_camera_list]
    json_data["corners_spherical_coord"] = [corners_spherical_list]

    # Save the updated JSON to the output file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)


if __name__ == "__main__":
    # Define input and output paths
    # base_folder = "/home/jchen293/igenex_code/downstream/states/AR_03.24_bbox/EU6Fwq7SyZv/E134/A000"
    # base_folder = "/home/jchen293/igenex_code/downstream/states/AR_03.24_bbox/QUCTc6BB5sX/E002/A000"
    # base_folder = "/home/jchen293/igenex_code/downstream/states/AR_03.24_bbox/QUCTc6BB5sX/E014/A000"
    base_folder = "/home/jchen293/igenex_code/downstream/states/AR_03.24_bbox/X7HyMhZNoso/E028/A000"

    input_path = os.path.join(base_folder, "bbox.json")
    output_path = os.path.join(base_folder, "bbox_tfm.json")

    # Process the JSON data
    process_json(input_path, output_path)
