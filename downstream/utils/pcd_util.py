import numpy as np
import open3d as o3d
import torch


def associate_object_entities(ref_entities, eval_entities):
    """
    Associates newly detected object entities with existing reference entities based on point cloud overlaps.
    Parameters:
        ref_entities (List[Dict]): A list of existing reference entities, each containing:
            - 'class' (str): The class label of the object.
            - 'pcd' (PointCloud): The point cloud of the object.
            - 'confidence' (float): The confidence score of the detection.
        eval_entities (List[Dict]): A list of newly evaluated entities to be associated, each containing:
            - 'class' (str): The class label of the object.
            - 'pcd' (PointCloud): The point cloud of the object.
            - 'confidence' (float): The confidence score of the detection.
    Returns:
        List[Dict]: The updated list of reference entities after association and merging.
    """
    for entity in eval_entities:
        if len(ref_entities) == 0:
            ref_entities.append(entity)
            continue

        overlap_score = []
        eval_pcd = entity["pcd"]
        for ref_entity in ref_entities:
            if eval_pcd.point.positions.shape[0] == 0:
                break

            cdist = pointcloud_distance(eval_pcd, ref_entity["pcd"])
            overlap_condition = cdist < 0.1
            nonoverlap_condition = overlap_condition.logical_not()
            eval_pcd = eval_pcd.select_by_index(
                o3d.core.Tensor(
                    nonoverlap_condition.cpu().numpy(), device=ref_entity["pcd"].device
                ).nonzero()[0]
            )
            overlap_score.append(
                (overlap_condition.sum() / (overlap_condition.shape[0] + 1e-6))
                .cpu().numpy()
            )
        max_overlap_score = np.max(overlap_score)
        arg_overlap_index = np.argmax(overlap_score)

        if max_overlap_score < 0.25:
            entity["pcd"] = eval_pcd
            ref_entities.append(entity)
        else:  # merges the point clouds of the current entity and the reference entity with the highest overlap score:
            argmax_entity = ref_entities[arg_overlap_index]
            argmax_entity["pcd"] = gpu_merge_pointcloud(
                argmax_entity["pcd"], eval_pcd
            )
            if (
                argmax_entity["pcd"].point.positions.shape[0]
                < entity["pcd"].point.positions.shape[0]
            ):
                argmax_entity["class"] = entity["class"]

            ref_entities[arg_overlap_index] = argmax_entity
    return ref_entities


def gpu_pointcloud_from_array(points, colors, device):
    device = o3d.core.Device(device.upper())
    pointcloud = o3d.t.geometry.PointCloud(device)  # points shape is (307200, 3)
    pointcloud.point.positions = o3d.core.Tensor(
        points, dtype=o3d.core.Dtype.Float32, device=device
    )
    pointcloud.point.colors = o3d.core.Tensor(
        colors.astype(np.float32) / 255.0, dtype=o3d.core.Dtype.Float32, device=device
    )
    return pointcloud


def get_pointcloud_from_depth_mask(
    depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray
):
    """
    The get_pointcloud_from_depth_mask function generates a point cloud from depth and mask images.
    Return:
        point_values (np.ndarray, (N, 3)): The generated point cloud coordinates in the format (x, z, -y).
    """
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    filter_z, filter_x = np.where((depth > 0) & (mask > 0))
    depth_values = depth[filter_z, filter_x]
    pixel_z = (
        (depth.shape[0] - 1 - filter_z - intrinsic[1][2])
        * depth_values
        / intrinsic[1][1]
    )
    pixel_x = (filter_x - intrinsic[0][2]) * depth_values / intrinsic[0][0]
    pixel_y = depth_values
    point_values = np.stack([pixel_x, pixel_z, -pixel_y], axis=-1)
    return point_values


def gpu_cluster_filter(pointcloud, eps=0.3, min_points=20):
    """
    The gpu_cluster_filter function filters a point cloud to retain only the largest
    cluster of points. It performs DBSCAN clustering on the point cloud, identifies the
    largest cluster, and returns the points belonging to that cluster.
    """
    labels = pointcloud.cluster_dbscan(
        eps=eps, min_points=min_points, print_progress=False
    )
    numpy_labels = labels.cpu().numpy()
    unique_labels = np.unique(numpy_labels)
    largest_cluster_label = max(unique_labels, key=lambda x: np.sum(numpy_labels == x))
    largest_cluster_pc = pointcloud.select_by_index(
        (labels == largest_cluster_label).nonzero()[0]
    )
    return largest_cluster_pc

def gpu_merge_pointcloud(pcdA, pcdB):
    if pcdA.is_empty():
        return pcdB
    if pcdB.is_empty():
        return pcdA
    return pcdA + pcdB

def pointcloud_distance(pcdA, pcdB, device):
    try:
        pointsA = torch.tensor(pcdA.point.positions.cpu().numpy(), device=device)
        pointsB = torch.tensor(pcdB.point.positions.cpu().numpy(), device=device)
    except:
        pointsA = torch.tensor(np.array(pcdA.points), device=device)
        pointsB = torch.tensor(np.array(pcdB.points), device=device)
    cdist = torch.cdist(pointsA, pointsB)
    min_distances1, _ = cdist.min(dim=1)
    return min_distances1
