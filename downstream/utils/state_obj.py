import numpy as np
import open3d as o3d
from jaxtyping import Float, Int32, UInt8, Bool
from numpy.typing import NDArray

from habitat_sim.utils.common import d3_40_colors_rgb
from data_filtering.pcd_reproject import (
    camera_to_world, habitat_rotation,
    world_to_camera, habitat_translation,
)
from downstream.utils.pcd_util import (
    gpu_pointcloud_from_array,
    gpu_cluster_filter,
    gpu_merge_pointcloud, pointcloud_distance,
)
from typing import List, Dict


class ObjectState:
    """
    Represents the state of an individual object, including:
        - position
        - radius
        - visited status
        - object_id
        - object_name
    """
    def __init__(
        self,
        object_id: int,
        pred_obj_name: str,
        pcd: np.ndarray,
        visited: bool = False,
    ):
        self.object_id = object_id
        self.pred_obj_name = pred_obj_name
        self.pcd = pcd
        self.visited = visited
        self.update_pcd(pcd)

    def update_pcd(self, pcd: np.ndarray):
        """
        Updates the point cloud of the object, and recomputes the position.
        """
        self.pcd = pcd
        radius = self.pcd.get_max_bound() - self.pcd.get_min_bound()
        self.position = self.pcd.get_center()
        self.radius = np.linalg.norm(radius) / 2.0


    def __repr__(self):
        return (f"ObjectState(object_id={self.object_id}, "
                f"visited={self.visited}, "
                f"pred_obj_name={self.pred_obj_name}),"
                f"pcd.shape={self.pcd.shape})")


class DetectedObjects:
    """
    Manages detected objects within an environment, tracking their states.
    Attributes:
        - detected_objs: Dict[int, ObjectState]
    """
    def __init__(self, explore_radius: float = 1.2, device="cuda:0"):
        self.clean_all()    # Initialize detected objects

        self.never_visited_obj_names = [
            "door", "stairs", "passage",
        ]
        self.explore_radius = explore_radius
        # For pcd processing:
        self.pcd_device = device
        self.pcd_resolution = 0.05
        self.overlap_threshold = 0.2
        self.distance_threshold = 0.15


    def is_passage(self, pred_obj_name) -> bool:
        """
        Determines if the object is a passage based on its name.
        """
        for name in self.never_visited_obj_names:
            if name in pred_obj_name:
                return True
        return False

    def clean_all(self):
        """
        Cleans up all detected objects and resets the object ID.
        """
        self._curr_obj_id = 0    # Initialize current object ID
        self.detected_objs: Dict[int, ObjectState] = {}
        self.never_visited_obj_ids = []
        self.exist_obj_names = []
        self.execution_count = 0

    def add_new_frame(
        self,
        camera_points: List[NDArray],
        obj_names: List[str],
        cam_position: Float[np.ndarray, "(3)"],
        cam_rotation: Float[np.ndarray, "(3 3)"],    # or a quaternion
    ):
        """
        Adds a new object if its radius is within acceptable limits. camera_points: List[NDArray, shape=(N, 3)]
        """
        cam_rotation = habitat_rotation(cam_rotation)
        cam_position = habitat_translation(cam_position)
        # if self.execution_count % 20 == 0:
        #     self.merge_existing_objs()

        # Process the current frame to update object point cloud entities.
        current_obj_entities = self.get_objects_with_pcd(
            camera_points, obj_names, cam_position, cam_rotation
        )
        curr_obj_ids = self.associate_object_entities(
            current_obj_entities
        )
        self.execution_count += 1

        return curr_obj_ids


    def get_objects_with_pcd(self, camera_points, classes, camera_position, camera_rotation):
        """
        Identifies and processes detected objects in the current frame to generate their corresponding
        point clouds in the world coordinate system.
        Args:
            depth (np.ndarray): Depth map of the current frame (shape: [H, W]).
            classes (List[str]): List of detected object class names.
            masks (List[np.ndarray]): List of binary masks for each detected object.
            agent_position (np.ndarray): Agent's position in world coordinates.
            agent_rotation (np.ndarray): Agent's rotation in world coordinates.
        Returns:
            List[Dict]: A list of object entities. Each entity is represented as a dictionary with keys:
                        - 'class': The object's class label.
                        - 'pcd': The point cloud of the object in world coordinates.
        """
        entities = []
        for cls, cam_points in zip(classes, camera_points):
            # if depth[mask].min() < 1.0:
            #     continue
            if cls not in self.exist_obj_names:
                self.exist_obj_names.append(cls)

            world_points = camera_to_world(
                cam_points, camera_position, camera_rotation
            )
            point_colors = np.array(
                [d3_40_colors_rgb[self.exist_obj_names.index(cls) % 40]]
                * world_points.shape[0]
            )

            # if world_points.shape[0] < 10:
            #     continue

            object_pcd = gpu_pointcloud_from_array(
                world_points, point_colors, self.pcd_device
            ).voxel_down_sample(self.pcd_resolution)  # Downsamples pcd

            object_pcd = gpu_cluster_filter(object_pcd)
            # if object_pcd.point.positions.shape[0] < 10:
            #     continue

            entity = {"class": cls, "pcd": object_pcd}
            entities.append(entity)
        return entities


    def associate_object_entities(self, eval_entities):
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
        curr_obj_ids = []
        for entity in eval_entities:
            if len(self.detected_objs) == 0:
                curr_id = self.register_new_object(entity["class"], entity["pcd"])
                curr_obj_ids.append(curr_id)
                continue

            overlap_score = {}
            eval_pcd = entity["pcd"]
            for obj_id, ref_entity in self.detected_objs.items():
                if eval_pcd.point.positions.shape[0] == 0:
                    break

                cdist = pointcloud_distance(eval_pcd, ref_entity.pcd, self.pcd_device)
                overlap_condition = cdist < self.distance_threshold
                nonoverlap_condition = overlap_condition.logical_not()
                eval_pcd = eval_pcd.select_by_index(
                    o3d.core.Tensor(
                        nonoverlap_condition.cpu().numpy(), device=ref_entity.pcd.device
                    ).nonzero()[0]
                )
                overlap_score[obj_id] = (
                    (overlap_condition.sum() / (overlap_condition.shape[0] + 1e-6))
                    .cpu().numpy()
                )
            max_overlap_id, max_overlap_score = max(
                overlap_score.items(), key=lambda item: item[1]
            )

            if max_overlap_score < self.overlap_threshold:
                entity["pcd"] = eval_pcd
                # ref_entities.append(entity)
                curr_id = self.register_new_object(entity["class"], entity["pcd"])
                curr_obj_ids.append(curr_id)
            else:  # merges the point clouds of the current entity and the reference entity with the highest overlap score:
                argmax_entity = self.detected_objs[max_overlap_id]
                updated_pcd = gpu_merge_pointcloud(
                    argmax_entity.pcd, eval_pcd
                )
                argmax_entity.update_pcd(updated_pcd)
                curr_obj_ids.append(argmax_entity.object_id)
                if (
                    argmax_entity.pcd.point.positions.shape[0]
                    < entity["pcd"].point.positions.shape[0]
                ):
                    argmax_entity.pred_obj_name = entity["class"]

        return curr_obj_ids


    def merge_existing_objs(self):
        """
        Merges the point clouds of existing objects that overlap above a threshold.

        This is similar to the logic in associate_object_entities() but operates
        only among already-registered objects in self.detected_objs.
        """
        # Convert the dictionary keys to a list so we can iterate over pairs
        obj_ids = list(self.detected_objs.keys())
        used = set()  # Track objects already merged/removed
        center_threshold = 2.5

        for i in range(len(obj_ids)):
            if obj_ids[i] in used:
                continue
            objA = self.detected_objs[obj_ids[i]]

            for j in range(i + 1, len(obj_ids)):
                if obj_ids[j] in used:
                    continue
                objB = self.detected_objs[obj_ids[j]]

                # --- 1) Prefilter with center distance to reduce computation ---
                center_dist = np.linalg.norm(objA.position - objB.position)
                if center_dist > center_threshold:
                    continue

                # --- 2) Compute overlap score between objA and objB only if prefilter passes ---
                cdist = pointcloud_distance(objA.pcd, objB.pcd, self.pcd_device)
                overlap_condition = cdist < self.distance_threshold
                overlap_score = (overlap_condition.sum() /
                                (overlap_condition.shape[0] + 1e-6))

                # --- 2) If overlap is high, merge objB into objA (or vice versa) ---
                if overlap_score > self.overlap_threshold:
                    # Merge B's point cloud into A's
                    merged_pcd = gpu_merge_pointcloud(objA.pcd, objB.pcd)
                    objA.update_pcd(merged_pcd)

                    # If B had more points, update A's predicted class name
                    if objA.pcd.point.positions.shape[0] < objB.pcd.point.positions.shape[0]:
                        objA.pred_obj_name = objB.pred_obj_name

                    # Remove objB from detected_objs and mark it as used
                    del self.detected_objs[obj_ids[j]]
                    print(f"Removed object {obj_ids[j]} due to merge with {obj_ids[i]}")
                    used.add(obj_ids[j])


    def register_new_object(self, cls, object_pcd):
        self.detected_objs[self._curr_obj_id] = ObjectState(
            object_id=self._curr_obj_id,
            pred_obj_name=cls,
            pcd=object_pcd,
        )
        # specially process for some objs: 
        if self.is_passage(cls):
            self.never_visited_obj_ids.append(self._curr_obj_id)
        self._curr_obj_id += 1
        return self._curr_obj_id - 1


    def label_as_visited(self, obj_id: int):
        """
        Marks the object with the specified ID as visited.
        """
        if obj_id in self.detected_objs:
            self.detected_objs[obj_id].visited = True

    def fetch_pred_obj_names(self, obj_ids: List[int]) -> List[str]:
        """
        Returns the predicted object name for the specified object ID.
        """
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]

        pred_obj_names = []
        for obj_id in obj_ids:
            if obj_id not in self.detected_objs:
                print(f"WARNING: Object ID {obj_id} not found in detected objects.")
                pred_obj_names.append("unknown")
                continue
            obj = self.detected_objs[obj_id]
            pred_obj_names.append(obj.pred_obj_name)
        return pred_obj_names


    def get_unvisited_object_ids(self) -> List[int]:
        """
        Returns the IDs of all unvisited objects.
        """
        return [obj.object_id for obj in self.detected_objs.values() if not obj.visited]

    def get_visited_object_ids(self) -> List[int]:
        """
        Returns the IDs of all visited objects.
        """
        return [obj.object_id for obj in self.detected_objs.values() if obj.visited]

    def get_object_positions(self, obj_ids: List[int]) -> List[List[float]]:
        """
        Returns the XY positions ([x, y]) of specified objects.
        Assumes positions are [x, z, y].
        """
        xy_positions = []
        for obj_id in obj_ids:
            obj = self.detected_objs[obj_id]
            # reverse tansfrormation for habitat_translation:
            position = obj.position.cpu().numpy()
            # xy_positions.append(np.array([position[0], position[2], position[1]]))
            xy_positions.append(np.array([position[0], position[1], position[2]]))
        return xy_positions

    def get_object_radius(self, obj_ids: List[int]) -> List[float]:
        """
        Returns the radius of specified objects.
        """
        radii = []
        for obj_id in obj_ids:
            obj = self.detected_objs[obj_id]
            # convert open3d.cuda.pybind.core.Tensor to python float
            radii.append(obj.radius.cpu().numpy())
        return radii


    def get_object_3d_bbox_corners(self, obj_ids: List[int], flip_yz) -> List[np.ndarray]:
        """
        Returns a list of 8 corners for each object's 3D bounding box.
        Each element in the returned list is an np.ndarray of shape (8, 3),
        where each row is [x, z, y] for a corner.
        NOTE: If flip_yz is True, return habtat_sim's [x, z, y] order, otherwise
              return the original [x, y, z] order.
        """
        all_bbox_corners = []
        for obj_id in obj_ids:
            obj_state = self.detected_objs[obj_id]
            # In standard [x, y, z] (y-up)
            min_x, min_y, min_z = obj_state.pcd.get_min_bound().cpu().numpy()
            max_x, max_y, max_z = obj_state.pcd.get_max_bound().cpu().numpy()
            # min_z, max_z = -min_z, -max_z

            # Corner order matching get_3d_bbox_corners:
            #  7: back_bottom_left   => (min_x, min_y, min_z)
            corners = np.array([
                [min_x, max_y, max_z],  # 0
                [max_x, max_y, max_z],  # 1
                [max_x, min_y, max_z],  # 2
                [min_x, min_y, max_z],  # 3
                [min_x, max_y, min_z],  # 4
                [max_x, max_y, min_z],  # 5
                [max_x, min_y, min_z],  # 6
                [min_x, min_y, min_z],  # 7
            ], dtype=float)

            # If needed, reorder from [x,y,z] => [x,z,y].
            if flip_yz:
                corners = corners[:, [0, 2, 1]]

            all_bbox_corners.append(corners)
        return all_bbox_corners


    def update_object_state(self, agent_pos, dist_fn):
        """
        Updates the visited status of objects based on the agent's position.
        An object is marked as visited if the distance to it is less than its radius.
        """
        agent_pos = habitat_translation(agent_pos)
        for obj in self.detected_objs.values():
            dist = dist_fn(agent_pos, obj.position)
            if (
                dist < self.explore_radius + obj.radius
                and obj.object_id not in self.never_visited_obj_ids
            ):
                obj.visited = True
