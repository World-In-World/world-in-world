import os
import cv2
import math
import numpy as np
import json
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
import magnum as mn
import imageio
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from habitat.utils.visualizations.maps import colorize_topdown_map, draw_agent



def display_map(dir, topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    # plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    # plt.show(block=False)
    # save the fig to a file
    plt.savefig(dir + "-topdown_map.jpg")
    plt.close()


class Recoder:
    def __init__(self, env=None):
        self.env = env
        self.rgb_trajectory = []
        self.depth_trajectory = []
        self.topdown_trajectory = []
        # height = habitat_env.sim.agents[0].state.position[2]

    def reset(self):
        self.rgb_trajectory = []
        self.depth_trajectory = []
        self.topdown_trajectory = []

    def draw_point(
        self,
        top_down_map: np.ndarray,
        waypoint: tuple,
        color: int = 10,
        radius: int = 6,
        thickness: int = -1  # filled circle if negative
    ) -> None:
        """
        Draw a single waypoint on the top down map.
        Args:
            top_down_map: A colored version of the map as a numpy array.
            waypoint: A tuple specifying the (row, col) coordinate of the waypoint.
            color: Color code (from TOP_DOWN_MAP_COLORS or a single int value) for the waypoint.
            radius: Radius of the circle to draw.
            thickness: Circle thickness (if negative, the circle is filled), if the value is 0, the circle is not drawn.
        """
        # The waypoint is expected to be in (y, x) order. If your waypoint is (x, y), swap the order.
        cv2.circle(
            top_down_map,
            center=waypoint[::-1],  # swap if necessary; OpenCV expects (x, y)
            radius=radius,
            color=color,
            thickness=thickness,
        )

    def draw_waypoints(self, path_points, left_points, top_down_map_info, output_height=1024):
        """
        Draw waypoints (path points) and the agent onto a colorized top-down map,
        then rotate and scale the map to the desired output height.
        """
        # 1. Retrieve and colorize the top-down map similarly to Habitat's approach
        height = self.env.sim.agents[0].state.position[1]   #height = self.env.sim.scene_aabb.y().min
        top_down_map = maps.get_topdown_map(
            self.env.sim.pathfinder, height, map_resolution=output_height,
        )
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        # top_down_map = top_down_map_info["map"]
        # fog_of_war_mask = top_down_map_info.get("fog_of_war_mask", None)
        # top_down_map = colorize_topdown_map(top_down_map, fog_of_war_mask)

        # 2. Draw the agent(s) according to top_down_map_info just like
        for agent_idx in range(len(top_down_map_info["agent_map_coord"])):
            map_agent_pos = top_down_map_info["agent_map_coord"][agent_idx]
            map_agent_angle = top_down_map_info["agent_angle"][agent_idx]
            # The radius is set in proportion to your map resolution
            agent_radius_px = min(top_down_map.shape[:2]) // 32

            top_down_map = maps.draw_agent(
                image=top_down_map,
                agent_center_coord=map_agent_pos,
                agent_rotation=map_agent_angle,
                agent_radius_px=agent_radius_px,
            )

        if not hasattr(self, 'height_range'):
            self.min_height, self.height_range = self.get_color_byheight(path_points)

        grid_h, grid_w = top_down_map.shape[:2]
        if len(path_points) > 0:
            self.draw_points_onmap(path_points, top_down_map, grid_h, grid_w)
        if len(left_points) > 0:
            self.draw_points_onmap(left_points, top_down_map, grid_h, grid_w, filled=False)

        # 5. Rotate the map if it is taller than it is wide (optional)
        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # 6. Scale the map to match the desired output height
        old_h, old_w, _ = top_down_map.shape
        top_down_height = output_height
        top_down_width = int(float(top_down_height) / old_h * old_w)

        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),  # (width, height)
            interpolation=cv2.INTER_CUBIC
        )

        return top_down_map

    def draw_points_onmap(self, path_points, top_down_map, grid_h, grid_w, filled=True):
        # 4. Draw the waypoints on the top_down_map
        for pt in path_points:
                # Using habitat's to_grid: assuming pt is [x, y, z] with:
                # x: right, y: vertical (ignored), z: forward.
            grid_pt = maps.to_grid(
                    pt[2],  # forward (z)
                    pt[0],  # right (x)
                    (grid_h, grid_w),
                    pathfinder=self.env.sim.pathfinder,
                )
                # Normalize pt[1] (height) to a value between 0 and 255.
            normalized_value = int(((pt[1] - self.min_height) / self.height_range) * 255)
            point_color = self.get_color_from_height(normalized_value)

            thickness = -1 if filled else 3
            self.draw_point(top_down_map, waypoint=grid_pt, color=point_color, thickness=thickness)


    def get_color_byheight(self, path_points):
        # 3. Convert your path_points to grid coordinates
        heights = [pt[1] for pt in path_points]
        max_height = np.max(heights)
        min_height = np.min(heights)
        height_range = max_height - min_height if max_height != min_height else 1
        return min_height, height_range


    def get_color_from_height(self, height_val: int) -> tuple:
        """
        Convert a value in [0, 255] to a BGR tuple using COLORMAP_JET.
        """
        # Create a 1x1 image with the scalar value.
        arr = np.array([[height_val]], dtype=np.uint8)
        # Apply COLORMAP_JET; result is BGR.
        colored = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
        # Extract the color tuple from the result.
        return tuple(int(c) for c in colored[0, 0])

    def draw_text_onimg(self, img, text, font_scale=1, thickness=2):
        # Get image dimensions
        height, width, _ = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get the text size to calculate the placement
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        # Calculate bottom-left corner of the text
        x = (width - text_width) // 2  # Center the text horizontally
        y = text_height + 10            # Place the text slightly below the top with a 10-pixel margin
        
        # Draw the text on the image
        cv2.putText(img, text, (x, y), font, font_scale, (88, 88, 88), thickness)
        return img


    def update_trajectory(self, obs_dict, save_imgs=False, dir='my_dataset/monitor-{}.jpg'):
        """
        Updates the trajectory with current observations and metrics.
        Args:
            obs_dict (dict): {
                'rgb_eq': torch.Tensor of shape (1, 3, H, W), range [0, 255],
                'depth_eq': torch.Tensor of shape (1, 1, H, W), range [0, 255]
            }
            save_imgs (bool): Whether to save images to disk.
            dir (str): Directory format string for saving images.
        Returns:
            bool: True if the image is too dark and should be skipped, otherwise False.
        """
        os.makedirs(os.path.dirname(dir), exist_ok=True)

        if obs_dict.get('waypoint_list', None) is not None:
            metrics = self.env.get_metrics()
            # add obs_dict['trajPoints] to topdown_image: format: [x, y, z] 
            topdown_image = self.draw_waypoints(
                obs_dict['waypoint_list'], obs_dict['left_points'],
                metrics['top_down_map'], 1024)
            # topdown_image = colorize_draw_agent_and_fit_to_height(
            #     metrics['top_down_map'], 1024)

            topdown_image = self.draw_text_onimg(topdown_image, f"waypoint_idx: {obs_dict['waypoint_idx']}, step: {obs_dict['step_id']}")
            self.topdown_trajectory.append(topdown_image)

        # agent_state = self.env.sim.get_agent_state().sensor_states['rgb']
        # position, rotation = agent_state.position, agent_state.rotation
        if save_imgs:
            # Convert tensors to numpy arrays for processing
            rgb = obs_dict['rgb_eq'].squeeze(0).permute(1, 2, 0).numpy()    #rgb is shape (H, W, 3) now
            if obs_dict['depth_eq'] is not None:
                depth = obs_dict['depth_eq'].squeeze(0).numpy()

            # Update RGB, depth, and panoramic trajectories
            # rgb_resized = cv2.resize(rgb, (512, 256), interpolation=cv2.INTER_LINEAR)
            # self.rgb_trajectory.append(rgb_resized)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)  #rgb2bgr
            # self.depth_trajectory.append(depth)

            status1 = cv2.imwrite(dir.format('rgb'), bgr)
            if not status1: print(f"ERROR: Error saving rgb image to disk for {dir}")
            if obs_dict['depth_eq'] is not None:
                # Convert the 8-bit depth [0, 255] to 16-bit [0, 65535] for higher fidelity.
                depth_high = (depth / 255.0 * 65535).astype(np.uint16)
                status2 = cv2.imwrite(dir.format('depth'), depth_high)
                if not status2: print(f"ERROR: Error saving depth image to disk for {dir}")


    def save_trajectory(self, output_dir):
        """Saves the trajectory as video files."""
        os.makedirs(output_dir, exist_ok=True)

        # Initialize writers for RGB, depth, and metrics videos
        writers = {
            "rgb": imageio.get_writer(os.path.join(output_dir, "fps.mp4"), fps=5),
            # "depth": imageio.get_writer(os.path.join(output_dir, "depth.mp4"), fps=4),
            "metrics": imageio.get_writer(os.path.join(output_dir, "metrics.mp4"), fps=5),
            # "panoramic": imageio.get_writer(os.path.join(output_dir, "panoramic.mp4"), fps=4),
        }

        # for img, dep, met in zip(self.rgb_trajectory, self.depth_trajectory, self.topdown_trajectory):
        # for met in self.topdown_trajectory:
        if len(self.rgb_trajectory) == 0:
            save_items = {'metrics': self.topdown_trajectory}
        elif len(self.topdown_trajectory) == 0:
            save_items = {'rgb': self.rgb_trajectory}
        else:
            save_items = {'rgb': self.rgb_trajectory, 'metrics': self.topdown_trajectory}
            
        for type, items in save_items.items():
            for img in items:
                writers[type].append_data(img)

        # for met, img in zip(*save_items):
        #     writers["rgb"].append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     # writers["depth"].append_data(dep)
        #     writers["metrics"].append_data(cv2.cvtColor(met, cv2.COLOR_BGR2RGB))
        #     # writers["panoramic"].append_data(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
            
        # Close writers
        for writer in writers.values():
            writer.close()

        print(f"Trajectory vis and info is saved to {os.path.join(output_dir, 'metrics.mp4')}")

    def del_mapped_scene_folder(self, dir_path):
        """
        remove the mapped scene folder and all its contents
        """
        import shutil
        shutil.rmtree(dir_path)
        print(f"Removed the mapped scene folder at {dir_path}")

