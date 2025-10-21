from habitat.utils.visualizations import maps
from habitat_sim.utils import common as utils
import magnum as mn
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import cv2


def draw_maps(sim, meters_per_pixel=0.025):
    agent_pos = sim.agents[0].get_state().position
    agent_rot = sim.agents[0].get_state().rotation

    # obs_new = sim.step('move_forward')
    # agent_pos_new = sim.agents[0].get_state().position
    # angle = get_azimuth_from_ponts(agent_pos, agent_pos_new)      # angle should == agent_orientation_curr

    height = agent_pos[1]
    top_down_map = maps.get_topdown_map(
        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    top_down_map = recolor_map[top_down_map]
    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])

    # 1. Convert the quat orientation into an azimuth angle:
    agent_orientation_curr = get_azimuth_from_quat(agent_rot)

    turn_angle = math.pi / 2    #turn right 90
    agent_orientation_target = agent_orientation_curr + turn_angle

    # Transform the agent_orientation_target (azimuth) back to a quaternion orientation:
    # Create a Magnum quaternion rotating around the Y-axis by the target azimuth angle.
    target_rotation_magnum = mn.Quaternion.rotation(
        mn.Rad(agent_orientation_target), mn.Vector3(0.0, 1.0, 0.0)
    )
    # 2. Convert the Magnum quaternion back to Habitat-sim's quaternion format.
    target_rotation = utils.quat_from_magnum(target_rotation_magnum)

    agent_grid_pos = maps.to_grid(
        agent_pos[2], agent_pos[0], grid_dimensions, pathfinder=sim.pathfinder
    )
    top_down_map_ = top_down_map.copy()

    maps.draw_agent(
        top_down_map, agent_grid_pos, agent_orientation_curr, agent_radius_px=8
    )
    maps.draw_agent(
        top_down_map_, agent_grid_pos, get_azimuth_from_quat(target_rotation), agent_radius_px=8
    )

    print("Display the map with agent and path overlay:")
    path = 'data_temp/dowstream_vis'
    display_map(top_down_map, os.path.join(path, "topdown_map.png"))
    display_map(top_down_map_, os.path.join(path, "topdown_map_target.png"))

def get_azimuth_from_quat(agent_rot):
    agent_forward = utils.quat_to_magnum(
        agent_rot
    ).transform_vector(mn.Vector3(0, 0, -1.0))
    agent_orientation_curr = math.atan2(agent_forward[0], agent_forward[2])
    return agent_orientation_curr


def get_azimuth_from_ponts(p1, p2):
    grid_tangent = mn.Vector2(
        p2[0] - p1[0], p2[2] - p1[2]     #x2-x1, y2-y1,
    )
    path_initial_tangent = grid_tangent / grid_tangent.length()
    angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
    return angle

# display a topdown map with matplotlib
def display_map(topdown_map, path, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.savefig(path)
    print(f"Map saved to {path}")