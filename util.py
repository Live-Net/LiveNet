import math
import torch
import config
import numpy as np
import numpy.linalg as LA

EPSILON = 0.001

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2) + EPSILON)
    return np.arccos(np.clip(dot_product, -1, 1))


def calculate_liveliness(ego_pos, opp_pos, ego_vel, opp_vel):
    vel_diff = opp_vel - ego_vel
    pos_diff = opp_pos - ego_pos
    l = np.pi - angle_between_vectors(pos_diff, vel_diff)
    ttc = LA.norm(pos_diff) / LA.norm(vel_diff) # Time-to-collision
    return l, ttc, pos_diff, vel_diff


def get_ray_intersection_point(ego_pos, ego_theta, opp_pos, opp_theta):
    ego_vel = np.array([np.cos(ego_theta), np.sin(ego_theta)])
    opp_vel = np.array([np.cos(opp_theta), np.sin(opp_theta)])
    if not check_intersection(ego_pos, opp_pos, ego_vel, opp_vel):
        return None

    x1, y1 = ego_pos
    x2, y2 = ego_pos + ego_vel
    x3, y3 = opp_pos
    x4, y4 = opp_pos + opp_vel

    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [px, py]
    

def check_intersection(ego_pos, opp_pos, ego_vel, opp_vel):
    ego_vel_uvec = ego_vel / np.linalg.norm(ego_vel)
    opp_vel_uvec = opp_vel / np.linalg.norm(opp_vel)
    dx = opp_pos[0] - ego_pos[0]
    dy = opp_pos[1] - ego_pos[1]
    det = opp_vel_uvec[0] * ego_vel_uvec[1] - opp_vel_uvec[1] * ego_vel_uvec[0]
    u = (dy * opp_vel_uvec[0] - dx * opp_vel_uvec[1]) * det
    v = (dy * ego_vel_uvec[0] - dx * ego_vel_uvec[1]) * det

    return u > 0 and v > 0


def calculate_all_metrics(ego_state, opp_state, liveness_thresh):
    ego_vel_vec = np.array([np.cos(ego_state[2]), np.sin(ego_state[2])]) * ego_state[3]
    opp_vel_vec = np.array([np.cos(opp_state[2]), np.sin(opp_state[2])]) * opp_state[3]
    l, ttc, pos_diff, vel_diff = calculate_liveliness(ego_state[:2], opp_state[:2], ego_vel_vec, opp_vel_vec)

    intersecting = check_intersection(ego_state[:2], opp_state[:2], ego_vel_vec, opp_vel_vec)

    is_live = False
    if config.consider_intersects:
        if l > liveness_thresh or not intersecting:
            is_live = True
    else:
        if l > liveness_thresh:
            is_live = True

    return l, ttc, pos_diff, vel_diff, intersecting, is_live


def get_x_is_d_goal_input(inputs, goal):
    x = goal[0] - inputs[0]
    y = goal[1] - inputs[1]
    theta = inputs[2]
    v = inputs[3]
    opp_x = inputs[4] - inputs[0] # opp_x - ego_x (ego frame)
    opp_y = inputs[5] - inputs[1] # opp_y - ego_y (ego frame)
    opp_theta = inputs[6]
    opp_v = inputs[7]
    inputs = np.array([x, y, theta, v, opp_x, opp_y, opp_theta, opp_v])
    return inputs


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def perturb_model_input(inputs, scenario_obstacles, num_total_opponents, x_is_d_goal, add_liveness_as_input, fixed_liveness_input, static_obs_xy_only, ego_frame_inputs, add_new_liveness_as_input, add_dist_to_static_obs, goal, metrics=None):
    if metrics is None and add_liveness_as_input:
        metrics = calculate_all_metrics(np.array(inputs[:4]), np.array(inputs[4:8]), config.liveness_threshold)

    ego_pos = np.array(inputs[:2].copy())
    ego_theta = inputs[2]
    ego_vel = inputs[3]
    opp_pos = np.array(inputs[4:6].copy())
    opp_theta = inputs[6]
    opp_vel = inputs[7]
    num_obstacles = num_total_opponents - 1
    agent_obs = sorted(scenario_obstacles, key = lambda o: np.linalg.norm(ego_pos - np.array(o[:2])))
    agent_obs = agent_obs[:num_obstacles]
    # print(ego_pos)
    # print(agent_obs)

    if x_is_d_goal:
        origin = np.array([0.0, 0.0])
        inputs = get_x_is_d_goal_input(inputs, goal)
        if ego_frame_inputs:
            goal_x, goal_y = rotate(origin, (inputs[0], inputs[1]), -ego_theta)
            opp_x, opp_y = rotate(origin, (inputs[4], inputs[5]), -ego_theta)
            opp_theta = inputs[6] - ego_theta
            inputs[0], inputs[1], inputs[2], inputs[4], inputs[5], inputs[6] = goal_x, goal_y, 0.0, opp_x, opp_y, opp_theta

    for obs_x, obs_y, _ in agent_obs:
        if x_is_d_goal:
            obs_inp_x, obs_inp_y = obs_x - ego_pos[0], obs_y - ego_pos[1]
            if ego_frame_inputs:
                obs_inp_x, obs_inp_y = rotate(origin, (obs_inp_x, obs_inp_y), -ego_theta)
            if static_obs_xy_only:
                obs_inp = np.array([obs_inp_x, obs_inp_y]) # opp - ego (ego frame)     
            else:
                obs_inp = np.array([obs_inp_x, obs_inp_y, 0.0, 0.0]) # opp - ego (ego frame)            
        else:
            if static_obs_xy_only:
                obs_inp = np.array([obs_x, obs_y]) # opp - ego (ego frame)
            else:
                obs_inp = np.array([obs_x, obs_y, 0.0, 0.0])
        if static_obs_xy_only and add_dist_to_static_obs:
            obs_inp = np.append(obs_inp, [np.sqrt(obs_x ** 2 + obs_y ** 2)])
        inputs = np.append(inputs, obs_inp)

    if add_liveness_as_input:
        if not metrics[-2]: # Not intersecting
            if fixed_liveness_input:
                liveness = np.pi
            else:
                liveness = 0.0
        else: # Intersecting
            liveness = metrics[0]
        inputs = np.append(inputs, liveness)

    if add_new_liveness_as_input:
        center_intersection = get_ray_intersection_point(list(ego_pos), ego_theta, list(opp_pos), opp_theta)
        vec_to_opp = opp_pos - ego_pos
        unit_vec_to_opp = vec_to_opp / np.linalg.norm(vec_to_opp)
        initial_closest_to_opp = ego_pos + unit_vec_to_opp * (config.agent_radius)
        opp_closest_to_initial = opp_pos - unit_vec_to_opp * (config.agent_radius)
        intersection = get_ray_intersection_point(list(initial_closest_to_opp), ego_theta, list(opp_closest_to_initial), opp_theta)
        if center_intersection is None or intersection is None or ego_vel == 0 or opp_vel == 0:
            inputs = np.append(inputs, 10.0)
        else:
            d0 = np.linalg.norm(initial_closest_to_opp - intersection)
            d1 = np.linalg.norm(opp_closest_to_initial - intersection)

            d0_center = np.linalg.norm(ego_pos - center_intersection)
            d1_center = np.linalg.norm(opp_pos - center_intersection)

            t0 = d0_center / ego_vel
            t1 = d1_center / opp_vel

            if t0 < t1: # Ego agent is faster
                barrier = d1 / opp_vel - d0 / ego_vel
            else: # Ego agent is slower
                barrier = d0 / ego_vel - d1 / opp_vel
            inputs = np.append(inputs, barrier)

    return inputs

