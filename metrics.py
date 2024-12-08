import config
import numpy as np
from data_logger import DataLogger

def dist_between_line_and_point(target_point, line_point, line_heading):
    cos_heading = np.cos(line_heading)
    sin_heading = np.sin(line_heading)

    dx = target_point[0] - line_point[0]
    dy = target_point[1] - line_point[1]

    distance = np.abs(cos_heading * dy - sin_heading * dx)
    
    return distance

def get_straight_line_desired_path(start, goal, discretization=0.01):
    number_of_points = int(np.linalg.norm(start - goal) / discretization) + 1
    xs = np.linspace(start[0] , goal[0], number_of_points)
    ys = np.linspace(start[1], goal[1], number_of_points)
    return np.array([xs, ys]).T


def calculate_path_deviation(traj, desired_path):
    total_deviation = 0
    for state in traj:
        path_deviation_array = np.linalg.norm(state[:2] - desired_path, axis=1)
        closest_idx = np.argmin(path_deviation_array)
        path_deviation_array[closest_idx] = np.inf
        second_closest_idx = np.argmin(path_deviation_array)

        path1 = desired_path[closest_idx]
        path2 = desired_path[second_closest_idx]
        line_heading = np.arctan2(path2[1] - path1[1], path2[0] - path1[0])
        deviation = dist_between_line_and_point(state[:2], desired_path[closest_idx], line_heading)

        total_deviation += deviation
    return total_deviation / len(traj)


def check_when_reached_goal(traj, goal):
    for iteration, state in enumerate(traj):
        # If we've reached the goal and come to a stop, stop counting.
        if np.linalg.norm(state[:2] - goal) < 0.05 and np.abs(state[3]) < 0.05:
            return iteration

    return None


def calculate_avg_delta_vel(traj):
    total_delta_v = 0.0
    prev_v = traj[0][3]
    count = 0
    for state in traj:
        total_delta_v += abs(state[3] - prev_v)
        prev_v = state[3]
        count += 1
    return total_delta_v / count


def check_for_traj_collisions(traj1, traj2, early_exit = False):
    min_dist = float("inf")
    for state1, state2 in zip(traj1, traj2):
        dist = np.linalg.norm(state1[:2] - state2[:2])
        min_dist = min(dist, min_dist)
        if early_exit:
            if dist < config.agent_radius * 2 + config.safety_dist:
                return min_dist, True

    return min_dist, min_dist < config.agent_radius * 2 + config.safety_dist


def check_for_obs_collisions(traj, obstacles, early_exit = False):
    min_dist = float("inf")
    collides = False
    for state in traj:
        for obs_x, obs_y, obs_r in obstacles:
            dist = np.linalg.norm(state[:2] - np.array([obs_x, obs_y]))
            min_dist = min(min_dist, dist)
            if dist < config.agent_radius + obs_r + config.safety_dist:
                collides = True

            if early_exit and collides:
                return min_dist, False

    return min_dist, collides


def load_desired_path(filename, agent_idx):
    logger = DataLogger.load_file(filename)
    path = []
    for iteration in logger.data['iterations']:
        path.append(iteration['states'][agent_idx][:2])
    return np.array(path)


def gather_all_metric_data(scenario, traj0, traj1, goals, compute_history, desired_path_0=None, desired_path_1=None):
    if desired_path_0 is None:
        desired_path_0 = get_straight_line_desired_path(scenario.initial[0], scenario.goals[0])
    if desired_path_1 is None:
        desired_path_1 = get_straight_line_desired_path(scenario.initial[1], scenario.goals[1])

    goal_reach_idx0 = check_when_reached_goal(traj0, goals[0, :2])
    goal_reach_idx1 = check_when_reached_goal(traj1, goals[1, :2])

    traj0_to_consider = traj0
    if goal_reach_idx0 is not None:
        traj0_to_consider = traj0[:goal_reach_idx0]

    traj1_to_consider = traj1
    if goal_reach_idx1 is not None:
        traj1_to_consider = traj1[:goal_reach_idx1]

    min_agent_dist, traj_collision = check_for_traj_collisions(traj0, traj1)
    obs_min_dist_0, obs_collision_0 = check_for_obs_collisions(traj0_to_consider, scenario.obstacles)
    obs_min_dist_1, obs_collision_1 = check_for_obs_collisions(traj1_to_consider, scenario.obstacles)
    delta_vel_0 = calculate_avg_delta_vel(traj0_to_consider)
    delta_vel_1 = calculate_avg_delta_vel(traj1_to_consider)
    path_dev_0 = calculate_path_deviation(traj0_to_consider, desired_path_0)
    path_dev_1 = calculate_path_deviation(traj1_to_consider, desired_path_1)

    goal_reach_idx0 = -1 if goal_reach_idx0 is None else goal_reach_idx0
    goal_reach_idx1 = -1 if goal_reach_idx1 is None else goal_reach_idx1

    compute_history = np.array(compute_history)
    avg_compute_0, avg_compute_1 = np.mean(compute_history, axis=0)

    return [goal_reach_idx0, goal_reach_idx1, min_agent_dist, traj_collision, obs_min_dist_0, obs_collision_0, obs_min_dist_1, obs_collision_1, delta_vel_0, delta_vel_1, path_dev_0, path_dev_1, avg_compute_0, avg_compute_1]
