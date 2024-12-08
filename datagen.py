"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import os
import matplotlib.pyplot as plt
import config
from mpc_cbf import MPC
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from environment import Environment
from simulation import run_simulation

from metrics import gather_all_metric_data
import numpy as np

SCENARIO = 'Doorway'
# SCENARIO = 'Intersection'

# start x, start y, goal x, goal y, opp gamma, obs gamma, liveness gamma
if SCENARIO == 'Doorway':
    scenario_configs = [
        # Starting vel 0, Facing goal

        (-1.0, 0.5, 2.0, 0.15, 0.0, True), # 0
        (-1.0, 0.5, 2.0, 0.30, 0.0, True), # 1
        (-1.0, 0.5, 2.0, 0.45, 0.0, True), # 2
        (-1.0, 0.4, 2.0, 0.15, 0.0, True), # 3
        (-1.0, 0.4, 2.0, 0.35, 0.0, True), # 4
        (-1.0, 0.3, 2.0, 0.15, 0.0, True), # 5
        (-1.0, 0.3, 2.0, 0.25, 0.0, True), # 6

        (-0.5, 0.5, 2.0, 0.15, 0.0, True), # 7
        (-0.5, 0.5, 2.0, 0.30, 0.0, True), # 8
        (-0.5, 0.5, 2.0, 0.45, 0.0, True), # 9
        (-0.5, 0.4, 2.0, 0.15, 0.0, True), # 10
        (-0.5, 0.4, 2.0, 0.35, 0.0, True), # 11
        (-0.5, 0.3, 2.0, 0.15, 0.0, True), # 12
        (-0.5, 0.3, 2.0, 0.25, 0.0, True), # 13

        # Starting vel 0, Not facing goal

        (-1.0, 0.5, 2.0, 0.15, 0.0, False), # 0
        (-1.0, 0.5, 2.0, 0.30, 0.0, False), # 1
        (-1.0, 0.5, 2.0, 0.45, 0.0, False), # 2
        (-1.0, 0.4, 2.0, 0.15, 0.0, False), # 3
        (-1.0, 0.4, 2.0, 0.35, 0.0, False), # 4
        (-1.0, 0.3, 2.0, 0.15, 0.0, False), # 5
        (-1.0, 0.3, 2.0, 0.25, 0.0, False), # 6

        (-0.5, 0.5, 2.0, 0.15, 0.0, False), # 7
        (-0.5, 0.5, 2.0, 0.30, 0.0, False), # 8
        (-0.5, 0.5, 2.0, 0.45, 0.0, False), # 9
        (-0.5, 0.4, 2.0, 0.15, 0.0, False), # 10
        (-0.5, 0.4, 2.0, 0.35, 0.0, False), # 11
        (-0.5, 0.3, 2.0, 0.15, 0.0, False), # 12
        (-0.5, 0.3, 2.0, 0.25, 0.0, False), # 13

        # Starting vel 0.3, Facing goal

        (-1.0, 0.5, 2.0, 0.15, 0.3, True), # 0
        (-1.0, 0.5, 2.0, 0.30, 0.3, True), # 1
        (-1.0, 0.5, 2.0, 0.45, 0.3, True), # 2
        (-1.0, 0.4, 2.0, 0.15, 0.3, True), # 3
        (-1.0, 0.4, 2.0, 0.35, 0.3, True), # 4
        (-1.0, 0.3, 2.0, 0.15, 0.3, True), # 5
        (-1.0, 0.3, 2.0, 0.25, 0.3, True), # 6

        (-0.5, 0.5, 2.0, 0.15, 0.3, True), # 7
        (-0.5, 0.5, 2.0, 0.30, 0.3, True), # 8
        (-0.5, 0.5, 2.0, 0.45, 0.3, True), # 9
        (-0.5, 0.4, 2.0, 0.15, 0.3, True), # 10
        (-0.5, 0.4, 2.0, 0.35, 0.3, True), # 11
        (-0.5, 0.3, 2.0, 0.15, 0.3, True), # 12
        (-0.5, 0.3, 2.0, 0.25, 0.3, True), # 13

        # Starting vel 0.3, Not facing goal

        (-1.0, 0.5, 2.0, 0.15, 0.3, False), # 0
        (-1.0, 0.5, 2.0, 0.30, 0.3, False), # 1
        (-1.0, 0.5, 2.0, 0.45, 0.3, False), # 2
        (-1.0, 0.4, 2.0, 0.15, 0.3, False), # 3
        (-1.0, 0.4, 2.0, 0.35, 0.3, False), # 4
        (-1.0, 0.3, 2.0, 0.15, 0.3, False), # 5
        (-1.0, 0.3, 2.0, 0.25, 0.3, False), # 6

        (-0.5, 0.5, 2.0, 0.15, 0.3, False), # 7
        (-0.5, 0.5, 2.0, 0.30, 0.3, False), # 8
        (-0.5, 0.5, 2.0, 0.45, 0.3, False), # 9
        (-0.5, 0.4, 2.0, 0.15, 0.3, False), # 10
        (-0.5, 0.4, 2.0, 0.35, 0.3, False), # 11
        (-0.5, 0.3, 2.0, 0.15, 0.3, False), # 12
        (-0.5, 0.3, 2.0, 0.25, 0.3, False), # 13
    ]
    scenarios = [DoorwayScenario(initial_x=conf[0], initial_y=conf[1], goal_x=conf[2], goal_y=conf[3], initial_vel=conf[4], start_facing_goal=conf[5]) for conf in scenario_configs]

    best_params = [
        (0.5, 0.2, 0.3, 16.0), # 0
        (0.8, 0.2, 0.4, 16.0), # 1
        (0.7, 0.2, 0.2, 16.0), # 2 redo
        (0.7, 0.2, 0.1, 18.0), # 3
        (0.7, 0.2, 0.2, 16.0), # 4 redo
        (0.8, 0.2, 0.2, 16.0), # 5
        (0.8, 0.2, 0.2, 16.0), # 6
        (0.9, 0.2, 0.2, 14.0), # 7 redo
        (0.9, 0.2, 0.2, 14.0), # 8 redo
        (0.9, 0.2, 0.2, 15.0), # 9 redo
        (0.9, 0.2, 0.2, 14.0), # 10
        (0.9, 0.2, 0.2, 14.0), # 11 redo
        (0.9, 0.2, 0.2, 14.0), # 12
        (0.9, 0.2, 0.2, 14.0), # 13 redo

        (0.5, 0.2, 0.3, 16.0), # 0
        (0.8, 0.2, 0.4, 16.0), # 1
        (0.7, 0.2, 0.2, 16.0), # 2 redo
        (0.7, 0.2, 0.1, 18.0), # 3
        (0.7, 0.2, 0.2, 16.0), # 4 redo
        (0.8, 0.2, 0.2, 16.0), # 5
        (0.8, 0.2, 0.2, 16.0), # 6
        (0.9, 0.2, 0.2, 16.0), # 7 redo
        (0.9, 0.2, 0.2, 14.0), # 8 redo
        (0.9, 0.2, 0.2, 15.0), # 9 redo
        (0.9, 0.2, 0.2, 14.0), # 10
        (0.9, 0.2, 0.2, 14.0), # 11 redo
        (0.9, 0.2, 0.2, 14.0), # 12
        (0.9, 0.2, 0.2, 14.0), # 13 redo

        (0.5, 0.2, 0.3, 16.0), # 0
        (0.8, 0.2, 0.4, 16.0), # 1
        (0.7, 0.2, 0.2, 16.0), # 2 redo
        (0.7, 0.2, 0.1, 18.0), # 3
        (0.7, 0.2, 0.2, 16.0), # 4 redo
        (0.8, 0.2, 0.2, 16.0), # 5
        (0.8, 0.2, 0.2, 16.0), # 6
        (0.9, 0.2, 0.2, 14.0), # 7 redo
        (0.9, 0.2, 0.2, 14.0), # 8 redo
        (0.9, 0.2, 0.2, 15.0), # 9 redo
        (0.9, 0.2, 0.2, 14.0), # 10
        (0.9, 0.2, 0.2, 14.0), # 11 redo
        (0.9, 0.2, 0.2, 14.0), # 12
        (0.9, 0.2, 0.2, 14.0), # 13 redo

        (0.5, 0.2, 0.2, 16.0), # 0
        (0.8, 0.2, 0.4, 16.0), # 1
        (0.7, 0.2, 0.2, 16.0), # 2 redo
        (0.7, 0.2, 0.1, 18.0), # 3
        (0.7, 0.2, 0.2, 16.0), # 4 redo
        (0.8, 0.2, 0.2, 16.0), # 5
        (0.8, 0.2, 0.2, 16.0), # 6
        (0.9, 0.2, 0.2, 14.0), # 7 redo
        (0.9, 0.2, 0.2, 14.0), # 8 redo
        (0.9, 0.2, 0.2, 15.0), # 9 redo
        (0.9, 0.2, 0.2, 14.0), # 10
        (0.9, 0.2, 0.2, 14.0), # 11 redo
        (0.9, 0.2, 0.2, 14.0), # 12
        (0.9, 0.2, 0.2, 14.0), # 13 redo
    ]

    scenario_configs = scenario_configs[28:]
    scenarios = scenarios[28:]
    best_params = best_params[28:]

    folder_to_save_to = 'doorway_scenario_suite_6/'

else:
    folder_to_save_to = 'intersection_scenario_suite3/'
    # best_params = [
    # (0.5, 0.3, 0.3, 0.5, True, 14.0), # 0
    # ]
    # scenarios = [IntersectionScenario()]

    scenario_configs = [
        (0.8, 0.8),
        (0.8, 1.0),
        (0.8, 1.2),
        (1.0, 0.8),
        (1.0, 1.0),
        (1.0, 1.2),
        (1.2, 0.8),
        (1.2, 1.0),
        (1.2, 1.2),
    ]
    scenarios = [IntersectionScenario(start=conf[0], goal=conf[1]) for conf in scenario_configs]

    best_params = [
        (0.6, 0.1, 13.0),
        (0.6, 0.1, 14.0),
        (0.6, 0.1, 16.0),
        (0.6, 0.1, 13.0),
        (0.6, 0.1, 14.0),
        (0.6, 0.1, 15.0),
        (0.6, 0.1, 16.0),
        (0.6, 0.1, 16.0),
        (0.6, 0.1, 16.0)
    ]

start_idx, end_idx = 21, 28
# animations/doorway_scenario_suite_4/s_doorway_-1.0_0.5_2.0_0.15_False_0.3_l_1_faster_off0..mp4
# start_idx, end_idx = 42, 43
scenarios = scenarios[start_idx:end_idx]
best_params = best_params[start_idx:end_idx]


assert(len(scenarios) == len(best_params))

offset = [0]
zero_faster = [True, False]
all_metric_data = []
for scenario, mpc_params in zip(scenarios, best_params):
    if type(scenario) == DoorwayScenario:
        config.opp_gamma = mpc_params[0]
        config.obs_gamma = mpc_params[1]
        config.liveliness_gamma = mpc_params[2]
        config.runtime = mpc_params[3]
    else:
        config.opp_gamma = mpc_params[0]
        config.liveliness_gamma = mpc_params[1]
        config.runtime = mpc_params[2]

    for z in zero_faster:
        for o in offset:
            print(f"Running scenario {str(scenario)} with z {z} and o {o}")
            # Don't include situations that won't happen.
            if o > 0 and z:
                continue
            if o < 0 and not z:
                continue

            config.agent_zero_offset = o
            config.mpc_p0_faster = z
            log_filename = f'{scenario.save_str()}_l_{0 if z else 1}_faster_off{o}.json'
            log_filepath = os.path.join(folder_to_save_to, log_filename)
            logger = DataLogger(log_filepath)

            # Matplotlib plotting handler
            config.ani_save_name = log_filepath.rstrip('json') + '.mp4'
            plotter = Plotter()
            # plotter = None

            # Add all initial and goal positions of the agents here (Format: [x, y, theta])
            goals = scenario.goals.copy()
            logger.set_obstacles(scenario.obstacles.copy())
            env = Environment(scenario.initial.copy(), scenario.goals.copy())
            controllers = []

            # Setup agents
            controllers.append(MPC(agent_idx=0, goal=goals[0,:], opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, static_obs=scenario.obstacles.copy(), delay_start=max(config.agent_zero_offset, 0.0)))
            controllers.append(MPC(agent_idx=1, goal=goals[1,:], opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, static_obs=scenario.obstacles.copy(), delay_start=max(-config.agent_zero_offset, 0.0)))

            x_cum, u_cum = run_simulation(scenario, env, controllers, logger, plotter)
            print("Saving scenario to:", log_filename)
            # plt.close()

            metric_data = gather_all_metric_data(scenario, x_cum[0], x_cum[1], scenario.goals, env.compute_history)
            all_metric_data.append(metric_data)

            all_metric_data_save = np.array(all_metric_data)
            save_filename = f"{folder_to_save_to.rstrip('/')}_{start_idx}_{end_idx}.csv"
            print(f"Saving experiment results to {save_filename}")
            np.savetxt(save_filename, all_metric_data_save, fmt='%0.4f', delimiter=', ', header='goal_reach_idx0, goal_reach_idx1, min_agent_dist, traj_collision, obs_min_dist_0, obs_collision_0, obs_min_dist_1, obs_collision_1, delta_vel_0, delta_vel_1, path_dev_0, path_dev_1, avg_compute_0, avg_compute_1')
