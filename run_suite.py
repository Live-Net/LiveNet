"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import os
import matplotlib.pyplot as plt
import config
import numpy as np
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import BlankLogger
from environment import Environment
from simulation import run_simulation
from model_controller import ModelController
from run_experiments import get_livenet_controllers, get_mpc_live_controllers
from metrics import gather_all_metric_data, load_desired_path

# start x, start y, goal x, goal y, opp gamma, obs gamma, liveness gamma
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

VIZ = False
SCENARIO = 'Doorway'
RUN_AGENT = 'LiveNet'
# model_path = "model_30_norm_doorsuite2_lfnew_0_1_bn_definition"
# model_path = "model_30_norm_doorsuite4_lfnew_nso_ego_0_1_bn_definition"
# model_path = "model_30_norm_doorsuite4_lfnew_so_nego_0_1_bn_definition"

# model_path = "model_30_norm_doorsuite4_lfnew_nso_nego_0_1_bn_definition" # 23 / 56

# model_path = "model_40_norm_doorsuite4_lfnew_nso_nego_wl_0_1_bn_definition"
# model_path = "model_40_norm_doorsuite4_lfnew_nso_nego_8o_0_1_bn_definition"
# model_path = "model_40_norm_doorsuite4_lfnew_nso_nego_seppen_0_1_bn_definition"
# model_path = "model_35_norm_doorsuite4_lfnew_nso_nego_8o_small_0_1_bn_definition"
# model_path = "model_35_norm_doorsuite4_lfnew_nso_nego_wnewl_small_0_1_bn_definition"
# model_path = "srikar_iter_5_3opp_od_0_1_bn_definition"
# model_path = "srikar_iter_6_3opp_od_seploop_0_1_bn_definition"

# model_path = "srikar_iter_6_3opp_od_seploop_suite5_0_1_bn_definition"
# model_path = "srikar_iter_7_suite5_0_1_bn_definition"
# model_path = "srikar_iter_7_nol_suite5_0_1_bn_definition"
# model_path = "srikar_iter_8_6nol_suite5_0_1_bn_definition"
# model_path = "srikar_iter_8_6l_suite5_0_1_bn_definition"
# model_path = "srikar_iter_8_nol_suite5_0_1_bn_definition"
# model_path = "srikar_iter_8_l_suite5_0_1_bn_definition"


all_metric_data = []
# scenario_configs = scenario_configs[:1]
for scenario_config in scenario_configs:
    scenario = DoorwayScenario(initial_x=scenario_config[0], initial_y=scenario_config[1], goal_x=scenario_config[2], goal_y=scenario_config[3], initial_vel=scenario_config[4], start_facing_goal=scenario_config[5])
    print(f"Running scenario {str(scenario)}")

    logger = BlankLogger()
    if VIZ:
        plotter = Plotter()
    else:
        plotter = None

    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    goals = scenario.goals.copy()
    logger.set_obstacles(scenario.obstacles.copy())
    env = Environment(scenario.initial.copy(), scenario.goals.copy())
    if RUN_AGENT == "LiveNet":
        # controllers = get_livenet_controllers(scenario, SCENARIO)
        model_def = f"weights/{model_path}.json"
        print(model_def)
        controllers = [
            ModelController(model_def, scenario.goals[0], scenario.obstacles.copy()),
            ModelController(model_def, scenario.goals[1], scenario.obstacles.copy()),
        ]

    elif RUN_AGENT == "MPC":
        controllers = get_mpc_live_controllers(scenario, SCENARIO)

    x_cum, u_cum = run_simulation(scenario, env, controllers, logger, plotter)

    desired_path_0 = load_desired_path(f"experiment_results/desired_paths/{SCENARIO}_{RUN_AGENT}_0.json", 0)
    desired_path_1 = load_desired_path(f"experiment_results/desired_paths/{SCENARIO}_{RUN_AGENT}_1.json", 1)
    metric_data = gather_all_metric_data(scenario, x_cum[0], x_cum[1], scenario.goals, env.compute_history, desired_path_0=desired_path_0, desired_path_1=desired_path_1)
    all_metric_data.append(metric_data)

    all_metric_data_save = np.array(all_metric_data)
    save_filename = f"experiment_results/{RUN_AGENT}_{SCENARIO}_{model_path}_suite.csv"
    print(f"Saving suite results to {save_filename}")
    np.savetxt(save_filename, all_metric_data_save, fmt='%0.4f', delimiter=', ', header='goal_reach_idx0, goal_reach_idx1, min_agent_dist, traj_collision, obs_min_dist_0, obs_collision_0, obs_min_dist_1, obs_collision_1, delta_vel_0, delta_vel_1, path_dev_0, path_dev_1, avg_compute_0, avg_compute_1')

    if VIZ:
        plt.close()
