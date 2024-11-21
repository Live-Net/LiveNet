"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import config
import numpy as np
from metrics import gather_all_metric_data
from mpc_cbf import MPC
from scenarios import DoorwayScenario, IntersectionScenario
from data_logger import BlankLogger
from environment import Environment
from model_controller import ModelController
from simulation import run_simulation

SCENARIO = 'Doorway'
# SCENARIO = 'Intersection'

# RUN_AGENT = 'MPC'
# RUN_AGENT = 'MPC_UNLIVE'
RUN_AGENT = 'BarrierNet'
# RUN_AGENT = 'LiveNet'

def get_mpc_live_controllers(scenario, zero_goes_faster):
    if SCENARIO == 'Doorway' or SCENARIO == 'Intersection':
        config.liveliness = True
        config.mpc_p0_faster = zero_goes_faster
        config.opp_gamma = 0.5
        config.obs_gamma = 0.3
        config.liveliness_gamma = 0.3
        config.liveness_threshold = 1.0

    controllers = [
        MPC(agent_idx=0, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=scenario.goals[0,:].copy(), static_obs=scenario.obstacles.copy()),
        MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=scenario.goals[1,:].copy(), static_obs=scenario.obstacles.copy())
    ]
    return controllers


def get_mpc_unlive_controllers(scenario):
    if SCENARIO == 'Doorway' or SCENARIO == 'Intersection':
        config.liveliness = False
        config.opp_gamma = 0.1
        config.obs_gamma = 0.1

    controllers = [
        MPC(agent_idx=0, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=scenario.goals[0,:].copy(), static_obs=scenario.obstacles.copy()),
        MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=scenario.goals[1,:].copy(), static_obs=scenario.obstacles.copy())
    ]
    return controllers


def get_barriernet_controllers(scenario):
    if SCENARIO == 'Doorway':
#        model_0_def = "weights/model_base_single_input_obs_wc_nolim_saf_suite_0_bn_definition.json"
#        model_1_def = "weights/model_base_single_input_obs_wc_nolim_saf_suite_1_bn_definition.json"
        model_0_def = "weights/model_base_single_input_obs_wc_nolim_saf_suite_0_1_bn_definition.json"
        model_1_def = "weights/model_base_single_input_obs_wc_nolim_saf_suite_0_1_bn_definition.json"

    controllers = [
        ModelController(model_0_def, scenario.goals[0], scenario.obstacles.copy()),
        ModelController(model_1_def, scenario.goals[1], scenario.obstacles.copy()),
    ]
    return controllers

def get_livenet_controllers(scenario):
    if SCENARIO == 'Doorway':
        model_0_def = "weights/model_base_single_input_obs_wc_nolim_linp_f_fullsuite_0_1_bn_definition.json"
        model_1_def = "weights/model_base_single_input_obs_wc_nolim_linp_f_fullsuite_0_1_bn_definition.json"

    controllers = [
        ModelController(model_0_def, scenario.goals[0], scenario.obstacles.copy()),
        ModelController(model_1_def, scenario.goals[1], scenario.obstacles.copy()),
    ]
    return controllers


# Scenarios: "doorway" or "intersection"
if SCENARIO == 'Doorway':
    scenario_params = (-1.0, 0.5, 2.0, 0.15)
    scenario = DoorwayScenario(initial_x=scenario_params[0], initial_y=scenario_params[1], goal_x=scenario_params[2], goal_y=scenario_params[3])
elif SCENARIO == 'Intersection':
    scenario = IntersectionScenario()


print(f"Running experimnets on agent {RUN_AGENT} on scenario {SCENARIO}")
NUM_SIMS = 50

all_metric_data = []
for sim in range(NUM_SIMS):
    print("Running sim:", sim)
    plotter = None
    logger = BlankLogger()

    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    goals = scenario.goals.copy()
    logger.set_obstacles(scenario.obstacles.copy())
    env = Environment(scenario.initial.copy(), scenario.goals.copy())
    if RUN_AGENT == 'MPC':
        controllers = get_mpc_live_controllers(scenario, True)
    elif RUN_AGENT == 'MPC_UNLIVE':
        controllers = get_mpc_unlive_controllers(scenario)
    elif RUN_AGENT == 'BarrierNet':
        controllers = get_barriernet_controllers(scenario)
    elif RUN_AGENT == 'LiveNet':
        controllers = get_livenet_controllers(scenario)

    x_cum, u_cum = run_simulation(scenario, env, controllers, logger, plotter)

    metric_data = gather_all_metric_data(scenario, x_cum[0], x_cum[1], scenario.goals)
    all_metric_data.append(metric_data)

all_metric_data = np.array(all_metric_data)
save_filename = f"experiment_results/{RUN_AGENT}_{SCENARIO}.csv"
print(f"Saving experiment results to {save_filename}")
np.savetxt(save_filename, all_metric_data, fmt='%0.4f', delimiter=', ', header='goal_reach_idx0, goal_reach_idx1, min_agent_dist, traj_collision, obs_min_dist_0, obs_collision_0, obs_min_dist_1, obs_collision_1, delta_vel_0, delta_vel_1, path_dev_0, path_dev_1')
