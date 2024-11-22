"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
pic_dir = os.path.join(current_dir, 'PIC')
sys.path.insert(0, pic_dir)
sys.path.insert(0, os.path.join(pic_dir, 'maddpg'))
sys.path.insert(0, os.path.join(pic_dir, 'models'))

import config
import numpy as np
from metrics import gather_all_metric_data
from mpc_cbf import MPC
from scenarios import DoorwayScenario, IntersectionScenario
# from data_logger import BlankLogger
from environment import Environment
# from model_controller import ModelController
from PIC_controller import pic_controller
from simulation import run_simulation
from macbf_torch_controller import macbf_torch_controller
from tqdm import tqdm

SCENARIO = 'Intersection'
RUN_AGENT = 'PIC'

def get_mpc_controllers(scenario, zero_goes_faster):
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


def get_macbf_controllers(scenario):
    controllers = [
        macbf_torch_controller("test", static_obs=scenario.obstacles.copy(), goal=goals[0,:]),
        macbf_torch_controller("test2", static_obs=scenario.obstacles.copy(), goal=goals[1,:])
    ]
    return controllers


def get_pic_controllers(scenario):
    controllers = [
        pic_controller('test', static_obs=scenario.obstacles.copy(), goal=goals[0,:]),
        pic_controller('test1', static_obs=scenario.obstacles.copy(), goal=goals[1,:])
    ]
    return controllers


# Scenarios: "doorway" or "intersection"
if SCENARIO == 'Doorway':
    scenario_params = (-1.0, 0.5, 2.0, 0.15)
    scenario = DoorwayScenario(initial_x=scenario_params[0], initial_y=scenario_params[1], goal_x=scenario_params[2], goal_y=scenario_params[3])
else:
    scenario = IntersectionScenario()


NUM_SIMS = 50

all_metric_data = []
for sim in tqdm(range(NUM_SIMS)):
    # print("Running sim:", sim)
    plotter = None
    # logger = BlankLogger()
    logger = None

    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    goals = scenario.goals.copy()

    if logger:
        logger.set_obstacles(scenario.obstacles.copy())

    env = Environment(scenario.initial.copy(), scenario.goals.copy())
    if RUN_AGENT == 'MPC':
        controllers = get_mpc_controllers(scenario, True)
    elif RUN_AGENT == 'MACBF':
        controllers = get_macbf_controllers(scenario)
    elif RUN_AGENT == "PIC":
        controllers = get_pic_controllers(scenario)

    x_cum, u_cum = run_simulation(scenario, env, controllers, logger, plotter)

    metric_data = gather_all_metric_data(scenario, x_cum[0], x_cum[1], scenario.goals)
    all_metric_data.append(metric_data)

all_metric_data = np.array(all_metric_data)
np.savetxt(f'experiment_results/{RUN_AGENT}_{SCENARIO}.csv', all_metric_data, fmt='%0.4f', delimiter=', ', header='goal_reach_idx0, goal_reach_idx1, min_agent_dist, traj_collision, obs_min_dist_0, obs_collision_0, obs_min_dist_1, obs_collision_1, delta_vel_0, delta_vel_1, path_dev_0, path_dev_1')
