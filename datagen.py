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

# start x, start y, goal x, goal y, opp gamma, obs gamma, liveness gamma
scenario_configs = [
    # (-1, 0.5, 2, 0.15, 0.5, 0.5, 0.3),
    # (-1, 0.5, 2, 0.25, 0.5, 0.5, 0.3),
    # (-1, 0.5, 2, 0.35, 0.5, 0.3),
    # (-1, 0.5, 2, 0.15, 0.5, 0.3),
    # (-1, 0.5, 2, 0.15, 0.5, 0.3),

    (-1.0, 0.5, 2.0, 0.15), # 0
    (-1.0, 0.5, 2.0, 0.30), # 1
    # (-1.0, 0.5, 2.0, 0.45), # 2
    (-1.0, 0.4, 2.0, 0.15), # 3
    (-1.0, 0.4, 2.0, 0.35), # 4
    (-1.0, 0.3, 2.0, 0.15), # 5
    (-1.0, 0.3, 2.0, 0.25), # 6

    (-0.5, 0.5, 2.0, 0.15), # 7
    (-0.5, 0.5, 2.0, 0.30), # 8
    (-0.5, 0.5, 2.0, 0.45), # 9
    (-0.5, 0.4, 2.0, 0.15), # 10
    (-0.5, 0.4, 2.0, 0.35), # 11
    (-0.5, 0.3, 2.0, 0.15), # 12
    (-0.5, 0.3, 2.0, 0.25), # 13
]

best_params = [
  (0.5, 0.3, 0.3, 1.0, True, 18.0), # 0
  (0.8, 0.4, 0.4, 1.0, True, 17.0), # 1
#   ??? # 2
  (0.7, 0.6, 0.1, 1.3, True, 18.0), # 3
  (0.7, 0.6, 0.2, 1.3, True, 18.0), # 4
  (0.8, 0.6, 0.2, 1.4, True, 16.0), # 5
  (0.8, 0.6, 0.2, 1.0, False, 16.0), # 6
  (0.9, 0.3, 0.2, 0.7, True, 15.0), # 7
  (0.9, 0.2, 0.2, 0.8, False, 16.0), # 8
  (0.9, 0.3, 0.2, 0.8, False, 16.0), # 9
  (0.9, 0.3, 0.2, 0.9, False, 14.0), # 10
  (0.9, 0.3, 0.2, 0.9, False, 15.0), # 11
  (0.9, 0.3, 0.2, 1.1, False, 14.0), # 12
  (0.9, 0.3, 0.2, 1.1, False, 15.0), # 13
]

folder_to_save_to = 'intersection_scenario_suite/'

# scenario_configs = scenario_configs[:1]
# best_params = best_params[:1]
# offset = [0, 1, 3, 5, 7, -1, -3, -5, -7]
offset = [0]
zero_faster = [True, False]
for scenario_config, mpc_params in zip(scenario_configs, best_params):
    # config.opp_gamma = mpc_params[0]
    # config.obs_gamma = mpc_params[1]
    # config.liveliness_gamma = mpc_params[2]
    # config.liveness_threshold = mpc_params[3]
    # config.mpc_use_opp_cbf = mpc_params[4]
    # config.runtime = mpc_params[5]
    for z in zero_faster:
        for o in offset:
            print(f"Running scenario {scenario_config} with z {z} and o {o}")
            # Don't include situations that won't happen.
            if o > 0 and z:
                continue
            if o < 0 and not z:
                continue

            config.agent_zero_offset = o
            config.mpc_p0_faster = z
            log_filename = f's_{scenario_config[0]}_{scenario_config[1]}_{scenario_config[2]}_{scenario_config[3]}_l_{0 if z else 1}_faster_off{o}.json'
            logger = DataLogger(os.path.join(folder_to_save_to, log_filename))

            # scenario = DoorwayScenario(initial_x=scenario_config[0], initial_y=scenario_config[1], goal_x=scenario_config[2], goal_y=scenario_config[3])
            scenario = IntersectionScenario()


            # Matplotlib plotting handler
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

            run_simulation(scenario, env, controllers, logger, plotter)
            plt.close()
