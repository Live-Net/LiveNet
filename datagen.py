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

folder_to_save_to = 'obs_doorway_with_offsets/'

# start x, start y, goal x, goal y, opp gamma, obs gamma, liveness gamma
scenario_params = [
    (-1, 0.5, 2, 0.15, 0.5, 0.5, 0.3),
    (-1, 0.5, 2, 0.25, 0.5, 0.5, 0.3),
    (-1, 0.5, 2, 0.35, 0.5, 0.3),
    (-1, 0.5, 2, 0.15, 0.5, 0.3),
    (-1, 0.5, 2, 0.15, 0.5, 0.3),
]

offset = [0, 1, 3, 5, 7, -1, -3, -5, -7]
# offset = [0]
zero_faster = [True, False]
for z in zero_faster:
    for o in offset:
        # Don't include situations that won't happen.
        if o > 0 and z:
            continue
        if o < 0 and not z:
            continue

        config.agent_zero_offset = o
        config.mpc_p0_faster = z
        logger = DataLogger(os.path.join(folder_to_save_to, f'l_{0 if z else 1}_faster_off{o}.json'))

        scenario = DoorwayScenario()
        # scenario = NoObstacleDoorwayScenario()

        # Matplotlib plotting handler
        # plotter = Plotter()
        plotter = None

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
