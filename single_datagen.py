"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import os
import numpy as np
import matplotlib.pyplot as plt
import config
from mpc_cbf import MPC
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from environment import Environment
from simulation import run_simulation

ref_file = 'obs_doorway_with_offsets/l_0_faster_off0.json'
ref_logger = DataLogger(ref_file)
folder_to_save_to = 'obs_doorway_with_offsets/'

config.agent_zero_offset = 0
config.mpc_p0_faster = True
# logger = BlankLogger()
logger = DataLogger(os.path.join(folder_to_save_to, f'l_0_faster_edge_cases.json'))

starting_states = [
    ([
        1.8084002277181483,
        -0.14664168367093852,
        -0.08916022175398637,
        0.11579714401470648
    ],
    [
        1.217295959270444,
        0.059611071267492245,
        0.1678623119667841,
        0.3
    ], 4.0),
    ([
        1.8525831602122353,
        -0.15054079705630236,
        -0.08467670546605474,
        0.09680908303388229
    ],
    [
        1.327262955516918,
        0.07764581985526209,
        0.14858513997807246,
        0.3
    ], 3.0),
    ([
        1.905040537225743,
        -0.15490605131338286,
        -0.07980280162539194,
        0.06988481108921601
    ],
    [
        1.4875561869008527,
        0.1005026695004883,
        0.12918509289630725,
        0.3
    ], 2.5),
    ([
        1.9312497088745468,
        -0.15698704990769338,
        -0.07754262593934155,
        0.05383987856327315
    ],
    [
        1.580254262820569,
        0.11233075776684293,
        0.12040271208599465,
        0.3
    ], 2.0)
]

for (starting_0, starting_1, runtime) in starting_states:
    scenario = DoorwayScenario()
    # scenario = NoObstacleDoorwayScenario()
    scenario.initial = np.array([starting_0, starting_1])
    config.runtime = runtime

    # Matplotlib plotting handler
    plotter = Plotter()
    # plotter = None

    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    goals = scenario.goals.copy()
    logger.set_obstacles(scenario.obstacles.copy())
    env = Environment(scenario.initial.copy(), scenario.goals.copy())
    controllers = []

    # Setup agents
    controllers.append(MPC(agent_idx=0, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[0,:], static_obs=scenario.obstacles.copy()))
    controllers.append(MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[1,:], static_obs=scenario.obstacles.copy()))

    run_simulation(scenario, env, controllers, logger, plotter)
    plt.close()
