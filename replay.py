"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import os
import numpy as np
import config
import matplotlib.pyplot as plt
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from util import calculate_all_metrics

# dirname = 'all_data_with_offsets'
# dirname = 'obs_doorway_with_offsets'
# dirname = 'doorway_scenario_suite'
# dirname = 'intersection_scenario_suite'
# dirname = 'doorway_scenario_suite2'
# dirname = 'doorway_scenario_suite3'
dirname = 'intersection_scenario_suite2'

bags = []
for filename in os.listdir(dirname):
    bags.append(os.path.join(dirname, filename))
bags.sort()

# bags = ['obs_doorway_with_offsets/l_0_faster_off0.json']
# bags = ['doorway_scenario_suite/s_-1.0_0.5_2.0_0.15_l_0_faster_off0.json']
# bags = ['obs_doorway_with_offsets/l_0_faster_edge_cases.json']

# bags = ['doorway_scenario_suite/s_-1.0_0.3_2.0_0.15_l_0_faster_off0.json']

# bags = [
#     'doorway_scenario_suite3/s_doorway_-1.0_0.5_2.0_0.45_l_0_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-1.0_0.5_2.0_0.45_l_1_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-1.0_0.4_2.0_0.35_l_0_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-1.0_0.4_2.0_0.35_l_1_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.5_2.0_0.15_l_0_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.5_2.0_0.15_l_1_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.5_2.0_0.3_l_0_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.5_2.0_0.3_l_1_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.5_2.0_0.45_l_0_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.5_2.0_0.45_l_1_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.4_2.0_0.35_l_0_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.4_2.0_0.35_l_1_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.3_2.0_0.25_l_0_faster_off0.json',
#     'doorway_scenario_suite3/s_doorway_-0.5_0.3_2.0_0.25_l_1_faster_off0.json'
# ]

for bag in bags:
    print("Viewing", bag)
    config.ani_save_name = bag.rstrip('json') + '.mp4'
    # scenario = NoObstacleDoorwayScenario()

    # Matplotlib plotting handler
    plotter = Plotter()
    # logger = DataLogger.load_file('doorway_train_data_with_liveness_0_faster.json')
    logger = DataLogger.load_file(bag)
    scenario = DoorwayScenario()
    # scenario = IntersectionScenario()
    scenario.goals = np.array(logger.data['iterations'][0]['goals'])
    # logger = DataLogger.load_file('all_data_with_offsets/doorway_train_data_with_liveness_0_faster_off0.json')
    output_logger = BlankLogger()

    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    scenario.obstacles = logger.data['obstacles'].copy()
    goals = scenario.goals.copy()
    # logger.set_obstacles(scenario.obstacles.copy())

    x_cum = [[], []]
    u_cum = [[], []]
    metrics = []
    for sim_iteration, iteration in enumerate(logger.data['iterations']):
        for agent_idx, state in enumerate(iteration['states']):
            x_cum[agent_idx].append(np.array(state))

        for agent_idx, controls in enumerate(iteration['controls']):
            u_cum[agent_idx].append(np.array(controls))

        metrics.append(calculate_all_metrics(x_cum[0][-1], x_cum[1][-1], config.liveness_threshold))

        # Plots
        if sim_iteration % config.plot_rate == 0 and config.plot_live and plotter is not None:
            plotter.plot_live(sim_iteration, scenario, x_cum, u_cum, metrics)

    # Discard the first element of both x1 and x2
    x_cum = np.array(x_cum)
    u_cum = np.array(u_cum)
    if config.plot_end and plotter is not None:
        plotter.plot(scenario, x_cum, u_cum, metrics)
    plt.close()

