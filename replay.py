import os
import numpy as np
import config
import matplotlib.pyplot as plt
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from util import calculate_all_metrics

# dirname = 'doorway_scenario_suite5'
dirname = 'intersection_scenario_suite2'

bags = []
for filename in os.listdir(dirname):
    bags.append(os.path.join(dirname, filename))
bags.sort()

for bag in bags:
    print("Viewing", bag)
    config.ani_save_name = bag.rstrip('json') + '.mp4'

    # Matplotlib plotting handler
    plotter = Plotter()
    logger = DataLogger.load_file(bag)
    scenario = DoorwayScenario()
    # scenario = IntersectionScenario()
    # scenario = NoObstacleDoorwayScenario()

    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    scenario.goals = np.array(logger.data['iterations'][0]['goals'])
    output_logger = BlankLogger()
    scenario.obstacles = logger.data['obstacles'].copy()
    goals = scenario.goals.copy()

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
