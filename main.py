"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import config
import numpy as np
from mpc_cbf import MPC
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from environment import Environment
from model_controller import ModelController

def run_simulation(scenario, env, controllers, logger, plotter):
    x_cum = [[], []]
    u_cum = [[], []]

    controllers[0].initialize_controller(env)
    controllers[1].initialize_controller(env)

    for sim_iteration in range(config.sim_steps):
        print(f"\nIteration: {sim_iteration}")
        for agent_idx in range(config.n):
            x_cum[agent_idx].append(env.initial_states[agent_idx])

        new_states, outputted_controls = env.run_simulation(sim_iteration, controllers, logger)

        for agent_idx in range(config.n):
            u_cum[agent_idx].append(outputted_controls[agent_idx])

        # Plots
        if sim_iteration % config.plot_rate == 0 and config.plot_live:
            plotter.plot_live(scenario, controllers, x_cum, u_cum)

    # Discard the first element of both x1 and x2
    x_cum = np.array(x_cum)
    u_cum = np.array(u_cum)
    if config.plot_end:
        plotter.plot(scenario, controllers, x_cum, u_cum)


# Scenarios: "doorway" or "intersection"
scenario = DoorwayScenario()
# scenario = NoObstacleDoorwayScenario(rotation=np.pi/2)
# scenario = NoObstacleDoorwayScenario()
# scenario = IntersectionScenario()

# Matplotlib plotting handler
plotter = Plotter()
if config.save_data_path is None:
    logger = BlankLogger()
else:
    logger = DataLogger(config.save_data_path)

# Add all initial and goal positions of the agents here (Format: [x, y, theta])
goals = scenario.goals.copy()
logger.set_obstacles(scenario.obstacles.copy())
env = Environment(scenario.initial.copy(), scenario.goals.copy())
controllers = []

# Setup agent 0
controllers.append(MPC(agent_idx=0, goal=goals[0,:], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_liveness_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("model_l_saf_0_bn_definition.json", static_obs=scenario.obstacles.copy()))

# Setup agent 1
# controllers.append(MPC(agent_idx=1, goal=goals[1,:], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_liveness_1_bn_definition.json", static_obs=scenario.obstacles.copy()))
controllers.append(ModelController("model_l_saf_1_bn_definition.json", static_obs=scenario.obstacles.copy()))

run_simulation(scenario, env, controllers, logger, plotter)
