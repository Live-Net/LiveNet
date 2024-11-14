"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import config
from mpc_cbf import MPC
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from environment import Environment
from model_controller import ModelController
from macbf_torch_controller import macbf_torch_controller

from macbf_controller import macbf_controller

from blank_controller import BlankController
from simulation import run_simulation

# Scenarios: "doorway" or "intersection"
scenario = DoorwayScenario()
# scenario = NoObstacleDoorwayScenario(rotation=np.pi/4)
# scenario = NoObstacleDoorwayScenario()
# scenario = IntersectionScenario()

# Matplotlib plotting handler
plotter = Plotter()
logger = BlankLogger()

# Add all initial and goal positions of the agents here (Format: [x, y, theta])
goals = scenario.goals.copy()
logger.set_obstacles(scenario.obstacles.copy())
env = Environment(scenario.initial.copy(), scenario.goals.copy())
controllers = []

# Setup agent 0
# controllers.append(BlankController())
# controllers.append(MPC(agent_idx=0, goal=goals[0,:], static_obs=scenario.obstacles.copy(), delay_start=max(config.agent_zero_offset, 0.0)))
# controllers.append(MPC(agent_idx=0, goal=goals[0,:], static_obs=scenario.obstacles.copy())) # Star
# controllers.append(ModelController("weights/model_liveness_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_g_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_o_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model2_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_obs_l_f_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(macbf_torch_controller("test", static_obs=scenario.obstacles.copy(), goal=goals[0,:]))
controllers.append(macbf_controller("test", static_obs=scenario.obstacles.copy(), goal=goals[0,:]))

# Setup agent 1
# controllers.append(MPC(agent_idx=1, goal=goals[1,:], static_obs=scenario.obstacles.copy(), delay_start=max(-config.agent_zero_offset, 0.0)))
# controllers.append(MPC(agent_idx=1, goal=goals[1,:], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_liveness_1_bn_definition.json", static_obs=scenario.obstacles.copy()))
# # controllers.append(ModelController("weights/model_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_g_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_o_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_newb_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_test2_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_smg_obs_l_s_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_base_w_lims_obs_l_s_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Star
# controllers.append(MPC(agent_idx=1, goal=goals[1,:], static_obs=scenario.obstacles.copy()))
# controllers.append(macbf_torch_controller("test2", static_obs=scenario.obstacles.copy(), goal=goals[1,:]))


# print("MONKEY BOX")
# print(goals)
# print(goals[1,:])
# print("")


controllers.append(macbf_controller("test2", static_obs=scenario.obstacles.copy(), goal=goals[1,:]))

run_simulation(scenario, env, controllers, logger, plotter)
