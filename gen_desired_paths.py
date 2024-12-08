"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import config
from plotter import Plotter
from data_logger import DataLogger
from environment import Environment
from blank_controller import BlankController
from simulation import run_simulation
from mpc_cbf import MPC
from metrics import get_straight_line_desired_path
from run_experiments import get_mpc_live_controllers, get_scenario, get_livenet_controllers

NUM_AGENTS = 2

# AGENT = "MPC"
# AGENT = "LiveNet"
AGENT = "MPC_UNLIVE"
# AGENT = "BarrierNet"

SCENARIO = "Doorway"
# SCENARIO = "Intersection"

scenario = get_scenario(SCENARIO)

for agent_idx in range(2):
    config.mpc_p0_faster = agent_idx == 0
    plotter = Plotter()
    config.ani_save_name = f"Desired_{SCENARIO}_{AGENT}_{agent_idx}.mp4"
    logger = DataLogger(f"experiment_results/desired_paths/{SCENARIO}_{AGENT}_{agent_idx}.json")
    logger.set_obstacles(scenario.obstacles.copy())
    env = Environment(scenario.initial.copy(), scenario.goals.copy())
    controllers = []

    if SCENARIO == "Doorway":
        if AGENT == "MPC" or AGENT == "MPC_UNLIVE" or AGENT == "BarrierNet":
            controllers = get_mpc_live_controllers(scenario, config.mpc_p0_faster)
            other_agent = 1 - agent_idx
            scenario.initial[other_agent][3] = 0.0
            env.initial_states[other_agent][3] = 0.0
            controllers[other_agent] = BlankController()
        elif AGENT == "LiveNet":
            config.liveliness = True
            config.opp_gamma = 0.5
            config.obs_gamma = 0.3
            config.liveliness_gamma = 0.3
            config.mpc_use_new_liveness_filter = True

            controllers = get_livenet_controllers(scenario, SCENARIO)
            controllers[1] = MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=scenario.goals[1,:].copy(), static_obs=scenario.obstacles.copy())

            if agent_idx == 0:
                scenario.initial[1][3] = 0.0
                env.initial_states[1][3] = 0.0

    elif SCENARIO == "Intersection":
        if AGENT == "MPC" or AGENT == "LiveNet" or AGENT == "MPC_UNLIVE" or AGENT == "BarrierNet":
            controllers = get_mpc_live_controllers(scenario, config.mpc_p0_faster)
            other_agent = 1 - agent_idx
            scenario.initial[other_agent][3] = 0.0
            env.initial_states[other_agent][3] = 0.0
            controllers[other_agent] = BlankController()

    run_simulation(scenario, env, controllers, logger, plotter)
