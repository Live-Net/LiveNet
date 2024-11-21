"""Configurations for the MPC controller."""

import os
import torch
import numpy as np
from enum import Enum, auto

class DynamicsModel(Enum):
    SINGLE_INTEGRATOR = auto()
    DOUBLE_INTEGRATOR = auto()

# Liveness parameters.
liveliness = True
liveness_threshold = 1.0
plot_rate = 1
plot_live = True
plot_live_pause_iteration = None
# plot_live_pause_iteration = 0
# plot_live_pause_iteration = 35
plot_arrows = False
plot_end = False
plot_end_ani_only = False
ani_save_name = 'base_barriernet_model.mp4'

dynamics = DynamicsModel.DOUBLE_INTEGRATOR
mpc_p0_faster = True
agent_zero_offset = 0
consider_intersects = True

if dynamics == DynamicsModel.SINGLE_INTEGRATOR:
    num_states = 3 # (x, y, theta)
    num_controls = 2 # (v, omega)
else:
    num_states = 4 # (x, y, theta, v)
    num_controls = 2 # (a, omega)

n = 2                                      # Number of agents
runtime = 80.0                             # Total runtime [s]
sim_ts = 0.2                                # Simulation Sampling time [s]
MPC_Ts = 0.1                                   # MPC Sampling time [s]
T_horizon = 6                              # Prediction horizon time steps

obstacle_avoidance = True
mpc_use_opp_cbf = True
# Gamma, in essence, is the leniancy on how much we can deprove the CBF.
opp_gamma = 0.5                            # CBF parameter in [0,1]
obs_gamma = 0.3                            # CBF parameter in [0,1]
liveliness_gamma = 0.3                     # CBF parameter in [0,1]
# safety_dist = 0.00                         # Safety distance
# agent_radius = 0.01                         # Robot radius (for obstacle avoidance)
safety_dist = 0.0                         # Safety distance
agent_radius = 0.1                         # Robot radius (for obstacle avoidance)
zeta = 3.0

# Actuator limits
v_limit = 0.30                             # Linear velocity limit
omega_limit = 0.5                          # Angular velocity limit
accel_limit = 0.1

# ------------------------------------------------------------------------------
COST_MATRICES = {
    # DynamicsModel.SINGLE_INTEGRATOR: {
    #     "Q": np.diag([15, 15, 0.005]),  # State cost matrix DOORWAY
    #     # "Q": np.diag([100, 100, 11]), # State cost matrix INTERSECTION
    #     "R": np.array([3, 1.5]),                  # Controls cost matrix
    # },
    DynamicsModel.DOUBLE_INTEGRATOR: {
        "Q": np.diag([20.0, 20.0, 0.0, 20.0]),  # State cost matrix DOORWAY
        # "Q": np.diag([100, 100, 11, 3]), # State cost matrix INTERSECTION
        "R": np.array([2.0, 5.0]),                  # Controls cost matrix
    }
}

# Training parameters.
use_barriernet = True
# agent_to_train = 1

# train_data_paths = ['doorway_train_data_with_liveness_0_faster.json', 'doorway_train_data_with_liveness_1_faster.json']
# train_data_paths = ['all_data_with_offsets/']
# train_data_paths = ['all_data_with_offsets/doorway_train_data_with_liveness_0_faster_off0.json', 'all_data_with_offsets/doorway_train_data_with_liveness_1_faster_off0.json']
# train_data_paths = ['all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off0.json', 'all_data_with_offsets/test_doorway_train_data_with_liveness_1_faster_off0.json']
# train_data_paths = ['all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off0.json',
#                     'all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off3.json',
#                     'all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off5.json',
#                     'all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off7.json',
#                     'all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off-1.json',
#                     'all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off-3.json',
#                     'all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off-5.json',
#                     'all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off-7.json']
# train_data_paths = ['obs_doorway_with_offsets/']
train_data_paths = [
#     # No liveness cases
#     'obs_doorway_with_offsets/l_0_faster_off-1.json',
#     'obs_doorway_with_offsets/l_0_faster_off-3.json',
#     'obs_doorway_with_offsets/l_0_faster_off-5.json',
#     'obs_doorway_with_offsets/l_0_faster_off-7.json',
#     'obs_doorway_with_offsets/l_1_faster_off1.json',
#     'obs_doorway_with_offsets/l_1_faster_off3.json',
#     'obs_doorway_with_offsets/l_1_faster_off5.json',
#     'obs_doorway_with_offsets/l_1_faster_off7.json',
#     # Liveness cases
    # 'obs_doorway_with_offsets/l_0_faster_off0.json',
    'doorway_scenario_suite/s_-1.0_0.5_2.0_0.15_l_0_faster_off0.json',
    'obs_doorway_with_offsets/l_0_faster_edge_cases.json',
#     'obs_doorway_with_offsets/l_1_faster_off0.json',
]

# train_data_paths = ['doorway_scenario_suite/s_-1.0_0.5_2.0_0.15_l_0_faster_off0.json', 'doorway_scenario_suite/s_-1.0_0.4_2.0_0.15_l_0_faster_off0.json', 'doorway_scenario_suite/s_-1.0_0.3_2.0_0.15_l_0_faster_off0.json',
#                     'doorway_scenario_suite/s_-0.5_0.5_2.0_0.15_l_0_faster_off0.json', 'doorway_scenario_suite/s_-0.5_0.4_2.0_0.15_l_0_faster_off0.json', 'doorway_scenario_suite/s_-0.5_0.3_2.0_0.15_l_0_faster_off0.json', ]

# train_data_paths = ['doorway_scenario_suite/']

# train_data_paths = []
# for filename in os.listdir('doorway_scenario_suite'):
#     if '0_faster' in filename:
#     # if '1_faster' in filename:
#         train_data_paths.append(os.path.join('doorway_scenario_suite', filename))

agents_to_train_on = [0]
# agents_to_train_on = [0, 1]

x_is_d_goal = True
separate_penalty_for_opp = True

# CBF Filters
add_control_limits = False
add_liveness_filter = False

# Changing the inputs / outputs
vx_vy_inputs = False
ax_ay_output = False
add_liveness_as_input = False
n_obs_as_input = 16

n_closest_obs = None
# n_closest_obs = 5

train_batch_size = 32
train_append_goal_xy = False
# train_batch_size = 1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
learning_rate = 1e-3
epochs = 30
nInputs = 8 + add_liveness_as_input
nHidden1 = 256
nHidden21 = 64
nHidden22 = 64
nHidden23 = 64
nHidden24 = 64
# l = liveness, nl = no liveness
# g = goal, ng = no goal
# saf = trained on both slow and fast variations.
saveprefix = f'weights/modelf_base_single_all_obs_nolim_'
saveprefix += '_'.join([str(i) for i in agents_to_train_on])

description = "Base model, no limits, all obs, single scenario, fixed Lf2b bug"
