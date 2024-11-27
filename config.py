"""Configurations for the MPC controller."""

import os
import torch
import numpy as np
from enum import Enum, auto

class DynamicsModel(Enum):
    SINGLE_INTEGRATOR = auto()
    DOUBLE_INTEGRATOR = auto()
    DOUBLE_INTEGRATOR_MACBF = auto()
    DOUBLE_INTEGRATOR_PIC = auto()

# Liveness parameters.
liveliness = True
liveness_threshold = 0.7
plot_rate = 1
plot_live = True
plot_live_pause_iteration = None
# plot_live_pause_iteration = 0
# plot_live_pause_iteration = 85
plot_arrows = False
plot_end = True
plot_end_ani_only = True
plot_text_on = True
# plot_text_on = False
ani_save_name = 'TEST.mp4'

# dynamics = DynamicsModel.SINGLE_INTEGRATOR
# dynamics = DynamicsModel.DOUBLE_INTEGRATOR
# dynamics = DynamicsModel.DOUBLE_INTEGRATOR_MACBF
dynamics = DynamicsModel.DOUBLE_INTEGRATOR_PIC

mpc_p0_faster = True
agent_zero_offset = 0
consider_intersects = True
mpc_use_new_liveness_filter = True
mpc_static_obs_non_cbf_constraint = False

if dynamics == DynamicsModel.SINGLE_INTEGRATOR:
    num_states = 3  # [x, y, theta]
    num_controls = 2  # (v, omega)
elif dynamics == DynamicsModel.DOUBLE_INTEGRATOR:
    num_states = 4  # [x, y, theta, v]
    num_controls = 2  # (a, omega)
elif dynamics == DynamicsModel.DOUBLE_INTEGRATOR_MACBF:
    num_states = 4  # [x, y, vx, vy]
    num_controls = 2  # [ax, ay]
elif dynamics == DynamicsModel.DOUBLE_INTEGRATOR_PIC:
    num_states = 4  # [x, y, vx, vy]
    num_controls = 2  # [ax, ay]

else:
    raise ValueError("Unsupported DynamicsModel selected.")

num_states = num_states
num_controls = num_controls

plot_text_on = True


n = 2                                      # Number of agents
runtime = 18.0                             # Total runtime [s]

if dynamics == DynamicsModel.DOUBLE_INTEGRATOR_MACBF or dynamics == DynamicsModel.DOUBLE_INTEGRATOR_PIC:
    sim_ts = 0.1
else:
    sim_ts = 0.2                                # Simulation Sampling time [s]

MPC_Ts = 0.1                                   # MPC Sampling time [s]
T_horizon = 6                              # Prediction horizon time steps

obstacle_avoidance = True
mpc_use_opp_cbf = True
# Gamma, in essence, is the leniancy on how much we can deprove the CBF.
opp_gamma = 0.6                            # CBF parameter in [0,1]
obs_gamma = 0.3                            # CBF parameter in [0,1]
liveliness_gamma = 0.1                     # CBF parameter in [0,1]
# safety_dist = 0.00                         # Safety distance
# agent_radius = 0.01                         # Robot radius (for obstacle avoidance)
mpc_liveness_safety_buffer = 0.03
safety_dist = 0.0                         # Safety distance
agent_radius = 0.1                         # Robot radius (for obstacle avoidance)
zeta = 3.0

# Actuator limits
apply_control_limit = False
apply_state_limit = False
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
        "Q": np.diag([20.0, 20.0, 0.0, 20.0]),  # State cost matrix
        "R": np.array([2.0, 5.0]),                  # Controls cost matrix
    }
}

# Training parameters.
use_barriernet = True

# train_data_paths = ['doorway_scenario_suite/s_-1.0_0.5_2.0_0.15_l_0_faster_off0.json', 'doorway_scenario_suite/s_-1.0_0.4_2.0_0.15_l_0_faster_off0.json', 'doorway_scenario_suite/s_-1.0_0.3_2.0_0.15_l_0_faster_off0.json',
#                     'doorway_scenario_suite/s_-0.5_0.5_2.0_0.15_l_0_faster_off0.json', 'doorway_scenario_suite/s_-0.5_0.4_2.0_0.15_l_0_faster_off0.json', 'doorway_scenario_suite/s_-0.5_0.3_2.0_0.15_l_0_faster_off0.json', ]

# train_data_paths = ['doorway_scenario_suite/']
train_data_paths = ['doorway_scenario_suite2/']
# train_data_paths = ['intersection_scenario_suite2/']
# train_data_paths = ['intersection_scenario_suite3/']

# train_data_paths = []
# for filename in os.listdir('doorway_scenario_suite'):
#     if '0_faster' in filename:
#     # if '1_faster' in filename:
#         train_data_paths.append(os.path.join('doorway_scenario_suite', filename))

# agents_to_train_on = [0]
agents_to_train_on = [0, 1]

# Liveness / CBF Filters (all the cool shit)
add_control_limits = False
add_liveness_filter = True
add_liveness_as_input = False
fixed_liveness_input = True

# Changing the inputs / outputs
x_is_d_goal = True
n_opponents = 12
separate_penalty_for_opp = False
static_obs_xy_only = True
ego_frame_inputs = True

train_batch_size = 32
# train_batch_size = 1
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

''' In order to use MPS, change all .double() to float() '''
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )


learning_rate = 1e-3
epochs = 30
nHidden1 = 256
nHidden21 = 128
nHidden22 = 64
nHidden23 = 64
nHidden24 = 64
# l = liveness, nl = no liveness
# g = goal, ng = no goal
# saf = trained on both slow and fast variations.
# wc = with checkpoint
# 
# Baseline options: weights/model_base_single_input_obs_wc_nolim_
# Baseline options: weights/model_base_single_input_obs_wc_nolim_saf_
# Baseline options: weights/model_base_single_input_obs_wc_nolim_saf_suite_0_1
# Baseline options: weights/model_base_single_input_obs_wc_nolim_saf_suite_0 and weights/model_base_single_input_obs_wc_nolim_saf_suite_1

# Livenet options: model_base_single_input_obs_wc_nolim_linp_f_fullsuite_0_1_bn_definition

# saveprefix = f'weights/model_base_single_input_obs_wc_nolim_linp_f_fullsuite_live_'
# saveprefix = f'weights/model_base_input_obs_wc_nolim_saf_intersuite_'
saveprefix = f'weights/model_30_norm_doorsuite2_lfnew_so_ego_'
saveprefix += '_'.join([str(i) for i in agents_to_train_on])
# saveprefix = "weights/test"


description = "Base model, no limits, no liveness, obs are inputs, run on intersection suite"
