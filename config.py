"""Configurations for the MPC controller."""

import torch
import numpy as np
from enum import Enum, auto

class DynamicsModel(Enum):
    SINGLE_INTEGRATOR = auto()
    DOUBLE_INTEGRATOR = auto()
    DOUBLE_INTEGRATOR_MACBF = auto()
    INFORMARL = auto()

# Liveness parameters.
liveliness = True
liveness_threshold = 1.0
plot_rate = 1
plot_live = True
plot_live_pause = False
plot_arrows = False
plot_end = False
plot_end_ani_only = False
ani_save_name = 'base_barriernet_model.mp4'

# dynamics = DynamicsModel.SINGLE_INTEGRATOR
# dynamics = DynamicsModel.DOUBLE_INTEGRATOR
dynamics = DynamicsModel.DOUBLE_INTEGRATOR_MACBF
mpc_p0_faster = True
agent_zero_offset = 0
consider_intersects = True

if dynamics == DynamicsModel.SINGLE_INTEGRATOR:
    num_states = 3  # [x, y, theta]
    num_controls = 2  # (v, omega)
elif dynamics == DynamicsModel.DOUBLE_INTEGRATOR:
    num_states = 4  # [x, y, theta, v]
    num_controls = 2  # (a, omega)
elif dynamics == DynamicsModel.DOUBLE_INTEGRATOR_MACBF:
    num_states = 4  # [x, y, vx, vy]
    num_controls = 2  # [ax, ay]
elif dynamics == DynamicsModel.INFORMARL:
    num_states = 4  # [x, y, vx, vy]
    num_controls = 2  # [ax, ay]
else:
    raise ValueError("Unsupported DynamicsModel selected.")

num_states = num_states
num_controls = num_controls


n = 2                                      # Number of agents
runtime = 180.0                             # Total runtime [s]
sim_ts = 0.2                                # Simulation Sampling time [s]
MPC_Ts = 0.1                                   # MPC Sampling time [s]
T_horizon = 6                              # Prediction horizon time steps
sim_steps = int(runtime / sim_ts)              # Number of iteration steps for each agent

obstacle_avoidance = True
# Gamma, in essence, is the leniancy on how much we can deprove the CBF.
obs_gamma = 0.6                            # CBF parameter in [0,1]
liveliness_gamma = 0.3                     # CBF parameter in [0,1]
# safety_dist = 0.00                         # Safety distance
# agent_radius = 0.01                         # Robot radius (for obstacle avoidance)
safety_dist = 0.03                         # Safety distance
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
bn_model = "barriernet"
# bn_model = "barriernetdopp"
use_barriernet = True
# include_goal = True
smg_barriernet = True
include_goal = False
agent_to_train = 1

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
    # 'obs_doorway_with_offsets/l_0_faster_off-1.json',
    # 'obs_doorway_with_offsets/l_0_faster_off-3.json',
    # 'obs_doorway_with_offsets/l_0_faster_off-5.json',
    # 'obs_doorway_with_offsets/l_0_faster_off-7.json',
    'obs_doorway_with_offsets/l_0_faster_off0.json',
    # 'obs_doorway_with_offsets/l_0_faster_off5.json',
    # 'obs_doorway_with_offsets/l_0_faster_off7.json',
    'obs_doorway_with_offsets/l_0_faster_edge_cases.json',
    # 'obs_doorway_with_offsets/l_1_faster_off0.json',
]

train_batch_size = 32
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

''' In order to use MPS, change all .double() to float() '''
# device = torch.device(
#     "cuda" if torch.cuda.is_available() else
#     "mps" if torch.backends.mps.is_available() else
#     "cpu"
# )


learning_rate = 1e-3
epochs = 20
nHidden1 = 256
nHidden21 = 64
nHidden22 = 64
nHidden23 = 32
# l = liveness, nl = no liveness
# g = goal, ng = no goal
# saf = trained on both slow and fast variations.
saveprefix = f'weights/model_smg_obs_l_s_'
saveprefix += str(agent_to_train)
