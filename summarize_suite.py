import config
import numpy as np

# AGENT = 'MPC'
# AGENT = 'MPC_UNLIVE'
# AGENT = 'BarrierNet'
AGENT = 'LiveNet'

SCENARIO = 'Doorway'
# SCENARIO = 'Intersection'

metrics = np.loadtxt(f'experiment_results/MPC_Doorway_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/{AGENT}_{SCENARIO}_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_model_30_norm_doorsuite4_lfnew_nso_nego_0_1_bn_definition_suite.csv', delimiter=',') # 23/56

# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_model_40_norm_doorsuite4_lfnew_nso_nego_wl_0_1_bn_definition_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_model_40_norm_doorsuite4_lfnew_nso_nego_8o_0_1_bn_definition_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_model_40_norm_doorsuite4_lfnew_nso_nego_seppen_0_1_bn_definition_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_model_35_norm_doorsuite4_lfnew_nso_nego_8o_small_0_1_bn_definition_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_model_35_norm_doorsuite4_lfnew_nso_nego_wnewl_small_0_1_bn_definition_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_5_3opp_od_0_1_bn_definition_suite.csv', delimiter=',') # GOATED 28/56
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_6_3opp_od_seploop_0_1_bn_definition_suite.csv', delimiter=',')

# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_6_3opp_od_seploop_suite5_0_1_bn_definition_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_7_suite5_0_1_bn_definition_suite.csv', delimiter=',') # 32 / 56
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_7_nol_suite5_0_1_bn_definition_suite.csv', delimiter=',') # 35 / 56 CASES. 25/28 of the LAST CASES

# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_8_6nol_suite5_0_1_bn_definition_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_8_6l_suite5_0_1_bn_definition_suite.csv', delimiter=',') # 13 / 28
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_8_nol_suite5_0_1_bn_definition_suite.csv', delimiter=',')
# metrics = np.loadtxt(f'experiment_results/LiveNet_Doorway_srikar_iter_8_l_suite5_0_1_bn_definition_suite.csv', delimiter=',')

metrics = metrics[28:]


IDXS = {
    'goal_reach_idx0': 0,
    'goal_reach_idx1': 1,
    'min_agent_dist': 2,
    'traj_collision': 3,
    'obs_min_dist_0': 4,
    'obs_collision_0': 5,
    'obs_min_dist_1': 6,
    'obs_collision_1': 7,
    'delta_vel_0': 8,
    'delta_vel_1': 9,
    'path_dev_0': 10,
    'path_dev_1': 11,
    'avg_compute_0': 12,
    'avg_compute_1': 13
}

# Get num sims
num_sims = len(metrics)

# Get num deadlocks / collisions
collision_rows = np.any(metrics[:, [IDXS['traj_collision'], IDXS['obs_collision_0'], IDXS['obs_collision_1']]], axis=1)
goal_reach_idxs = metrics[:, [IDXS['goal_reach_idx0'], IDXS['goal_reach_idx1']]]
deadlock_rows = np.any(goal_reach_idxs == -1, axis=1)

failed_rows = np.logical_or(collision_rows, deadlock_rows)
passed_rows = np.logical_not(failed_rows)
passed = np.sum(passed_rows)

print(f"Analyzed suite metrics for {AGENT} agents in {SCENARIO} scenario")
print(f"Number of passing scenarios: {passed} / {num_sims}")
print(passed_rows[:14])
print(passed_rows[14:28])
print(passed_rows[28:42])
print(passed_rows[42:56])
