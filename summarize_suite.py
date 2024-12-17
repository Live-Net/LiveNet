import numpy as np

# AGENT = 'MPC'
AGENT = 'LiveNet'

SCENARIO = 'Doorway'

metrics = np.loadtxt(f'experiment_results/{AGENT}_{SCENARIO}_suite.csv', delimiter=',')

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
