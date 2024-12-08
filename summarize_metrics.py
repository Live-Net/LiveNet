import config
import numpy as np

# AGENT = 'MPC'
AGENT = 'MPC_UNLIVE'
# AGENT = 'BarrierNet'
# AGENT = 'LiveNet'

SCENARIO = 'Doorway'
# SCENARIO = 'Intersection'

metrics = np.loadtxt(f'experiment_results/{AGENT}_{SCENARIO}.csv', delimiter=',')

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

# Get num collisions
collision_rows = np.any(metrics[:, [IDXS['traj_collision'], IDXS['obs_collision_0'], IDXS['obs_collision_1']]], axis=1)
collisions = np.sum(collision_rows)

# Get num deadlocks
goal_reach_idxs = metrics[:, [IDXS['goal_reach_idx0'], IDXS['goal_reach_idx1']]]
deadlock_rows = np.any(goal_reach_idxs == -1, axis=1)
deadlocks = np.sum(deadlock_rows)

# Get slowest TTG. Consider only non-deadlock scenarios.
reached_idxs = goal_reach_idxs[np.all(goal_reach_idxs != -1, axis=1)]
slower_ttgs = np.max(goal_reach_idxs, axis = 1) * config.sim_ts
avg_slower_ttg = np.average(slower_ttgs)
err_slower_ttg = np.std(slower_ttgs) / np.sqrt(slower_ttgs.size)

# Get average delta V.
deltaVs = metrics[:, [IDXS['delta_vel_0'], IDXS['delta_vel_1']]].flatten()
avg_delta_v = np.average(deltaVs)
err_delta_v = np.std(deltaVs) / np.sqrt(deltaVs.size)

# Get average delta path deviation.
deltaPaths = metrics[:, [IDXS['path_dev_0'], IDXS['path_dev_1']]].flatten()
avg_delta_path = np.average(deltaPaths)
err_delta_path = np.std(deltaPaths) / np.sqrt(deltaPaths.size)

# Get average compute time.
compute_times = metrics[:, [IDXS['avg_compute_0'], IDXS['avg_compute_1']]].flatten() * 1000.0 # S -> MS conversion
avg_compute_time = np.average(compute_times)
err_compute_time = np.std(compute_times) / np.sqrt(compute_times.size)

print(f"Accumulated metrics for {AGENT} agents in {SCENARIO} scenario")
print("Num simulations run:", num_sims)
print("Number of collisions:", collisions)
print("Number of deadlocks:", deadlocks)
print(f"Slower TTG: {avg_slower_ttg} +/- {err_slower_ttg}")
print(f"Delta Velocity: {avg_delta_v} +/- {err_delta_v}")
print(f"Path Deviation: {avg_delta_path} +/- {err_delta_path}")
print(f"Compute Time: {avg_compute_time} +/- {err_compute_time}")

# number of collisions
# number of deadlocks
# slower ttg
# avg delta V
# avg path deviation

