import config
import numpy as np

SCENARIO = 'Doorway'
# SCENARIO = 'Intersection'

# AGENT = 'MPC'
# AGENT = 'MPC_UNLIVE'
# AGENT = 'BarrierNet'
# AGENT = 'LiveNet'
# AGENT = 'MACBF'
AGENT = 'PIC'

file_path = f'experiment_results/{AGENT}_{SCENARIO}_limits.csv'
print(f"Loading metrics from: {file_path}")

# Load metrics robustly
metrics = np.genfromtxt(file_path, delimiter=',', skip_header=1)

# Check the shape of the loaded data
print("Initial Metrics shape:", metrics.shape)

# If metrics is 1D, reshape it to 2D with one row
if metrics.ndim == 1:
    metrics = metrics.reshape(1, -1)
    print("Reshaped Metrics to 2D:", metrics.shape)

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

# Get number of simulations
num_sims = metrics.shape[0]

# Get number of collisions
collision_rows = np.any(metrics[:, [IDXS['traj_collision'], IDXS['obs_collision_0'], IDXS['obs_collision_1']]] > 0, axis=1)
collisions = np.sum(collision_rows)

# Get number of deadlocks
goal_reach_idxs = metrics[:, [IDXS['goal_reach_idx0'], IDXS['goal_reach_idx1']]]
deadlock_rows = np.any(goal_reach_idxs == -1, axis=1)
deadlocks = np.sum(deadlock_rows)

# Get slowest TTG. Consider only non-deadlock scenarios.
reached_idxs = goal_reach_idxs[np.all(goal_reach_idxs != -1, axis=1)]
if reached_idxs.size > 0:
    slower_ttgs = np.max(reached_idxs, axis=1) * config.sim_ts
    avg_slower_ttg = np.average(slower_ttgs)
    err_slower_ttg = np.std(slower_ttgs) / np.sqrt(slower_ttgs.size)
else:
    avg_slower_ttg = float('nan')
    err_slower_ttg = float('nan')

# Get average delta V.
deltaVs = metrics[:, [IDXS['delta_vel_0'], IDXS['delta_vel_1']]].flatten()
avg_delta_v = np.average(deltaVs)
err_delta_v = np.std(deltaVs) / np.sqrt(deltaVs.size)

# Get average delta path deviation.
deltaPaths = metrics[:, [IDXS['path_dev_0'], IDXS['path_dev_1']]].flatten()
avg_delta_path = np.average(deltaPaths)
err_delta_path = np.std(deltaPaths) / np.sqrt(deltaPaths.size)

# Get average compute time.
compute_times = metrics[:, [IDXS['avg_compute_0'], IDXS['avg_compute_1']]].flatten() * 1000.0  # S -> MS conversion
avg_compute_time = np.average(compute_times)
err_compute_time = np.std(compute_times) / np.sqrt(compute_times.size)

print(f"\nAccumulated metrics for {AGENT} agents in {SCENARIO} scenario")
print("Num simulations run:", num_sims)
print("Number of collisions:", collisions)
print("Number of deadlocks:", deadlocks)
print(f"Slower TTG: {avg_slower_ttg} +/- {err_slower_ttg}")
print(f"Delta Velocity: {avg_delta_v} +/- {err_delta_v}")
print(f"Path Deviation: {avg_delta_path} +/- {err_delta_path}")
print(f"Compute Time: {avg_compute_time} +/- {err_compute_time}")
