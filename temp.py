import os
import numpy as np
from data_logger import DataLogger
from metrics import gather_all_metric_data
from scenarios import DoorwayScenario

metrics = []
folder = 'doorway_scenario_suite_5/'
for filename in os.listdir(folder):
    if not filename.endswith('json'):
        continue
    filepath = os.path.join(folder, filename)
    logger = DataLogger.load_file(filepath)
    x0_cum = np.array([iteration['states'][0] for iteration in logger.data['iterations']])
    x1_cum = np.array([iteration['states'][1] for iteration in logger.data['iterations']])
    compute_history = np.array([[0.0, 0.0] for _ in range(len(x0_cum))])
    conf_str = filename.lstrip('s_doorway_')
    conf_str = conf_str[:conf_str.index('l_') - 1]
    conf = [True if i == "True" else False if i == "False" else float(i) for i in conf_str.split("_")]
    scenario = DoorwayScenario(initial_x=conf[0], initial_y=conf[1], goal_x=conf[2], goal_y=conf[3], initial_vel=conf[4], start_facing_goal=conf[5])
    metrics.append(gather_all_metric_data(scenario, x0_cum, x1_cum, scenario.goals, compute_history))
metrics = np.array(metrics)


# all_metrics = []
# for i in range(0, 56, 14):
#     metrics = np.loadtxt(f'doorway_scenario_suite_5_{i}_{i+14}.csv', delimiter=',')
#     all_metrics.extend(metrics)
# metrics = np.array(all_metrics)

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

print(sorted(metrics[:, IDXS['obs_min_dist_0']]))
closest_0 = np.min(metrics[:, IDXS['obs_min_dist_0']])
closest_1 = np.min(metrics[:, IDXS['obs_min_dist_1']])
closest_agent = np.min(metrics[:, IDXS['min_agent_dist']])

print(f"Number of passing scenarios: {passed} / {num_sims}")
print(f"Closest obs dist agent 0: {closest_0}")
print(f"Closest obs dist agent 1: {closest_1}")
print(f"Closest agent dist: {closest_agent}")
