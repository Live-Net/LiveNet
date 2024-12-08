import os
import numpy as np

def delete(arr, idxs):
    mask = np.ones(len(arr), dtype=bool)
    mask[idxs] = False
    return arr[mask,...]

def delete_bad_instances(metrics):
    metrics = delete(metrics, np.where(metrics[:, 4] < 0))
    metrics = delete(metrics, np.where(metrics[:, 5] < 0))
    metrics = delete(metrics, np.where(metrics[:, 7] > 0.5))
    metrics = delete(metrics, np.where(metrics[:, 9] > 0.5))
    metrics = delete(metrics, np.where(metrics[:, 11] > 0.5))
    return metrics

def get_top(metrics, reflist, num, minimize):
    sorted_idxs = reflist.argsort()
    if not minimize:
        sorted_idxs = sorted_idxs[::-1]
    if len(sorted_idxs) > num:
        sorted_idxs = sorted_idxs[:num]
    return sorted_idxs, metrics[sorted_idxs, :4]


folder = "scenario_generation3/"

scenarios = [(0.0, 0.5, 2.0, 0.15),
        (0.0, 0.5, 2.0, 0.30),
        (0.0, 0.5, 2.0, 0.45),
        (0.0, 0.4, 2.0, 0.15),
        (0.0, 0.4, 2.0, 0.35),
        (0.0, 0.3, 2.0, 0.15),
        (0.0, 0.3, 2.0, 0.25)]

# filename = "scenario_0.0_0.3_3.0_0.15.csv"
# scenario = tuple(float(i) for i in filename.lstrip("scenario_").rstrip(".csv").split("_"))

scenario = scenarios[0]
filename = f"scenario_{scenario[0]}_{scenario[1]}_{scenario[2]}_{scenario[3]}.csv"

metrics = np.loadtxt(os.path.join(folder, filename), delimiter=',')

# opp_gamma, obs_gamma, live_gamma, liveness_thresh, goal_reach_idx0, goal_reach_idx1, min_agent_dist, traj_collision, obs_min_dist_0, obs_collision_0, obs_min_dist_1, obs_collision_1, delta_vel_0, delta_vel_1, path_dev_0, path_dev_1
print(metrics.shape)
metrics = delete_bad_instances(metrics)
print(metrics.shape)
print(metrics)

count = 20

print("Top 10 by traj min agent dist:")
print(get_top(metrics, metrics[:, 6], count, False))

print("Top 10 by obs agent dist:")
print(get_top(metrics, np.max(metrics[:, [8,10]], axis=1), count, False))

print("Top 10 by path deviation:")
idxs = get_top(metrics, np.min(metrics[:, [14,15]], axis=1), count, True)
print(idxs)
print(metrics)
print(metrics[idxs, [14,15]])

print("Top 10 by velocity diff:")
idxs = get_top(metrics, np.min(metrics[:, [12,13]], axis=1), count, True)
print(idxs)
print(metrics[idxs, [12,13]])

