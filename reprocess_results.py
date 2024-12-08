import config
import numpy as np
from metrics import check_when_reached_goal
from data_logger import DataLogger
from util import get_ray_intersection_point
from metrics import calculate_avg_delta_vel, calculate_path_deviation

# SCENARIO = 'Doorway'
SCENARIO = 'Intersection'

RUN_AGENT = 'MPC'
# RUN_AGENT = 'LiveNet'
# RUN_AGENT = 'BarrierNet'
# RUN_AGENT = 'MPC_UNLIVE'

def get_liveness_cbf(ego_state, opp_state, is_faster):
    center_intersection = get_ray_intersection_point(ego_state[:2], ego_state[2], opp_state[:2], opp_state[2])
    vec_to_opp = opp_state[:2] - ego_state[:2]
    unit_vec_to_opp = vec_to_opp / np.linalg.norm(vec_to_opp)
    initial_closest_to_opp = ego_state[:2] + unit_vec_to_opp * (config.agent_radius)
    opp_closest_to_initial = opp_state[:2] - unit_vec_to_opp * (config.agent_radius)
    intersection = get_ray_intersection_point(initial_closest_to_opp, ego_state[2], opp_closest_to_initial, opp_state[2])
    if center_intersection is None or intersection is None or ego_state[3] == 0 or opp_state[3] == 0:
        return None

    d0 = np.linalg.norm(initial_closest_to_opp - intersection)
    d1 = np.linalg.norm(opp_closest_to_initial - intersection)

    if is_faster: # Ego agent is faster
        barrier = d1 / opp_state[3] - d0 / ego_state[3]
    else: # Ego agent is slower
        barrier = d0 / ego_state[3] - d1 / opp_state[3]
    return barrier


filename = f'experiment_results/histories/{RUN_AGENT}_{SCENARIO}.json'
logger = DataLogger.load_file(filename)

logger0 = DataLogger.load_file(f"experiment_results/desired_paths/{SCENARIO}_{RUN_AGENT}_0.json")
desired0 = np.array([iteration['states'][0] for iteration in logger0.data['iterations']])
logger1 = DataLogger.load_file(f"experiment_results/desired_paths/{SCENARIO}_{RUN_AGENT}_1.json")
desired1 = np.array([iteration['states'][1] for iteration in logger1.data['iterations']])

traj0 = np.array([iteration['states'][0] for iteration in logger.data['iterations']])
traj1 = np.array([iteration['states'][1] for iteration in logger.data['iterations']])

first_reached_goal = check_when_reached_goal(traj0, logger.data['iterations'][0]['goals'][0][:2])
second_reached_goal = check_when_reached_goal(traj1, logger.data['iterations'][0]['goals'][1][:2])

liveness_cbf_vals = []
for iteration in logger.data['iterations']:
    ego_state = np.array(iteration['states'][0])
    opp_state = np.array(iteration['states'][1])
    liveness_cbf_vals.append(get_liveness_cbf(ego_state, opp_state, True))

liveness_last_idx = len(liveness_cbf_vals) if None not in liveness_cbf_vals[1:] else liveness_cbf_vals[1:].index(None) + 1

traj0_cropped = traj0[:liveness_last_idx]
traj1_cropped = traj1[:liveness_last_idx]
avg_delta_vel0 = calculate_avg_delta_vel(traj0_cropped)
avg_delta_vel1 = calculate_avg_delta_vel(traj1_cropped)
avg_delta_vel = (avg_delta_vel0 + avg_delta_vel1) / 2.0

avg_delta_path0 = calculate_path_deviation(traj0_cropped[:, :2], desired0[:, :2])
avg_delta_path1 = calculate_path_deviation(traj1_cropped[:, :2], desired1[:, :2])
avg_delta_path = (avg_delta_path0 + avg_delta_path1) / 2.0


print(f"Statistics for {RUN_AGENT} on {SCENARIO}")
print("Last liveness idx:", liveness_last_idx)
# print("Avg vel 0:", avg_delta_vel0)
# print("Avg vel 1:", avg_delta_vel1)
print("Avg vel:", avg_delta_vel)
# print("Avg path 0:", avg_delta_path0)
# print("Avg path 1:", avg_delta_path1)
print("Avg path:", avg_delta_path)


