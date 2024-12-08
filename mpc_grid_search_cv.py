"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import numpy as np
from mpc_cbf import MPC
from scenarios import DoorwayScenario
from data_logger import BlankLogger
from environment import Environment
from simulation import run_simulation
from metrics import get_desired_path, gather_all_metric_data
import multiprocessing as mp

def gen_metrics_single(scenario_vals, desired_path_0, desired_path_1, params, queue):
    # Setup scenario config
    start_x, start_y, goal_x, goal_y = scenario_vals
    opp_gamma, obs_gamma, live_gamma, liveness_thresh = params
    print("Calculating metrics for params:", params)

    # Run scenario
    logger = BlankLogger()
    scenario = DoorwayScenario(initial_x=start_x, initial_y=start_y, goal_x=goal_x, goal_y=goal_y)

    goals = scenario.goals.copy()

    env = Environment(scenario.initial.copy(), scenario.goals.copy())
    controllers = [
        MPC(agent_idx=0, opp_gamma=opp_gamma, obs_gamma=obs_gamma, live_gamma=live_gamma, liveness_thresh=liveness_thresh, goal=goals[0,:], static_obs=scenario.obstacles.copy()),
        MPC(agent_idx=1, opp_gamma=opp_gamma, obs_gamma=obs_gamma, live_gamma=live_gamma, liveness_thresh=liveness_thresh, goal=goals[1,:], static_obs=scenario.obstacles.copy())
    ]

    x_cum, u_cum = run_simulation(scenario, env, controllers, logger, None)
    traj0 = x_cum[0]
    traj1 = x_cum[1]

    # Caclulate metrics
    metric_data = gather_all_metric_data(scenario, traj0, traj1, goals, desired_path_0, desired_path_1)
    metric_params = [opp_gamma, obs_gamma, live_gamma, liveness_thresh]
    metric = metric_params + metric_data

    queue.put((scenario_vals, metric))
    return metric


def gen_metrics(scenario, gridsearch):
    print("Metric generation for scenario:", scenario)
    start_x, start_y, goal_x, goal_y = scenario
    csv_filepath = f"{folder_to_save_to}/scenario_{start_x}_{start_y}_{goal_x}_{goal_y}.csv"

    scenario = DoorwayScenario(initial_x=start_x, initial_y=start_y, goal_x=goal_x, goal_y=goal_y)
    desired_path_0 = get_desired_path(scenario.initial[0], scenario.goals[0])
    desired_path_1 = get_desired_path(scenario.initial[1], scenario.goals[1])
    metric_history = []

    for opp_gamma, obs_gamma, live_gamma, liveness_thresh in gridsearch:
        metric_history.append(gen_metrics_single(scenario, desired_path_0, desired_path_1, opp_gamma, obs_gamma, live_gamma, liveness_thresh))
        print(f"\tSaving metrics to {csv_filepath}")
        np.savetxt(csv_filepath, np.array(metric_history), fmt='%0.4f', delimiter=', ', header='opp_gamma, obs_gamma, live_gamma, liveness_thresh, goal_reach_idx0, goal_reach_idx1, traj_collision, obs_collision_0, obs_collision_1, delta_vel_0, delta_vel_1, path_dev_0, path_dev_1')


def file_writer(q):
    '''listens for messages on the q, writes to file. '''
    scenario_metrics = {}
    metrics_written = 0
    folder_to_save_to = 'scenario_generation3'
    while 1:
        m = q.get()
        if m == 'kill':
            break

        scenario, metric = m

        csv_filepath = f"{folder_to_save_to}/scenario_{scenario[0]}_{scenario[1]}_{scenario[2]}_{scenario[3]}.csv"
        history = scenario_metrics.get(scenario, [])
        history.append(metric)
        scenario_metrics[scenario] = history

        metrics_written += 1
        print(f"Metric written for scenario {scenario} to file {csv_filepath}. Metric count: {metrics_written}")
        try:
            np.savetxt(csv_filepath, np.array(history), fmt='%0.4f', delimiter=', ', header='opp_gamma, obs_gamma, live_gamma, liveness_thresh, goal_reach_idx0, goal_reach_idx1, traj_collision, obs_collision_0, obs_collision_1, delta_vel_0, delta_vel_1, path_dev_0, path_dev_1')
        except Exception as e:
            print("Exception when writing to file:", e)


if __name__ == "__main__":
    scenarios = [
        # (-1.0, 0.5, 2.0, 0.15),
        # (-1.0, 0.5, 2.0, 0.30),
        # (-1.0, 0.5, 2.0, 0.45), ??
        # (-1.0, 0.4, 2.0, 0.15),
        # (-1.0, 0.4, 2.0, 0.35),
        # (-1.0, 0.3, 2.0, 0.15),
        # (-1.0, 0.3, 2.0, 0.25),

        # (-0.5, 0.5, 2.0, 0.15),
        # (-0.5, 0.5, 2.0, 0.30),
        # (-0.5, 0.5, 2.0, 0.45),
        # (-0.5, 0.4, 2.0, 0.15),
        # (-0.5, 0.4, 2.0, 0.35),
        # (-0.5, 0.3, 2.0, 0.15),
        # (-0.5, 0.3, 2.0, 0.25),

        (0.0, 0.5, 2.0, 0.15),
        (0.0, 0.5, 2.0, 0.30),
        (0.0, 0.5, 2.0, 0.45),
        (0.0, 0.4, 2.0, 0.15),
        (0.0, 0.4, 2.0, 0.35),
        (0.0, 0.3, 2.0, 0.15),
        (0.0, 0.3, 2.0, 0.25),

        (-1.0, 0.5, 3.0, 0.15),
        (-1.0, 0.5, 3.0, 0.30),
        (-1.0, 0.5, 3.0, 0.45),
        (-1.0, 0.4, 3.0, 0.15),
        (-1.0, 0.4, 3.0, 0.35),
        (-1.0, 0.3, 3.0, 0.15),
        (-1.0, 0.3, 3.0, 0.25),

        (-0.5, 0.5, 3.0, 0.15),
        (-0.5, 0.5, 3.0, 0.30),
        (-0.5, 0.5, 3.0, 0.45),
        (-0.5, 0.4, 3.0, 0.15),
        (-0.5, 0.4, 3.0, 0.35),
        (-0.5, 0.3, 3.0, 0.15),
        (-0.5, 0.3, 3.0, 0.25),

        # (0.0, 0.5, 3.0, 0.15),
        # (0.0, 0.5, 3.0, 0.30),
        # (0.0, 0.5, 3.0, 0.45),
        # (0.0, 0.4, 3.0, 0.15),
        # (0.0, 0.4, 3.0, 0.35),
        # (0.0, 0.3, 3.0, 0.15),
        # (0.0, 0.3, 3.0, 0.25),
    ]

    opp_gammas = np.arange(0.4, 0.9, 0.1) # 5
    obs_gammas = np.arange(0.2, 0.6, 0.1) # 4
    live_gammas = np.arange(0.2, 0.3, 0.1) # 1
        # live_gammas = np.arange(0.2, 0.7, 0.1) # 5
    liveliness = np.arange(0.8, 1.21, 0.1) # 5
    # 18 * 25 = 450
    # 450 * 7 = 3150
    # 25 * 12 = 300 * 40 = 12000

    gridsearch = np.array(np.meshgrid(opp_gammas, obs_gammas, live_gammas, liveliness)).T.reshape(-1, 4)

    # opp_gammas = np.arange(0.1, 0.5, 0.1)
    # opp_gammas = np.array([config.opp_gamma])
    # obs_gammas = np.array([config.obs_gamma])
    # live_gammas = np.array([config.liveliness_gamma])
    # liveliness = np.array([config.liveness_threshold])

    # Single processing
    # for scenario in scenarios:
    #     gen_metrics(scenario, gridsearch)

    # Multi processing

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(processes=22)

    watcher = pool.apply_async(file_writer, (q,))

    todos = []
    for scenario in scenarios:
        scenario_obj = DoorwayScenario(initial_x=scenario[0], initial_y=scenario[1], goal_x=scenario[2], goal_y=scenario[3])
        desired_path_0 = get_desired_path(scenario_obj.initial[0], scenario_obj.goals[0])
        desired_path_1 = get_desired_path(scenario_obj.initial[1], scenario_obj.goals[1])
        for grid in gridsearch:
            todos.append((scenario, desired_path_0, desired_path_1, grid, q))
    
    # print(len(todos))
    # print(1/0)

    pool.starmap(gen_metrics_single, todos)
    print("Done!")
    q.put('kill')
    pool.close()
    pool.join()
