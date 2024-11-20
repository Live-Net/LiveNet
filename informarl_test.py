import sys
import os

# Add the InforMARL directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "InforMARL"))
sys.path.append(parent_dir)

import numpy as np
import torch
import pickle
from InforMARL.onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy as Policy

# Device
device = torch.device("cpu")



def create_graph_inputs(agent_positions, obstacle_positions, goal_positions):
    n_rollout_threads = 1
    num_agents = agent_positions.shape[0]
    num_obstacles = obstacle_positions.shape[0]
    num_goals = goal_positions.shape[0]
    total_nodes = num_agents + num_obstacles + num_goals  # Should equal 7

    # 1. Create obs - agent observations (unchanged)
    obs = np.zeros((n_rollout_threads, num_agents, 6))
    for i in range(num_agents):
        obs[0, i] = [
            agent_positions[i, 2],  # vx
            agent_positions[i, 3],  # vy
            agent_positions[i, 0],  # x
            agent_positions[i, 1],  # y
            goal_positions[i, 0],   # target_x
            goal_positions[i, 1]    # target_y
        ]

    # 2. Create node_obs - features for all nodes
    node_obs = np.zeros((n_rollout_threads, num_agents, total_nodes, total_nodes))

    # Create a list of all node positions
    all_positions = np.vstack((agent_positions[:, :2], obstacle_positions, goal_positions))

    # Fill node_obs with relative positions between all nodes
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                # Calculate relative position between nodes i and j
                rel_pos = all_positions[j] - all_positions[i]
                dist = np.linalg.norm(rel_pos)

                # Only connect nodes within max_edge_dist
                if dist <= 1.0:  # from config max_edge_dist: 1
                    # Store relative position features for both agents' perspectives
                    # node_obs[0, :, i, j] = [
                    #     rel_pos[0],  # relative x
                    #     rel_pos[1],  # relative y
                    #     dist,        # distance
                    #     i < num_agents,      # is_agent flag
                    #     num_agents <= i < num_agents + num_obstacles,  # is_obstacle flag
                    #     i >= num_agents + num_obstacles,  # is_goal flag
                    #     1.0          # connection flag
                    # ]
                    node_obs[0, :, i, j] = [dist, dist]

    # 3. Create adjacency matrix based on distances
    adj = np.zeros((n_rollout_threads, num_agents, total_nodes, total_nodes))
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                dist = np.linalg.norm(all_positions[j] - all_positions[i])
                # Connect nodes within max_edge_dist
                if dist <= 1.0:
                    adj[0, :, i, j] = 1

    # 4. Create agent_ids (unchanged)
    agent_id = np.zeros((n_rollout_threads, num_agents, 1))
    for i in range(num_agents):
        agent_id[0, i] = i

    return obs, agent_id, node_obs, adj


# Example usage:
agent_positions = np.array([
    [0, 0, 1, 1],  # agent 1
    [1, 1, 1, 1]   # agent 2
])
obstacle_positions = np.array([
    [2, 2],  # obstacle 1
    [3, 3],  # obstacle 2
    [4, 4],  # obstacle 3
])
goal_positions = np.array([
    [0, 0],  # goal 1
    [1, 1]   # goal 2
])

obs, agent_id, node_obs, adj = create_graph_inputs(agent_positions, obstacle_positions, goal_positions)

with open("InforMARL/saved_args.pkl", "rb") as f:
    loaded_args = pickle.load(f)

with open("InforMARL/observation_space.pkl", "rb") as f:
    observation_space = pickle.load(f)

with open("InforMARL/share_observation_space.pkl", "rb") as f:
    share_observation_space = pickle.load(f)

with open("InforMARL/node_observation_space.pkl", "rb") as f:
    node_observation_space = pickle.load(f)

with open("InforMARL/edge_observation_space.pkl", "rb") as f:
    edge_observation_space = pickle.load(f)

with open("InforMARL/action_space.pkl", "rb") as f:
    action_space = pickle.load(f)

policy = Policy(
    loaded_args,
    observation_space,
    share_observation_space,
    node_observation_space,
    edge_observation_space,
    action_space,
    device=device,
)

rnn_states = np.zeros(
    (
        1,
        2,
        loaded_args.recurrent_N,
        loaded_args.hidden_size,
    ),
    dtype=np.float32,
)
masks = np.ones(
    (1, 2, 1), dtype=np.float32
)

print(f"obs {np.concatenate(obs).shape}")
print(f"node_obs {np.concatenate(node_obs).shape}")
print(f"adj {np.concatenate(adj).shape}")
print(f"agent_id {np.concatenate(agent_id).shape}")
print(f"rnn_states {rnn_states.shape}")
print(f"masks {masks.shape}")

action, rnn_states = policy.act(
    np.concatenate(obs),
    np.concatenate(node_obs),
    np.concatenate(adj),
    np.concatenate(agent_id),
    np.concatenate(rnn_states),
    np.concatenate(masks),
    deterministic=True,
)

print(action.shape)
print(action)
