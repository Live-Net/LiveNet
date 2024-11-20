import numpy as np
import torch
import pickle
import os
from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy as Policy

# Device 
device = torch.device("cpu")


def create_graph_inputs(agent_positions, obstacle_positions, goal_positions):
    n_rollout_threads = 1
    num_agents = 2
    num_obstacles = len(obstacle_positions)
    num_goals = len(goal_positions)
    total_nodes = num_agents + num_obstacles + num_goals  # Should equal 7
    
    # 1. Create obs - agent observations (unchanged)
    obs = np.zeros((n_rollout_threads, num_agents, 6))
    for i in range(num_agents):
        obs[0, i] = [
            0,                      # vx
            0,                      # vy
            agent_positions[i][0],  # x
            agent_positions[i][1],  # y
            goal_positions[i][0],   # target_x
            goal_positions[i][1]    # target_y
        ]
    
    # 2. Create node_obs - features for all nodes
    # Shape: (1, 2, 7, 7)
    node_obs = np.zeros((n_rollout_threads, num_agents, total_nodes, total_nodes))
    
    # Create a list of all node positions
    all_positions = (
        agent_positions +    # First nodes are agents
        obstacle_positions + # Then obstacles
        goal_positions      # Then goals/landmarks
    )
    
    # Fill node_obs with relative positions between all nodes
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                # Calculate relative position between nodes i and j
                rel_pos = np.array(all_positions[j]) - np.array(all_positions[i])
                dist = np.linalg.norm(rel_pos)
                
                # Only connect nodes within max_edge_dist
                if dist <= 1.0:  # from config max_edge_dist: 1
                    # Store relative position features for both agents' perspectives
                    node_obs[0, :, i, j] = [
                        rel_pos[0],  # relative x
                        rel_pos[1],  # relative y
                        dist,        # distance
                        i < num_agents,      # is_agent flag
                        i >= num_agents and i < num_agents + num_obstacles,  # is_obstacle flag
                        i >= num_agents + num_obstacles,  # is_goal flag
                        1.0          # connection flag
                    ]
    
    # 3. Create adjacency matrix based on distances
    adj = np.zeros((n_rollout_threads, num_agents, total_nodes, total_nodes))
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                pos_i = np.array(all_positions[i])
                pos_j = np.array(all_positions[j])
                dist = np.linalg.norm(pos_j - pos_i)
                # Connect nodes within max_edge_dist
                if dist <= 1.0:
                    adj[0, :, i, j] = 1
    
    # 4. Create agent_ids (unchanged)
    agent_id = np.zeros((n_rollout_threads, num_agents, 1))
    for i in range(num_agents):
        agent_id[0, i] = i
    
    return obs, agent_id, node_obs, adj

# Example usage:
agent_positions = [
    [0, 0],  # agent 1
    [1, 1]   # agent 2
]
obstacle_positions = [
    [2, 2],  # obstacle 1
    [3, 3],  # obstacle 2
    [4, 4]   # obstacle 3
]
goal_positions = [
    [20, 20],  # goal 1
    [-20, -20]   # goal 2
]

obs, agent_id, node_obs, adj = create_graph_inputs(agent_positions, obstacle_positions, goal_positions)

# print(obs.shape)
# print(agent_id.shape)
# print(node_obs.shape)
# print(adj.shape)


# Load the Namespace object
with open("saved_args.pkl", "rb") as f:
    loaded_args = pickle.load(f)

# Load all the spaces from pickle files
with open(os.path.expanduser("~/Documents/Research/code/InforMARL/observation_space.pkl"), "rb") as f:
    observation_space = pickle.load(f)

with open(os.path.expanduser("~/Documents/Research/code/InforMARL/share_observation_space.pkl"), "rb") as f:
    share_observation_space = pickle.load(f)

with open(os.path.expanduser("~/Documents/Research/code/InforMARL/node_observation_space.pkl"), "rb") as f:
    node_observation_space = pickle.load(f)

with open(os.path.expanduser("~/Documents/Research/code/InforMARL/edge_observation_space.pkl"), "rb") as f:
    edge_observation_space = pickle.load(f)

with open(os.path.expanduser("~/Documents/Research/code/InforMARL/action_space.pkl"), "rb") as f:
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