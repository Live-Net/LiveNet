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



class informarl_controller:
    def __init__(self, model_definition_filepath, static_obs, goal):
        with open("InforMARL/saved_args.pkl", "rb") as f:
            self.loaded_args = pickle.load(f)

        with open("InforMARL/observation_space.pkl", "rb") as f:
            self.observation_space = pickle.load(f)

        with open("InforMARL/share_observation_space.pkl", "rb") as f:
            self.share_observation_space = pickle.load(f)

        with open("InforMARL/node_observation_space.pkl", "rb") as f:
            self.node_observation_space = pickle.load(f)

        with open("InforMARL/edge_observation_space.pkl", "rb") as f:
            self.edge_observation_space = pickle.load(f)

        with open("InforMARL/action_space.pkl", "rb") as f:
            self.action_space = pickle.load(f)

        self.policy = Policy(
            self.loaded_args,
            self.observation_space,
            self.share_observation_space,
            self.node_observation_space,
            self.edge_observation_space,
            self.action_space,
            device=device,
        )

        self.rnn_states = np.zeros(
            (
                2,
                self.loaded_args.recurrent_N,
                self.loaded_args.hidden_size,
            ),
            dtype=np.float32,
        )
        self.masks = np.ones(
            (1, 2, 1), dtype=np.float32
        )

        static_obs_informarl = []
        for obs in static_obs:
            static_obs_informarl.append([obs[0], obs[1]])
        
        self.static_obs_informarl = np.array(static_obs_informarl)
        self.goal = np.expand_dims(goal[:2], axis=0)



    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):        
        # self.initial_state = np.expand_dims(np.concatenate((initial_state[2:], initial_state[:2], self.goal[0])), axis=0)
        # self.opp_state = np.expand_dims(np.concatenate((opp_state[2:], opp_state[:2], np.zeros(2))), axis=0)
        self.initial_state = np.expand_dims(initial_state, axis=0)  # Shape: (1, 4)
        self.opp_state = np.expand_dims(opp_state, axis=0)  # Shape: (1, 4)



    def make_step(self, timestamp, initial_state):
        # Expand dimensions of initial_state to match expected shape
        self.initial_state = np.expand_dims(initial_state, axis=0)  # Shape: (1, 4)

        # Combine agent positions
        agent_positions = np.concatenate((self.initial_state, self.opp_state), axis=0)
        opp_goal = np.expand_dims(np.zeros(2), axis=0)
        goal_positions = np.concatenate((self.goal, opp_goal), axis=0)

        # Compute distances between initial_state[0, :2] and all static obstacles
        initial_xy = self.initial_state[0, :2]  # Extract x, y positions
        distances = np.linalg.norm(self.static_obs_informarl - initial_xy, axis=1)  # Euclidean distances

        # Get indices of the top 3 closest obstacles
        closest_indices = np.argsort(distances)[:3]

        # Select the top 3 closest obstacles
        closest_obstacles = self.static_obs_informarl[closest_indices]

        print(agent_positions.shape)
        print(closest_obstacles.shape)
        print(goal_positions.shape)


        obs, agent_id, node_obs, adj = create_graph_inputs(agent_positions, closest_obstacles, goal_positions)

        # print(f"obs {np.concatenate(obs).shape}")
        # print(f"node_obs {np.concatenate(node_obs).shape}")
        # print(f"adj {np.concatenate(adj).shape}")
        # print(f"agent_id {np.concatenate(agent_id).shape}")
        # print(f"rnn_states {self.rnn_states.shape}")
        # print(f"masks {self.masks.shape}")

        action, self.rnn_states = self.policy.act(
            np.concatenate(obs),
            np.concatenate(node_obs),
            np.concatenate(adj),
            np.concatenate(agent_id),
            self.rnn_states,
            np.concatenate(self.masks),
            deterministic=True,
        )
        
        # actions: [None, ←, →, ↓, ↑, comm1, comm2]
        
        

        res_action = np.array([0, 0])

        cur_action = action[0, 0]
        if cur_action == 0:
            res_action = np.array([0, 0])
        elif cur_action == 1:
            res_action = np.array([-0.1, 0])
        elif cur_action == 2:
            res_action = np.array([0.1, 0])
        elif cur_action == 3:
            res_action = np.array([0, -0.1])
        elif cur_action == 4:
            res_action = np.array([0, 0.1])

        print(res_action)
        
        return np.reshape(res_action, (2, 1))
        

