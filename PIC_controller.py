# import sys
# import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# pic_dir = os.path.join(current_dir, 'PIC')
# sys.path.insert(0, pic_dir)
# sys.path.insert(0, os.path.join(pic_dir, 'maddpg'))
# sys.path.insert(0, os.path.join(pic_dir, 'models'))

import torch
import numpy as np

class pic_controller:
    def __init__(self, model_definition_filepath, static_obs, goal):
        self.use_for_training = False
        
        # Load Model

        #'PIC/maddpg/ckpt_plot/simple_goal_n2_gcn_max_soft_hiddensize128_9/agents_best.ckpt'
        
        checkpoint = torch.load('PIC/maddpg/ckpt_plot/simple_goal_all_reward/agents_best.ckpt', map_location='cpu')  # Adjust 'cpu' if using GPU
        self.saved_agent = checkpoint['agents']

        # Load Static Obstacles
        static_obs_pic = []
        for obs in static_obs:
            static_obs_pic.append([obs[0], obs[1]])
        self.static_obs = np.array(static_obs_pic)

        # Load Goal
        self.goal = goal[:2]
        self.u_force = np.zeros(2)

    
    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):
        # Shape: (4, ) -> [x, y, vx, vy]
        self.initial_state = initial_state
        self.opp_state = opp_state

    def make_step(self, timestamp, initial_state):
        # Observation [1, 40]
        '''
        Observation includes:
        - Agent's own velocity (2D)
        - Agent's own position (2D)
        - Relative position to its goal (2D)
        - Relative positions of other agents (2D * 2)
        - Relative positions of nearby obstacles (2D * 16)
        '''
        self.initial_state = initial_state

        own_position = self.initial_state[0:2]  # [x, y]
        own_velocity = self.initial_state[2:4]  # [vx, vy]
                
        relative_goal_position = self.goal - own_position  # [goal_x - x, goal_y - y]
        relative_opp_position = self.opp_state[0:2] - own_position  # [opp_x - x, opp_y - y]
        relative_static_obs = self.static_obs - own_position  # Subtract own position from all obstacles
        num_obstacles = len(relative_static_obs)
        if num_obstacles < 16:
            padding = np.zeros((16 - num_obstacles, 2))
            relative_static_obs = np.vstack((relative_static_obs, padding))
        else:
            distances = np.linalg.norm(relative_static_obs, axis=1)
            closest_indices = np.argsort(distances)[:16]
            relative_static_obs = relative_static_obs[closest_indices]
        
        relative_static_obs_flattened = relative_static_obs.flatten()

        observation = np.concatenate([
            own_velocity,              # [vx, vy]
            own_position,              # [x, y]
            relative_goal_position,    # [goal_x - x, goal_y - y]
            relative_opp_position,     # [opp_x - x, opp_y - y]
            relative_static_obs_flattened  # [obs_1_x - x, obs_1_y - y, ..., obs_16_x - x, obs_16_y - y]
        ])
        observation = observation.reshape(1, -1)

        obs_input = torch.from_numpy(observation).float()
        action_output = self.saved_agent.select_action(obs_input)
        print(action_output)

        
        self.u_force[0] = action_output[0][1] - action_output[0][2]
        self.u_force[1] = action_output[0][3] - action_output[0][4]
        sensitivity = 5
        self.u_force *= sensitivity

        mass = 1
        u_acceleration = self.u_force / mass
        u_acceleration = np.reshape(u_acceleration, (2, 1))
        
        return u_acceleration
