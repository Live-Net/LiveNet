import numpy as np
from macbf_torch.core import *
from macbf_torch.config_macbf import *
import torch


class macbf_torch_controller:
    def __init__(self, model_definition_filepath, static_obs, goal):
        self.cbf_net = CBFNetwork().to(device)
        self.action_net = ActionNetwork().to(device)
        
        self.cbf_net.load_state_dict(torch.load('macbf_torch/checkpoints_barrier_eval/cbf_net_step_70000.pth', weights_only=True, map_location=device))
        self.action_net.load_state_dict(torch.load('macbf_torch/checkpoints_barrier_eval/action_net_step_70000.pth', weights_only=True, map_location=device))
        self.cbf_net.eval()
        self.action_net.eval()

        static_obs_macbf = []
        for obs in static_obs:
            static_obs_macbf.append([obs[0], obs[1], 0, 0])
        static_obs_np = np.array(static_obs_macbf)    # Shape: (Num obs, 4)
        self.static_obs = torch.tensor(static_obs_np, dtype=torch.float32, device=device)

        goal_np = goal[:2]
        self.goal = torch.tensor(goal_np, dtype=torch.float32, device=device)
        self.goal = self.goal.unsqueeze(0)
        
        

    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):
        self.initial_state = np.expand_dims(initial_state, axis=0)  # Shape: (1, 4)
        self.opp_state = np.expand_dims(opp_state, axis=0)  # Shape: (1, 4)
    
    def make_step(self, initial_state):
        self.initial_state = np.expand_dims(initial_state, axis=0)  # Shape: (1, 4)
        with torch.no_grad():
            s_np = np.concatenate((self.initial_state, self.opp_state))
            s = torch.tensor(s_np, dtype=torch.float32, device=device)
            neighbor_features_cbf, indices = compute_neighbor_features(s, config_macbf.DIST_MIN_THRES, config_macbf.TOP_K, wall_agents=self.static_obs, include_d_norm=True)
            neighbor_features_action, _ = compute_neighbor_features(s, config_macbf.DIST_MIN_THRES, config_macbf.TOP_K, wall_agents=self.static_obs, include_d_norm=False, indices=indices)
            
            s = s[:1]
            neighbor_features_cbf = neighbor_features_cbf[:1]
            neighbor_features_action = neighbor_features_action[:1]
            
            h = self.cbf_net(neighbor_features_cbf)
            a = self.action_net(s, self.goal, neighbor_features_action)
        a_res = torch.zeros_like(a, requires_grad=True)
        optimizer_res = torch.optim.SGD([a_res], lr=config_macbf.REFINE_LEARNING_RATE)

        for _ in range(config_macbf.REFINE_LOOPS):
            optimizer_res.zero_grad()   
            dsdt = dynamics_macbf(s, a + a_res)
            s_next = s + dsdt * config_macbf.TIME_STEP
            neighbor_features_cbf_next, _ = compute_neighbor_features(s_next, config_macbf.DIST_MIN_THRES, config_macbf.TOP_K,  wall_agents=self.static_obs, include_d_norm=True, indices=None)
            h_next = self.cbf_net(neighbor_features_cbf_next)
            deriv = h_next - h + config_macbf.TIME_STEP * config_macbf.ALPHA_CBF * h
            deriv_flat = deriv.view(-1)
            error = torch.sum(torch.relu(-deriv_flat))
            error.backward()
            optimizer_res.step()
        
        with torch.no_grad():
            a_opt = a + a_res.detach()
        a_opt = torch.reshape(a_opt, (2, 1))
        return a_opt.cpu().numpy()
        # res = np.array([1, 1])
        # return np.expand_dims(res, axis=1)
    

        

