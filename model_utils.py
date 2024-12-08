import os
import json
from cvxopt import solvers, matrix
from dataclasses import dataclass, asdict
from typing import Optional


def solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    
    return sol['x']


@dataclass
class ModelDefinition:
    is_barriernet: bool
    weights_path: Optional[str]
    nHidden1: int
    nHidden21: int
    nHidden22: Optional[int]
    nHidden24: Optional[int]
    input_mean: float
    input_std: float
    label_mean: float
    label_std: float
    add_control_limits: bool # Default = false
    add_liveness_filter: bool # Default = false
    separate_penalty_for_opp: bool # Default = false
    x_is_d_goal: bool # Default = false
    add_liveness_as_input: bool
    n_opponents: int
    fixed_liveness_input: bool
    static_obs_xy_only: bool
    ego_frame_inputs: bool
    add_new_liveness_as_input: bool
    sep_pen_for_each_obs: bool
    add_dist_to_static_obs: bool
    nHidden23: Optional[int] = None

    def get_num_inputs(self):
        if self.static_obs_xy_only:
            return 4 + 4 + (self.n_opponents - 1) * (2 + self.add_dist_to_static_obs) + self.add_liveness_as_input + self.add_new_liveness_as_input
        return 4 + self.n_opponents * 4 + self.add_liveness_as_input + self.add_new_liveness_as_input


    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f)
    
    @staticmethod
    def from_json(path: str):
        with open(path, 'r') as f:
            data = json.load(f)
            path_dir = os.path.dirname(path)
            weights_path = os.path.join(path_dir, data['weights_path'])
            data['weights_path'] = weights_path
            if 'add_control_limits' not in data:
                data['add_control_limits'] = False
            if 'add_liveness_filter' not in data:
                data['add_liveness_filter'] = False
            if 'separate_penalty_for_opp' not in data:
                data['separate_penalty_for_opp'] = False
            if 'x_is_d_goal' not in data:
                data['x_is_d_goal'] = False
            if 'add_liveness_as_input' not in data:
                data['add_liveness_as_input'] = False
            if 'n_opponents' not in data:
                data['n_opponents'] = 1
            if 'fixed_liveness_input' not in data:
                data['fixed_liveness_input'] = False
            if 'static_obs_xy_only' not in data:
                data['static_obs_xy_only'] = False
            if 'ego_frame_inputs' not in data:
                data['ego_frame_inputs'] = False
            if 'add_new_liveness_as_input' not in data:
                data['add_new_liveness_as_input'] = False
            if 'sep_pen_for_each_obs' not in data:
                data['sep_pen_for_each_obs'] = False
            if 'add_dist_to_static_obs' not in data:
                data['add_dist_to_static_obs'] = False
            return ModelDefinition(**data)
