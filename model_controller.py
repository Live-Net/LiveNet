import torch
import config
import numpy as np
from model_utils import ModelDefinition
from models import BarrierNet
from util import perturb_model_input, axay_to_aw_control


class ModelController:
    def __init__(self, model_definition_filepath, goal, static_obs):
        self.model_definition = ModelDefinition.from_json(model_definition_filepath)
        self.goal = goal
        self.model = BarrierNet(self.model_definition).to(config.device)
        self.static_obs = static_obs
        print(self.model_definition.weights_path)
        self.model.load_state_dict(torch.load(self.model_definition.weights_path))
        self.model.eval()

    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):
        self.initial_state = initial_state
        self.opp_state = opp_state
    
    def make_step(self, timestamp, initial_state):
        self.use_for_training = False
        self.initial_state = initial_state
        model_input_original = np.append(self.initial_state, self.opp_state)
        model_input_original = perturb_model_input(
            model_input_original,
            self.static_obs,
            self.model_definition.n_opponents,
            self.model_definition.x_is_d_goal,
            self.model_definition.add_liveness_as_input,
            self.model_definition.fixed_liveness_input,
            self.model_definition.static_obs_xy_only,
            self.model_definition.ego_frame_inputs,
            self.model_definition.add_new_liveness_as_input,
            self.model_definition.add_dist_to_static_obs,
            self.goal
        )

        model_input = np.nan_to_num((model_input_original - self.model_definition.input_mean) / self.model_definition.input_std)

        with torch.no_grad():
            model_input = torch.autograd.Variable(torch.from_numpy(model_input), requires_grad=False)
            model_input = torch.reshape(model_input, (1, self.model_definition.get_num_inputs())).to(config.device)
            model_output = self.model(model_input, 0)
            if self.model_definition.is_barriernet:
                model_output = np.array([model_output[0], model_output[1]])
            else:
                model_output = model_output.reshape(-1).cpu().detach().numpy()

        output = model_output * self.model_definition.label_std + self.model_definition.label_mean
        # print("Outputted controls:", output)
        output = output.reshape(-1, 1)

        return output
