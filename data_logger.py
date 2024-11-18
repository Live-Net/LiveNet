import os
import json
import torch
import config
import numpy as np
from util import get_x_is_d_goal_input

class Dataset(torch.utils.data.Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    # Denotes the total number of samples
    def __len__(self):
        return len(self.features)

    # Generates one sample of data
    def __getitem__(self, index):
        X = self.features[index]
        y = self.labels[index]

        return X, y


class DataLogger:
    def __init__(self, filename):
        self.filename = filename
        self.data = {
            'iterations': [],
            'obstacles': []
        }
    
    def set_obstacles(self, obstacles):
        self.data['obstacles'] = obstacles


    def log_iteration(self, states, goals, controls):
        states = [state.reshape(-1).tolist() for state in states]
        goals = [goal.reshape(-1).tolist() for goal in goals]
        controls = [control.reshape(-1).tolist() for control in controls]
        self.data['iterations'].append({
            'states': states,
            'goals': goals,
            'controls': controls
        })
        json.dump(self.data, open(self.filename, 'w'))


    @staticmethod
    def load_file(filename):
        logger = DataLogger(filename)
        logger.data = json.load(open(filename))
        return logger


class BlankLogger:
    def __init__(self):
        pass
    
    def set_obstacles(self, obstacles):
        pass

    def log_iteration(self, states, goals, controls):
        pass


# Extracts inputs and outputs from data files.
class DataGenerator:
    def __init__(self, filenames, x_is_d_goal):
        self.x_is_d_goal = x_is_d_goal
        self.data_streams = []
        for filename in sorted(filenames):
            if os.path.isdir(filename):
                folder = filename
                for subfile in os.listdir(folder):
                    if not subfile.endswith('.json'):
                        continue
                    print(os.path.join(folder, subfile))
                    self.data_streams.append(json.load(open(os.path.join(folder, subfile))))
            else:
                print(filename)
                self.data_streams.append(json.load(open(filename)))
        print("Number of file streams:", len(self.data_streams))


    def get_inputs(self, agent_idx, normalize):
        data = []
        for data_stream in self.data_streams:
            for iteration in data_stream['iterations']:
                # 4 + 4 = 8 inputs.
                inputs = iteration['states'][agent_idx] + iteration['states'][1 - agent_idx]
                if self.x_is_d_goal:
                    inputs = get_x_is_d_goal_input(inputs, iteration['goals'][agent_idx])

                data.append(np.array(inputs))
        data = np.array(data)

        if not normalize:
            return data

        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std


    def get_outputs(self, agent_idx, normalize):
        data = []
        for data_stream in self.data_streams:
            for iteration in data_stream['iterations']:
                data.append(iteration['controls'][agent_idx])
        data = np.array(data)

        if not normalize:
            return data

        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_normalized = (data - data_mean) / data_std
        return data_normalized, data_mean, data_std


    # TODO: Fix this when obstacles become part of the input (dynamic).
    def get_obstacles(self):
        return self.data_streams[0]['obstacles'].copy()

