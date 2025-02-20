import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils import params

initial_timestep = params['initial_timestep']
end_timestep = initial_timestep + params['total_timestep']

class Traffic_Flow_Data(Dataset):
    def __init__(self, data, window_size):

        if isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
        else:
            self.data = data.float()
        self.window_size = window_size

    def __len__(self):
        return self.data.shape[0] - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.window_size]
        y = self.data[idx + self.window_size]

        return x, y


def data_loader_full():
    occupancy = pd.read_csv('../Data_Clean/Occupancy_all_Long.csv')
    # Convert occupancy to percentage by *100
    occupancy['occu'] *= 100
    flow = pd.read_csv('../Data_Clean/Flow_all_Long.csv')
    speed = pd.read_csv('../Data_Clean/Speed_all_Long.csv')

    occupancy = occupancy[(occupancy['TimeStep'] > initial_timestep) & (occupancy['TimeStep'] < end_timestep)].values
    flow = flow[(flow['TimeStep'] > initial_timestep) & (flow['TimeStep'] < end_timestep)].values
    speed = speed[(speed['TimeStep'] > initial_timestep) & (speed['TimeStep'] < end_timestep)].values

    speed = np.nan_to_num(speed, nan=0)

    return occupancy, flow, speed


def tensor_reshape(data):
    # Get unique timesteps and locations (assuming they are numeric)
    timesteps, t_indices = torch.unique(data[:, 0], sorted=True, return_inverse=True)
    locations, l_indices = torch.unique(data[:, 1], sorted=True, return_inverse=True)

    # Create an empty tensor for the reshaped data
    reshaped = torch.empty((len(timesteps), len(locations)))

    # Iterate over each row in the original data and place the flow value
    for row in data:
        t = row[0].item()  # timestep
        l = row[1].item()  # location
        flow = row[2]
        # Find the indices in the unique arrays
        t_idx = (timesteps == t).nonzero(as_tuple=True)[0].item()
        l_idx = (locations == l).nonzero(as_tuple=True)[0].item()
        reshaped[t_idx, l_idx] = flow

    return reshaped
