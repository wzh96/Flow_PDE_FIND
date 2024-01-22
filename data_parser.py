import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import params

initial_timestep = params['initial_timestep']
end_timestep = initial_timestep + params['total_timestep']

def data_loader():
    occupancy = pd.read_csv('Data_Clean/Occupancy_Long.csv')
    # Convert occupancy to percentage by *100
    occupancy['Occupancy'] *= 100
    flow = pd.read_csv('Data_Clean/Flow_Long.csv')
    speed = pd.read_csv('Data_Clean/Speed_Long.csv')

    occupancy = occupancy[(occupancy['TimeStep'] > initial_timestep) & (occupancy['TimeStep'] < end_timestep)].values
    flow = flow[(flow['TimeStep'] > initial_timestep) & (flow['TimeStep'] < end_timestep)].values
    speed = speed[(speed['TimeStep'] > initial_timestep) & (speed['TimeStep'] < end_timestep)].values

    return occupancy, flow, speed

def data_loader_full():
    occupancy = pd.read_csv('Data_Clean/Occupancy_all_Long.csv')
    # Convert occupancy to percentage by *100
    occupancy['Occupancy'] *= 100
    flow = pd.read_csv('Data_Clean/Flow_all_Long.csv')
    speed = pd.read_csv('Data_Clean/Speed_all_Long.csv')

    occupancy = occupancy[(occupancy['TimeStep'] > initial_timestep) & (occupancy['TimeStep'] < end_timestep)].values
    flow = flow[(flow['TimeStep'] > initial_timestep) & (flow['TimeStep'] < end_timestep)].values
    speed = speed[(speed['TimeStep'] > initial_timestep) & (speed['TimeStep'] < end_timestep)].values

    return occupancy, flow, speed
