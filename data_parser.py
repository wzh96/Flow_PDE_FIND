import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def data_loader():
    occupancy = pd.read_csv('Data_Clean/Occupancy_Long.csv')
    # Convert occupancy to percentage by *100
    occupancy['Occupancy'] *= 100
    flow = pd.read_csv('Data_Clean/Flow_Long.csv')
    speed = pd.read_csv('Data_Clean/Speed_Long.csv')

    occupancy = occupancy[occupancy['TimeStep']<289].values
    flow = flow[flow['TimeStep']<289].values
    speed = speed[speed['TimeStep']<289].values

    return occupancy, flow, speed



def data_loader_full():
    occupancy = pd.read_csv('Data_Clean/Occupancy_all_Long.csv')
    # Convert occupancy to percentage by *100
    occupancy['Occupancy'] *= 100
    flow = pd.read_csv('Data_Clean/Flow_all_Long.csv')
    speed = pd.read_csv('Data_Clean/Speed_all_Long.csv')

    occupancy = occupancy[occupancy['TimeStep'] < 289].values
    flow = flow[flow['TimeStep'] < 289].values
    speed = speed[speed['TimeStep'] < 289].values

    return occupancy, flow, speed


    # Standardized inputs: "TimeStep" and "Station_Number"
    # scaler = StandardScaler()
    # columns_to_standardize = ['TimeStep', 'Station_Number']
    # occupancy = pd.read_csv('Data_Clean/Occupancy_Long.csv')
    # occupancy[columns_to_standardize] = scaler.fit_transform(occupancy[columns_to_standardize])
    #
    # flow = pd.read_csv('Data_Clean/Flow_Long.csv')
    # flow[columns_to_standardize] = scaler.fit_transform(flow[columns_to_standardize])
    #
    # speed = pd.read_csv('Data_Clean/Speed_Long.csv')
    # speed[columns_to_standardize] = scaler.fit_transform(speed[columns_to_standardize])