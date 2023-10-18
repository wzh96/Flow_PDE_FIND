from network import fullnetwork
import torch
from data_parser import data_loader, data_loader_full
from utils import params
from loss_functions import data_loss
from network import Network
import torch.nn as nn
import numpy as np
import pandas as pd

import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Network = fullnetwork(params).to(device)

data_occupancy, data_flow, data_speed = data_loader()

X_occu, y_occu = data_occupancy[:, :2], data_occupancy[:, 2]
X_occu, y_occu = torch.tensor(X_occu, dtype=torch.float32,requires_grad=True).to(device), \
                 torch.tensor(y_occu, dtype=torch.float32).unsqueeze(1).to(device)

y_flow = torch.tensor(data_flow[:,2], dtype=torch.float32).unsqueeze(1).to(device)
y_speed = torch.tensor(data_speed[:,2], dtype=torch.float32).unsqueeze(1).to(device)

optimizer = torch.optim.Adam(Network.parameters(), lr = params['learning_rate'])

num_epochs = 3000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = Network(X_occu)
    loss_all, losses = data_loss(params, output, y_occu, y_flow, y_speed)
    loss_all.backward()
    optimizer.step()

    loss_occu = losses['occupancy']
    loss_flow = losses['flow']
    loss_speed = losses['speed']
    loss_k_t = losses['k_t']

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_all.item()}, loss_occu: {loss_occu},'
          f'loss_flow: {loss_flow}, loss_speed: {loss_speed}, loss_kt: {loss_k_t}')

# load full data
data_occupancy_all, data_flow_all, data_speed_all = data_loader_full()
X_occu_all, y_occu_all = data_occupancy_all[:, :2], data_occupancy_all[:, 2]
X_occu_all, y_occu_all = torch.tensor(X_occu_all, dtype=torch.float32, requires_grad=True).to(device), \
                 torch.tensor(y_occu_all, dtype=torch.float32).unsqueeze(1).to(device)

pred = Network(X_occu_all)

file_path = 'Predicted/results.pkl'
# Save the dictionary using Pickle
with open(file_path, 'wb') as pickle_file:
    pickle.dump(pred, pickle_file)

