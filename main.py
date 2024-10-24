import torch
import torch.nn as nn
from Train import Traffic_PDE_Learn
from utils import params
from data_parser import data_loader, data_loader_full

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_occupancy, data_flow, data_speed = data_loader()
    X_occu, y_occu = data_occupancy[:, :2], data_occupancy[:, 2]
    X_occu, y_occu = torch.tensor(X_occu, dtype=torch.float32, requires_grad=True).to(device), \
                     torch.tensor(y_occu, dtype=torch.float32).unsqueeze(1).to(device)

    y_flow = torch.tensor(data_flow[:, 2], dtype=torch.float32).unsqueeze(1).to(device)
    y_speed = torch.tensor(data_speed[:, 2], dtype=torch.float32).unsqueeze(1).to(device)

    model = Traffic_PDE_Learn(params).to(device)

    model.Train(X_occu, y_occu, y_flow, y_speed)

    # load full data
    data_occupancy_all, data_flow_all, data_speed_all = data_loader_full()
    X_occu_all, y_occu_all = data_occupancy_all[:, :2], data_occupancy_all[:, 2]
    X_occu_all, y_occu_all = torch.tensor(X_occu_all, dtype=torch.float32, requires_grad=True).to(device), \
                             torch.tensor(y_occu_all, dtype=torch.float32).unsqueeze(1).to(device)

    model.Test(X_occu_all)



