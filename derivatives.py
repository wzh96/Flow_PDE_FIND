import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def get_derivative_2var(input, output):
    external_grad = torch.ones_like(output)
    dev = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=external_grad, retain_graph=True, create_graph=True)[0]
    k_t, k_x = dev[:,0].unsqueeze(1), dev[:,1].unsqueeze(1)
    return k_t, k_x

def get_derivative_3var(input, output):
    external_grad = torch.ones_like(output)
    dev = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=external_grad, retain_graph=True, create_graph=True)[0]
    qv_t, qv_x, qv_k = dev[:,0].unsqueeze(1), dev[:,1].unsqueeze(1), dev[:,2].unsqueeze(1)
    return qv_t, qv_x, qv_k

# def get_derivative_2var_2ord(input, output):
#     external_grad = torch.ones_like(output)
#     dev_1 = torch.autograd.grad(outputs=output, inputs=input, grad_outputs=external_grad, retain_graph=True, create_graph=True)[0]
#     dev_2 = torch.autograd.grad(outputs=dev_1, inputs=input, retain_graph=True)[0]
#     return




