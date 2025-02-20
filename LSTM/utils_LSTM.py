from collection_library_utils import library_size
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {}

# LSTM params
params['window_size'] = 10
params['input_size'] = 17
params['hidden_size'] = 100
params['num_layers'] = 3
params['output_size'] = 17
params['learning_rate'] = 0.0001
params['batch_size'] = 32
params['num_epochs'] = 2000
