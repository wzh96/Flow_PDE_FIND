from collection_library_utils import library_size
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {}

# LSTM params
params['window_size'] = 10
params['input_size'] = 17
params['d_model'] = 512
params['num_heads'] = 8
params['num_encoder_layers'] = 3
params['dim_feedforward'] = 2048
params['dropout'] = 0.1
params['output_size'] = 17
params['learning_rate'] = 0.0001
params['batch_size'] = 32
params['num_epochs'] = 2000
