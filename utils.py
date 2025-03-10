from collection_library_utils import library_size
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

params = {}
params['initial_timestep'] = 120
params['total_timestep'] = 240
params['device'] = device
params['batch_size'] = 512
params['learning_rate'] = 0.001
params['widths_OccupancyNet'] = [2,50,100,100,50,1]
params['widths_FlowNet'] = [3,50,100,100,50,1]
params['widths_SpeedNet'] = [3,50,100,100,50,1]
params['hidden_activation'] = 'rational'

params['occupancy_weight'] = 2
params['flow_weight'] = 0.04
params['speed_weight'] = 0.6
params['k_t_weight'] = 5
params['sparsity_weight'] = 0.1

params['second_order'] = False
params['poly_order'] = 2

params['coeff_init'] = 'constant'

if params['second_order']:
    params['dim'] = 9
else:
    params['dim'] = 6

params['include_sine'] = False
params['library_dim'] =library_size(params['dim'], params['poly_order'], params['include_sine'], True)

params['burn_in_epoch'] = 1000
params['num_epochs'] = 2500
params['refinement_epochs'] = 1500

params['sequential_thresholding'] = True
params['threshold_frequency'] = 400
params['coefficient_threshold'] = 0.0001
params['coefficient_mask'] = torch.ones((params['library_dim'], 1)).to(device)



