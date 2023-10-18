import torch

def data_loss(params, output, occupancy, flow, speed):
    '''
    Define loss function as loss_all
    '''
    losses = {}
    losses['occupancy'] = torch.mean((output['occupancy_recon']-occupancy)**2)
    losses['flow'] = torch.mean((output['flow_recon']-flow)**2)
    losses['speed'] = torch.mean((output['speed_recon']-speed)**2)
    losses['k_t'] = torch.mean((output['k_t']-output['k_t_predict'])**2)

    coefficients = torch.Tensor(params['coefficient_mask']).to(params['device']) * output['coeff']
    losses['sparsity'] = torch.sum(torch.abs(coefficients))


    loss_burnin = params['occupancy_weight']*losses['occupancy']\
                  +params['flow_weight']*losses['flow']\
                  +params['speed_weight']*losses['speed']\
                  +params['k_t_weight']*losses['k_t']

    loss_all = params['occupancy_weight']*losses['occupancy']\
               +params['flow_weight']*losses['flow']\
               +params['speed_weight']*losses['speed']\
               +params['k_t_weight']*losses['k_t']\
               +params['sparsity_weight']*losses['sparsity']

    loss_refinement = params['occupancy_weight']*losses['occupancy']\
                      +params['flow_weight']*losses['flow']\
                      +params['speed_weight']*losses['speed']\
                      +params['k_t_weight']*losses['k_t']

    return loss_burnin, loss_all, loss_refinement, losses


