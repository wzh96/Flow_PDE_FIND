import torch
import torch.nn as nn
import numpy as np
from network import fullnetwork
from loss_functions import data_loss
import time
import pickle

class Traffic_PDE_Learn(nn.Module):
    def __init__(self, params):
        super(Traffic_PDE_Learn, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.params = params
        self.Network = fullnetwork(params).to(self.device)
        self.optimizer = torch.optim.Adam(self.Network.parameters(), lr = self.params['learning_rate'])
        self.LRSchdular = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size= 500, gamma=0.9)
    def Train(self, X, k, q, v):
        self.Network.train()
        print("###### burn-in model training in process ######")
        loss_burnin_all = {}
        loss_main_all = {}
        loss_refine_all = {}
        # active terms = {}
        num_epochs_burnin = self.params['burn_in_epoch']
        for epoch_burnin in range(num_epochs_burnin):
            start_time = time.time()
            self.optimizer.zero_grad()
            score = self.Network(X)
            loss_burnin, _, _, losses = data_loss(self.params, score, k, q, v)
            loss_burnin.backward()
            self.optimizer.step()
            self.LRSchdular.step()

            loss_occu = losses['occupancy']
            loss_flow = losses['flow']
            loss_speed = losses['speed']
            loss_k_t = losses['k_t']
            loss_burnin_all[epoch_burnin] = losses

            duration = time.time() - start_time
            print(f'Epoch [{epoch_burnin + 1}/{num_epochs_burnin}], Loss: {loss_burnin.item()}, loss_occu: {loss_occu},'
                  f'loss_flow: {loss_flow}, loss_speed: {loss_speed}, loss_kt: {loss_k_t}, Duration: {duration}')

        print("###### model training in process ######")
        num_epochs = self.params['num_epochs']
        for epoch in range(num_epochs):
            start_time = time.time()
            self.optimizer.zero_grad()
            score = self.Network(X)
            _, loss_all, _, losses = data_loss(self.params, score, k, q, v)
            loss_all.backward()
            self.optimizer.step()
            self.LRSchdular.step()

            loss_occu = losses['occupancy']
            loss_flow = losses['flow']
            loss_speed = losses['speed']
            loss_k_t = losses['k_t']
            loss_main_all[epoch] = losses

            if self.params['sequential_thresholding'] and (epoch % self.params['threshold_frequency'] == 0) and (epoch > 0):
                self.params['coefficient_mask'] = torch.abs(score['coeff']) > self.params['coefficient_threshold']
                print('THRESHOLDING: %d active coefficients' % torch.sum(self.params['coefficient_mask']))

            duration = time.time() - start_time
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss_all.item()}, loss_occu: {loss_occu},'
                  f'loss_flow: {loss_flow}, loss_speed: {loss_speed}, loss_kt: {loss_k_t}, Duration: {duration}')

        print('###### Model refinement in process ######')
        num_epochs_refine = self.params['refinement_epochs']
        for epoch_refine in range(num_epochs_refine):
            start_time = time.time()
            score = self.Network(X)
            _, _, loss_refine, losses = data_loss(self.params, score, k, q, v)
            self.optimizer.zero_grad()
            loss_refine.backward()
            self.optimizer.step()
            self.LRSchdular.step()

            loss_occu = losses['occupancy']
            loss_flow = losses['flow']
            loss_speed = losses['speed']
            loss_k_t = losses['k_t']
            loss_refine_all[epoch_refine] = losses

            duration = time.time() - start_time
            print(f'Epoch [{epoch_refine + 1}/{num_epochs_refine}], Loss: {loss_refine.item()}, loss_occu: {loss_occu},'
                  f'loss_flow: {loss_flow}, loss_speed: {loss_speed}, loss_kt: {loss_k_t}, Duration: {duration}')

        torch.save({
            'model_state_dict': self.Network.state_dict(),
        }, 'Saved_Model/model_checkpoint.pt')

        file_path = 'Losses/loss_burnin_all.pkl'
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(loss_burnin_all, pickle_file)

        file_path = 'Losses/loss_main_all.pkl'
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(loss_main_all, pickle_file)

        file_path = 'Losses/loss_refine_all.pkl'
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(loss_refine_all, pickle_file)

    def Test(self, X):
        score = self.Network(X)
        score['coefficients'] = torch.mul(self.params['coefficient_mask'], score['coeff'])

        file_path = 'Predicted/results.pkl'
        # Save the dictionary using Pickle
        with open(file_path, 'wb') as pickle_file:
            pickle.dump(score, pickle_file)










