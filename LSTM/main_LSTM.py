import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from LSTM import LSTM
from Train_LSTM import LSTM_Train
from utils_LSTM import params
from traffic_data_loader import Traffic_Flow_Data, data_loader_full, tensor_reshape
import pickle
import io

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read reconstructed traffic flow data
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            else:
                return super().find_class(module, name)


    file_path = '../Model_Final/Predicted/results.pkl'
    with open(file_path, 'rb') as pickle_file:
        pred = CPU_Unpickler(pickle_file).load()
    data = pred['flow_recon'].to(device)

    data_occupancy_all, data_flow_all, data_speed_all = data_loader_full()
    X_occu_all, _ = data_occupancy_all[:, :2], data_occupancy_all[:, 2]
    X_occu_all = torch.tensor(X_occu_all, dtype=torch.float32).to(device)

    data = torch.cat((X_occu_all, data), dim=1).detach()

    data = tensor_reshape(data)

    # Select the first 70% of the data for training
    data_train = data[:int(data.size(0) * 0.7),:]

    dataset = Traffic_Flow_Data(data_train, window_size = params['window_size'])
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle = True)
    
    input_size = params['input_size']
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    output_size = params['output_size']

    model_LSTM = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

    loss_history = LSTM_Train(model = model_LSTM, dataloader=dataloader,
                              num_epochs = params['num_epochs'], learning_rate = params['learning_rate'],
                              device = device)

    # save the trained model for furture validation on traffic flow prediction
    save_path = 'saved_model/model_LSTM.pth'
    torch.save(model_LSTM.state_dict(), save_path)
    print(f"Model saved to {save_path}")


