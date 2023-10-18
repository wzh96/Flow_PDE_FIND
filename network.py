import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device
from utils import params
from derivatives import get_derivative_2var, get_derivative_3var
from collection_library_utils import build_collection_library

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Rational(nn.Module):
    def __init__(self):
        super(Rational, self).__init__()
        self.device = device
        self.a = nn.parameter.Parameter(
            torch.tensor((1.1915, 1.5957, 0.5, .0218),
                         dtype=torch.float32,
                         device=self.device)
        )
        self.a.requires_grad_(True)

        self.b = nn.parameter.Parameter(
            torch.tensor((2.3830, 0.0, 1.0),
                         dtype=torch.float32,
                         device=self.device)
        )
        self.b.requires_grad_(True)

    def forward(self, X: torch.tensor):
        a = self.a
        b = self.b
        N_X = a[0] + X * (a[1] + X * (a[2] + a[3] * X))
        D_X = b[0] + X * (b[1] + b[2] * X)

        return torch.div(N_X, D_X)


class Network(nn.Module):
    def __init__(self, widths, hidden_act, output_act="None"):
        # hidden_act: str: 'none', 'tanh'
        super(Network, self).__init__()
        self.device = device
        self.widths = widths
        self.num_layers = len(widths) - 1
        self.num_hiddens = self.num_layers - 1

        self.Layers = nn.ModuleList()

        for i in range(self.num_layers):
            self.Layers.append(nn.Linear(in_features=widths[i],out_features=widths[i + 1],bias=True))

        # #initialize weights and bias
        # for i in range(self.num_layers):
        #     nn.init.xavier_uniform_(self.Layers[i].weight)
        #     nn.init.zeros_(self.Layers[i].bias)

        self.Activation_Functions = nn.ModuleList()
        for i in range(self.num_hiddens):
            self.Activation_Functions.append(self._Get_Activation_Function(Encoding=hidden_act))

        self.Activation_Functions.append(self._Get_Activation_Function(Encoding=hidden_act))

    def _Get_Activation_Function(self, Encoding):
        if Encoding == "none":
            return nn.Identity()
        elif Encoding == "tanh":
            return nn.Tanh()
        elif Encoding == "sigmoid":
            return nn.Sigmoid()
        elif Encoding == "elu":
            return nn.ELU()
        elif Encoding == "softmax":
            return nn.Softmax()
        elif Encoding == "relu":
            return nn.ReLU()
        elif Encoding == "rational":
            return Rational()
        else:
            raise ValueError("Unknown Activation Function. Got %s" % Encoding)

    def _Get_Activation_String(self, Activation):
        if isinstance(Activation, nn.Identity):
            return "none"
        elif isinstance(Activation, nn.Tanh):
            return "tanh"
        elif isinstance(Activation, nn.Sigmoid):
            return "sigmoid"
        elif isinstance(Activation, nn.ELU):
            return "elu"
        elif isinstance(Activation, nn.Softmax):
            return "softmax"
        elif isinstance(Activation, nn.ReLU):
            return "relu"
        elif isinstance(Activation, Rational):
            return "rational"
        else:
            raise ValueError("Unknown Activation Function. Got %s" % str(type(Activation)))

    def Get_State(self):
        State = {}
        State['widths'] = self.widths
        Layer_States = []
        for i in range(self.num_layers):
            Layer_States.append(self.Layers[i].state_dict())

        State['Layers'] = Layer_States

        Activation_Types = []
        Activation_States = []
        for i in range(self.num_layers):
            Activation_Types.append(self._Get_Activation_String(self.Activation_Functions[i]))
            Activation_States.append(self.Activation_Functions[i].state_dict())

        State['Activation_Types'] = Activation_Types
        State['Activation_States'] = Activation_States

        return State

    def Load_State(self, State):
        assert (len(self.widths) == len(State['widths']))
        for i in range(len(self.widths)):
            assert (self.widths[i] == State["widths"][i])

        for i in range(self.num_layers):
            assert (self._Get_Activation_String(self.Activation_Functions[i]) == State["Activation_Types"][i])

        for i in range(self.num_layers):
            self.Layers[i].load_state_dict(State['Layers'][i])

        for i in range(self.num_layers):
            self.Activation_Functions[i].load_state_dict(State['Activation_States'][i])

    def forward(self, X):
        for i in range(self.num_layers):
            X = self.Activation_Functions[i](self.Layers[i](X))
        return X


class fullnetwork(nn.Module):
    def __init__(self, params):
        super(fullnetwork, self).__init__()
        self.params = params
        self.OccupancyNet = Network(widths=params['widths_OccupancyNet'], hidden_act=params['hidden_activation'])
        self.FlowNet = Network(widths=params['widths_FlowNet'], hidden_act=params['hidden_activation'])
        self.SpeedNet = Network(widths=params['widths_SpeedNet'], hidden_act=params['hidden_activation'])

        self.library_dim = params['library_dim']
        if params['coeff_init'] == 'xavier':
            self.coeff = nn.Parameter(torch.rand(self.library_dim, 1))
        elif params['coeff_init'] == 'constant':
            self.coeff = nn.Parameter(torch.zeros(self.library_dim, 1))
        elif params['coeff_init'] == 'normal':
            self.coeff = nn.Parameter(torch.randn(self.library_dim, 1))


    def forward(self,X):
        output = {}
        Occupancy_recon = self.OccupancyNet(X)
        X_withk = torch.cat((X, Occupancy_recon), dim=1)
        Flow_recon = self.FlowNet(X_withk)
        Speed_recon = self.SpeedNet(X_withk)

        k_t, k_x = get_derivative_2var(input=X, output=Occupancy_recon)
        q_lambda, q_phi, q_k = get_derivative_3var(input=X_withk,output=Flow_recon)
        v_lambda, v_phi, v_k = get_derivative_3var(input=X_withk,output=Speed_recon)

        q_x = q_k * k_x + q_phi
        v_x = v_k * k_x + v_phi

        z = torch.cat((Occupancy_recon, Flow_recon, Speed_recon, k_x, q_x, v_x), dim=1)

        if self.params['second_order']:
            _, k_xx = get_derivative_2var(input=X, output=k_x)
            _, q_phi_phi, _ = get_derivative_3var(input=X_withk, output=q_phi)
            _, _, q_kk = get_derivative_3var(input=X_withk, output=q_k)
            _, v_phi_phi, _ = get_derivative_3var(input=X_withk, output=v_phi)
            _, _, v_kk = get_derivative_3var(input=X_withk, output=v_k)
            q_xx = q_kk * torch.pow(k_x, 2) + q_k * k_xx + q_phi_phi
            v_xx = v_kk * torch.pow(k_x, 2) + v_k * k_xx + v_phi_phi
            z = torch.cat((z, k_xx, q_xx, v_xx), dim=1)

        Theta = build_collection_library(z, poly_order=self.params['poly_order'], include_sine=False)

        if self.params['sequential_thresholding']:
            coefficient_mask = torch.Tensor(self.params['coefficient_mask']).to(device)
            k_t_predict = torch.matmul(Theta, torch.mul(coefficient_mask, self.coeff))
            output['coefficient_mask'] = coefficient_mask
        else:
            k_t_predict = torch.matmul(Theta, self.coeff)

        output['occupancy_recon'] = Occupancy_recon
        output['flow_recon'] = Flow_recon
        output['speed_recon'] = Speed_recon
        output['k_t'] = k_t
        output['k_x'] = k_x
        output['q_x'] = q_x
        output['v_x'] = v_x
        if self.params['second_order']:
            output['k_xx'] = k_xx
            output['q_xx'] = q_xx
            output['v_xx'] = v_xx
        output['k_t_predict'] = k_t_predict
        output['coeff'] = self.coeff.data

        return output



