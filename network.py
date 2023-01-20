import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions import Normal
from typing import List, Tuple



class QNetwork(nn.Sequential):
    def __init__(self, 
                 input_dim : int, 
                 n_actions : int, 
                 hidden_dim : List[int],
                 output_dim : int = 1):
        '''
        It builts a fully-connected NN whose hidden layers
        dimension have to be passed as a List[int]
        '''
        
        super(QNetwork, self).__init__()
        
        dim_list = [input_dim + n_actions] + hidden_dim + [output_dim]
        layers = []
        for i in range(len(dim_list)-2):
            layers += [nn.Linear(dim_list[i], dim_list[i+1]), nn.ReLU()]
        layers += [nn.Linear(dim_list[-2], dim_list[-1])]

        super().__init__(*layers)


class PolicyNetwork(nn.Module):
    def __init__(self, 
                 input_dim : int, 
                 n_actions : int, 
                 hidden_dim : List[int], 
                 log_std_min : float, 
                 log_std_max : float,
                 scale : float):
        
        '''
        It builts a fully-connected NN whose hidden layers
        dimension have to be passed as a List[int]
        '''
            
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.scale = scale
        
        dim_list = [input_dim] + hidden_dim
        layers = []
        for i in range(len(dim_list)-1):
            layers += [nn.Linear(dim_list[i], dim_list[i+1]), nn.ReLU()]

        self.layers = nn.ModuleList(layers)
        self.mean_linear = nn.Linear(dim_list[-1], n_actions)
        self.log_std_linear = nn.Linear(dim_list[-1], n_actions)


    def forward(self,
                state : Tensor) -> Tuple[float, float]:

        x = state
        for layer in self.layers:
          x = layer(x)

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std


    def action(self, 
               state : Tensor) -> Tuple[float, float, float]:

        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        gauss_distrib = Normal(mean, std)
        x = gauss_distrib.rsample()
        action = torch.tanh(x)
        
        log_prob = gauss_distrib.log_prob(x) - torch.log(self.scale*(1 - action**2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action*self.scale, log_prob, torch.tanh(mean)*self.scale
        
        
        
        
