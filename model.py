from torch import nn, Tensor, optim
from typing import List, Tuple
import torch
from network import QNetwork, PolicyNetwork


class Critic(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 n_actions: int, 
                 hidden_dim: List[int], 
                 lr : float):
        '''
        It defines the Critic of SAC algorithm.
        It is composed by two soft Q-Networks.
        '''
        
        super(Critic, self).__init__()

        self.q_net1 = QNetwork(input_dim, n_actions, hidden_dim)
        self.q_net2 = QNetwork(input_dim, n_actions, hidden_dim)
        
        self.optim = optim.Adam(self.parameters(), lr)
        self.criterion = nn.MSELoss()


    def forward(self, 
                state : Tensor, 
                action : Tensor) -> Tuple[Tensor, Tensor]:

        x = torch.cat([state, action], 1)
        q1 = self.q_net1(x)
        q2 = self.q_net2(x)

        return q1, q2


    def update(self, 
               state : Tensor, 
               action : Tensor, 
               target_q : float) -> float:
      
        pred_q1, pred_q2 = self.forward(state, action)
        q1_loss = self.criterion(pred_q1, target_q)
        q2_loss = self.criterion(pred_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.optim.zero_grad()
        critic_loss.backward()
        self.optim.step()

        return critic_loss.item()
    
    def soft_update(self, 
                    new : QNetwork, 
                    tau : float) -> None:
        '''
        For updating the parameters of critic target networks following
        the formula : 
        θ_target = (1-τ)*θ_target + τ*θ
        where θ are the parameters of the critic networks while θ_target
        the paramters of the critic target ones
        '''
        for old_p, new_p in zip(self.parameters(), new.parameters()):
            old_p.data.copy_(old_p.data * (1.0 - tau) + new_p.data * tau)
    


class Actor(nn.Module):
    def __init__(self, 
                 input_dim : int, 
                 n_actions : int, 
                 hidden_dim : List[int], 
                 log_std_min : float,
                 log_std_max : float, 
                 scale : float,
                 lr : float):
        
        '''
        It defines the Actor of SAC algorithm.
        It follows a Gaussian policy whose mean and
        std are computed by a NN.
        '''
        
        super(Actor, self).__init__()

        self.policy_net = PolicyNetwork(input_dim, n_actions, hidden_dim, log_std_min, log_std_max, scale)
        self.optim = optim.Adam(self.policy_net.parameters(), lr)


    def forward(self, 
                state : Tensor) -> float:
        mean, log_std = self.policy_net(state)
        return mean, log_std


    def criterion(self, 
                  log_p : Tensor, 
                  pred_q : Tensor, 
                  alpha : float) -> float:

        return (alpha * log_p - pred_q).mean()


    def update(self,
               log_p : Tensor, 
               pred_q : Tensor, 
               alpha : float) -> float:

        policy_loss = self.criterion(log_p, pred_q, alpha)
        self.optim.zero_grad()
        policy_loss.backward()
        self.optim.step()

        return policy_loss.item()

    
    def action(self, 
               state : Tensor) -> Tuple[float, float, float]:
               
        action, log_prob, mean = self.policy_net.action(state)

        return action, log_prob, mean



