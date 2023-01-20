import torch
from torch import optim, Tensor
import os
from argparse import ArgumentParser
from typing import Tuple
from model import Critic, Actor
from replay_buffer import ReplayBuffer
import numpy as np

class SAC:
    def __init__(self, 
                 input_dim : int, 
                 n_actions : int,
                 scale : float, 
                 replay_buffer : ReplayBuffer, 
                 args : ArgumentParser.parse_args):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = args.gamma # Discount factor
        self.tau = args.tau # Target smoothing coefficient
        self.batch_size = args.batch_size
        
        self.alpha = torch.Tensor([args.alpha]).to(self.device) # Temperatur paramter
        self.alpha_tuning = args.alpha_tuning

        self.K = args.K # Number of value to average for Averaged-SAC
        if self.K :
          self.value_list = torch.zeros([self.batch_size, self.K]).to(self.device)

        assert not (self.K and self.alpha_tuning), 'You are trying to use two variants of SAC at the same time!'
        
        if self.alpha_tuning:
          self.variant_mode = 'sac_alpha_tuning'
        elif self.K :
          self.variant_mode = f'sac_avg_K_{self.K}'
        else:
          self.variant_mode = 'sac'

        # Replay buffer
        replay_buffer.device = self.device
        self.replay_buffer = replay_buffer
        
        # Actor
        self.actor = Actor(input_dim, n_actions, args.hidden_dim_p, 
                           args.log_std_min, args.log_std_max, scale, args.lr_p).to(self.device)

        # Critic and critic target
        self.critic = Critic(input_dim, n_actions, args.hidden_dim_q, args.lr_c).to(self.device)
        self.critic_t = Critic(input_dim, n_actions, args.hidden_dim_q, args.lr_c).to(self.device)
        for target_param, param in zip(self.critic_t.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Optimizers
        self.policy_optimizer = self.actor.optim
        self.critic_optimizer = self.critic.optim

        if self.alpha_tuning:
            self.target_entropy = -Tensor([n_actions]).to(self.device).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], args.lr_a)


    def learning_step(self) -> dict:
        '''
        Core function of the SAC algorithm. It takes care of
        the learning process.
        '''
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        with torch.no_grad():
            next_actions, next_logs_pi, _ = self.actor.action(next_states)
            target_q1, target_q2 = self.critic_t(next_states, next_actions)
            min_q_t = torch.min(target_q1, target_q2)
            if self.K:
                new_value = self.alpha * next_logs_pi
                self.value_list = torch.cat([self.value_list[:, 1:], new_value.view(-1, 1)], dim=1)
                value = self.value_list.mean(axis=1).view(-1, 1)
            else:
                value = self.alpha * next_logs_pi

            target_q = rewards + (1 - dones) * self.gamma * (min_q_t - value)

        # Updating Q1 and Q2 critic networks
        critic_loss = self.critic.update(states, actions, target_q)

        pred_actions, log_prob, _ = self.actor.action(states)
        q1, q2 = self.critic(states, pred_actions)
        min_q = torch.min(q1, q2)

        # Updating policy network
        policy_loss = self.actor.update(log_prob, min_q, self.alpha)

        # Updating target critic networks
        self.critic_t.soft_update(self.critic, self.tau)

        loss_dict = {'critic_loss' : critic_loss,
                     'policy_loss' : policy_loss}

        if self.alpha_tuning:
            # Automatic temperature parameter tuning
            entropy_loss = self.temperature_tuning(log_prob)
            loss_dict['entropy_loss'] = entropy_loss
            loss_dict['alpha_value'] = self.alpha

        return loss_dict


    def temperature_tuning(self, 
                           log_prob : Tensor) -> float:
        '''
        Function for optimization of 
        the temperature parameter alpha.
        '''
        entropy_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        entropy_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()
        return entropy_loss.item()


    @torch.no_grad()
    def get_action(self, 
                   state : Tensor, 
                   train : bool) -> np.array:
        '''
        The action returned at training time and test time are different.
        At training time, it returns a sample from a gaussian distribution 
        whose mean and std are computed by the NN.
        At test time it return the rescaled mean predicted by the NN.
        '''
        state = torch.Tensor(state).float().to(self.device).unsqueeze(0)
        if train:
            action, _, _ = self.actor.action(state)
        else:
            _, _, action = self.actor.action(state)
        return action.detach().cpu().numpy()[0]


    def save_model(self, 
                   env_name : str, 
                   episode : int, 
                   max_reward : float, 
                   avg_reward : float, 
                   train_steps : int) -> None:
        
        checkpoints_dir = os.path.join("checkpoints", env_name, self.variant_mode)
        
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
        path = os.path.join(checkpoints_dir, f"{env_name}_{self.variant_mode}_ep_{episode}_test_rew_{max_reward:.1f}")
        print(f'Saving models ...')

        state_dict = {'policy_state_dict': self.actor.state_dict(),
                      'critic_state_dict': self.critic.state_dict(),
                      'critic_target_state_dict': self.critic_t.state_dict(),
                      'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                      'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                      'episode' : episode,
                      'max_reward' : max_reward,
                      'avg_reward' : avg_reward,
                      'train_steps' : train_steps}

        if self.alpha_tuning : 
            state_dict['alpha'] = self.alpha
            state_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()

        if self.K :
            state_dict['value_list'] = self.value_list

        torch.save(state_dict, path) 
        print(f'Model saved to {path}')
       
        
    def load_checkpoint(self, 
                        path : str) -> Tuple[int, float, float, int]:
        
        print(f'Loading model ...')
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_t.load_state_dict(checkpoint['critic_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if self.alpha_tuning:
            self.alpha = torch.Tensor([checkpoint['alpha']]).to(self.device)
            self.log_alpha = self.alpha.log().to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

        if self.K :
            self.value_list = checkpoint['value_list']

        episode = checkpoint['episode']
        max_reward = checkpoint['max_reward']
        avg_reward = checkpoint['avg_reward']
        train_steps = checkpoint['train_steps']
        
        print(f'Model loaded from {path}')
        return episode, max_reward, avg_reward, train_steps
    
    
    def load_episode(self, 
                     env_name : str, 
                     path : str) -> Tuple[int, float, float, int]:
        '''
        Function used for resuming the training. It is sufficient to
        pass the path to the model checkpoint. It takes care of loading
        all the required information and the replay buffer memory which 
        should be stored in the folder "./buffer/{variant}/".
        '''
        episode, max_reward, avg_reward, train_steps = self.load_checkpoint(path)
        self.replay_buffer.load(env_name, episode, self.variant_mode)
        
        return episode, max_reward, avg_reward, train_steps

    
    def save_episode(self, 
                     env_name : str, 
                     episode : int, 
                     max_reward : float, 
                     avg_reward : float, 
                     train_steps : int):
        '''
        It saves the model information for resuming the training and the buffer memory
        '''
        self.save_model(env_name, episode, max_reward, avg_reward, train_steps)
        self.replay_buffer.save(env_name, episode, self.variant_mode)


