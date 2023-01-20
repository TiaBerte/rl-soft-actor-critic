import pickle
import random
from torch import Tensor
import numpy as np
from typing import Tuple
import os

class ReplayBuffer:
    def __init__(self, 
                 min_size : int = 1e4,
                 capacity : int = 1e6, 
                 device : str = "cpu",
                ):
        '''
        Buffer which contains memory of previous visited state.
        min_size : Define the minimum number of randomly visited states
                   before start the training process.
        capacity : Define the mx number of states kept in memory, once
                   the number of states in memory is equal to capacity,
                   following a FIFO policy, states already in memory are deleted
                   to introduce new ones. 
        '''
        self.min_size = min_size
        self.capacity = capacity
        self.memory = []
        self.device = device


    def __len__(self) -> int:
        return len(self.memory)


    def add(self, 
            state : Tensor, 
            action : Tensor, 
            reward : float, 
            next_state : Tensor, 
            done : bool) -> None:
            '''
            Before adding a new sample, it checks if the 
            buffer is already full, in that case the first 
            element is removed.
            '''
            if self.__len__() >= self.capacity:
                self.memory.pop(0)
            self.memory.append([state, action, reward, next_state, done])


    def sample(self,
               batch_size : int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        '''
        It randomly samples a number of elements equale to batch_size from the buffer.
        '''
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))

        states = Tensor(states).to(self.device)
        actions = Tensor(actions).to(self.device)
        rewards = Tensor(rewards).unsqueeze(1).to(self.device)
        next_states = Tensor(next_states).to(self.device)
        dones = Tensor(np.float32(dones)).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones


    def save(self, 
             env_name : str, 
             train_ep: int,
             variant_mode : str) -> str :
        '''
        In case we want to stop the training and restart from 
        a checkpoint, we need also to save the buffer.
        '''
        buffer_dir = os.path.join("buffer", env_name, variant_mode)
        if not os.path.exists(buffer_dir):
            os.makedirs(buffer_dir)
        
        path = os.path.join(buffer_dir, f"buffer_{env_name}_{variant_mode}_ep_{train_ep}")
        print('Saving buffer ...')
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

        print(f'Buffer saved to {path}')


    def load(self, 
             env_name : str,
             train_ep : str,
             variant_mode : str) -> None:

        path = os.path.join("buffer", f"{env_name}", f"{variant_mode}", f"buffer_{env_name}_{variant_mode}_ep_{train_ep}")
        print(f'Loading buffer memory from {path}')
        with open(path, "rb") as f:
            self.memory = pickle.load(f)
