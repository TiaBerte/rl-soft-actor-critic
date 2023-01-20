import torch
import gym
from argparse import ArgumentParser
from typing import Tuple
import os
import random
from torch.utils.tensorboard import SummaryWriter
from sac import SAC
from replay_buffer import ReplayBuffer


def buffer_initialization(env: gym.Env,
                          agent: SAC) -> None:
    '''
    Initializing the buffer memory.
    Agent acts randomly until a minimum number of sample 
    is added to the replay buffer
    '''
    print('='*30)
    print('BUFFER INITIALIZATION')
    print('='*30)
    while len(agent.replay_buffer.memory) < agent.replay_buffer.min_size :
      print('NEW EPISODE')
      state = env.reset()
      done = False
      while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
      print('Samples :', len(agent.replay_buffer.memory))


def train_ep(env : gym.Env, 
             agent : SAC, 
             args : ArgumentParser.parse_args,
             writer : SummaryWriter, 
             train_steps : int) -> Tuple[float, int]:

    steps = 0
    ep_reward = 0
    state = env.reset()
    done = False
    while not done:
        
        action = agent.get_action(state, True)
        next_state, reward, done, _ = env.step(action)

        agent.replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward

        if steps % args.update_steps == 0 :# and min_size_check:
            losses = agent.learning_step()
            train_steps += 1
            for id, loss in losses.items():
                writer.add_scalar(f"loss/{id}", loss, train_steps)

        steps += 1
        if steps > args.max_ep_steps:
            done = True

    return ep_reward, train_steps


def test_ep(env: gym.Env, 
            agent: SAC, 
            args : ArgumentParser.parse_args) -> float:
    state = env.reset()

    ep_reward = 0
    done = False
    steps = 0

    while not done:
        action = agent.get_action(state, False)
        state, reward, done, _ = env.step(action)
        ep_reward += reward

        steps += 1
        if steps > args.max_ep_steps:
            done = True

    return ep_reward


def test(env: gym.Env,
         agent: SAC,
         args : ArgumentParser.parse_args) -> float:

    avg_reward = 0
    for episode in range(args.test_episodes):
        reward = test_ep(env, agent, args) 
        avg_reward += reward

    avg_reward /= args.test_episodes
    
    return avg_reward


def train(env : gym.Env, 
          agent : SAC, 
          args : ArgumentParser.parse_args,
          writer : SummaryWriter):

    env_name = env.unwrapped.spec.id

    no_improv_ep = 0

    if args.model_path:
      episode, max_reward, avg_reward, train_steps = agent.load_episode(env_name, args.model_path)
      episode += 1

    else:
      episode = 1
      max_reward = 0
      avg_reward = 0
      train_steps = 0
      buffer_initialization(env, agent)

    while episode <= args.max_episodes and train_steps < args.max_train_steps:

        reward, train_steps = train_ep(env, agent, args, writer, train_steps)
        avg_reward = 0.2*reward + 0.8*avg_reward

        if episode % args.print_stats == 0:

            print('='*30)
            print(f"Episode {episode}\nReward: {reward:.2f}\nAvg reward: {avg_reward:.2f}\nTrain steps: {train_steps}")
            print('='*30)

        writer.add_scalar("reward/Train_Reward", reward, episode)
        writer.add_scalar("reward/Avg_Train_Reward", avg_reward, episode)

        if episode % args.eval_episode == 0:

            test_reward = test(env, agent, args)
            print('='*30)
            print(f"Test Episode {episode}\nTest Reward: {test_reward:.2f}")
            print('='*30)

            writer.add_scalar("reward/Test_Reward", test_reward, episode)

            if test_reward > max_reward:
                no_improv_ep = 0
                max_reward = test_reward
                print('Updating best model!')
                agent.save_episode(env_name, episode, max_reward, avg_reward, train_steps)
            else:
                no_improv_ep += args.eval_episode
        
        if no_improv_ep > args.no_improv_ep:
            break
            
        episode += 1


def main(args : ArgumentParser.parse_args):
    
    env = gym.make(args.env_name)

    '''
    Setting random seed for reproducibility
    '''
    seed = 44
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    action_dim = env.action_space.shape[0]
    state_dim  = env.observation_space.shape[0]
    scale = env.action_space.high[0]

    replay_buffer = ReplayBuffer(args.min_buffer_size, args.buffer_capacity)

    agent = SAC(state_dim, action_dim, scale, replay_buffer, args)

    writer_path = os.path.join('logs')
    writer = SummaryWriter(writer_path)

    if args.training:
        train(env, agent, args, writer)

    if args.test:
        assert args.model_path, f"You are trying to test the model without loading the weights!"
        agent.load_checkpoint(args.model_path)
        avg_test_reward = test(env, agent, args)
        print(f"Test reward : {avg_test_reward:.2f}")
        
    env.close()
