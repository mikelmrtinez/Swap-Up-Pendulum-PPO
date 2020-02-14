# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:24:16 2020

@author: Mikel
"""
from collections import namedtuple
import os
from env.continuous_cartpole import ContinuousCartPoleEnv
from model.PPO import PPO
from utils.rewards import *

reward = reward_15


EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

if __name__ == "__main__":
    
    #Hyperparameters settings for PPO
    episodes = 5000
    time_steps = 500
    lr = 2.5e-4
    epochs = 64
    gamma = 0.99
    eps = 0.4
    update_time = 4096
    a_std = 0.7
    c1 = 0.7
    c2 = 0.01
    
    #Name of out training model
    path = "ppo_model"
    #Path of our petrained model
    path_model = None#"./results/saved_models/ppo_new_5_6000.pkl"
    
    #Initialize enviroment and set state and action dimentions
    env = ContinuousCartPoleEnv(reward)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    model = PPO(state_dim, action_dim, gamma, lr, epochs, eps, a_std, c1, c2, path_model=path_model)
  
    model.train(episodes, time_steps, update_time, env, path)

    
        
    