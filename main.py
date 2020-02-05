# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 00:25:54 2020

@author: Mikel
"""
from collections import namedtuple
from env.continuous_cartpole import ContinuousCartPoleEnv
from ppo_oneOpt import PPO, plot_episode_stats
from utils.rewards import reward_6

reward = reward_6

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])

if __name__ == "__main__":
    
    env = ContinuousCartPoleEnv(reward)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    episodes = 10000
    time_steps = 500
    lr = 2.5e-4
    betas = (0.9, 0.999)
    k_epochs = 64
    gamma = 0.99
    eps = 0.2
    update_time = 4096
    a_std = 0.5
    c1 = 0.7
    c2 = 0.05
    
    path_save_models = "./models/ppo_oneopt_example"
    path_save_plots = "./results/ppo_oneopt_"
    
    #If continue training plug here the models
    path_ac = None#"../models/ppo_shared_w_3000_V.pkl"
    
    model = PPO(state_dim, action_dim, gamma, lr, betas, k_epochs, eps, a_std, c1, c2, path_ac=path_ac)
    stats = model.train(episodes, time_steps, update_time, env, path_save_models)
    
    plot_episode_stats(stats, path_save_plots)
