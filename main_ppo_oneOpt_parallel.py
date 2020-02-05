# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 00:25:54 2020

@author: Mikel
"""
from collections import namedtuple
from env.continuous_cartpole import ContinuousCartPoleEnv
from ppo_oneOpt import PPO, plot_episode_stats
from rewards import reward_6
from multiprocessing import Process
import timeit

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
    
    path_save_models = "./models/ppo_oneopt_parallel_"
    path_save_plots = "./results/ppo_oneopt_"
    
    #If continue training plug here the models
    path_ac = None
    
    model1 = PPO(state_dim, action_dim, gamma, lr, betas, k_epochs, eps, a_std, c1, c2, path_ac=path_ac)
    model2 = PPO(state_dim, action_dim, gamma, lr, betas, k_epochs, eps, a_std, c1, c2, path_ac=path_ac)
    model3 = PPO(state_dim, action_dim, gamma, lr, betas, k_epochs, eps, a_std, c1, c2, path_ac=path_ac)
    model4 = PPO(state_dim, action_dim, gamma, lr, betas, k_epochs, eps, a_std, c1, c2, path_ac=path_ac)
    model5 = PPO(state_dim, action_dim, gamma, lr, betas, k_epochs, eps, a_std, c1, c2, path_ac=path_ac)

    start = timeit.timeit()
    p1 = Process(target=model1.train(episodes, time_steps, update_time, env, path_save_models+"1_"))
    p1.start()
    p2 = Process(target=model2.train(episodes, time_steps, update_time, env, path_save_models+"2_"))
    p2.start()
    p3 = Process(target=model3.train(episodes, time_steps, update_time, env, path_save_models+"3_"))
    p3.start()
    p4 = Process(target=model4.train(episodes, time_steps, update_time, env, path_save_models+"4_"))
    p4.start()
    p5 = Process(target=model5.train(episodes, time_steps, update_time, env, path_save_models+"5_"))
    p5.start()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    end = timeit.timeit()
    elapse_1 = end - start
    print(elapse_1)
    #stats = model.train(episodes, time_steps, update_time, env, path_save_models)
    
    plot_episode_stats(stats, path_save_plots)