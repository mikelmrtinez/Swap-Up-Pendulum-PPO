# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:34:39 2020

@author: Mikel
"""
import torch
from env.continuous_cartpole import ContinuousCartPoleEnv
from ppo_oneOpt import ActorCritic
from rewards import reward_6


reward = reward_6


def get_action(ac, state, a_std):

    state = torch.FloatTensor(state.reshape(1, -1))
    mu = ac.policy(state)

    dist = torch.distributions.Normal(mu, a_std**2)
    actions = dist.sample(mu.size())
    
    #We clip here the actions ! ! !
    action = torch.clamp(actions, -1., 1.) 
    log_prob = dist.log_prob(action)
    
    state = torch.squeeze(state)
    log_prob = torch.squeeze(log_prob)
    action = torch.squeeze(action)
    
    return action.cpu().data.numpy().flatten()


if __name__ == '__main__':

    episodes = 10
    time_steps = 300
    a_std = 0.5
    path_with_model = "./models/ppo_oneopt_3_a_12500.pkl"
    env = ContinuousCartPoleEnv(reward)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    ac = ActorCritic(state_dim, action_dim)
    checkpoint_pi = torch.load(path_with_model)
    ac.load_state_dict(checkpoint_pi['model_state_dict'])

    for eps_i in range(episodes):
        s = env.reset()
        for t in range(time_steps):
            env.render()
            a = get_action(ac, s, a_std)
            s, _, d, _ = env.step(a)
            if d:
                break
    
    env.close()
