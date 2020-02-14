# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:09:41 2020

@author: Mikel
"""
import numpy as np
import matplotlib.pyplot as plt


number_runs = 8
path = "ppo_new_"
steps_avg = 50
max_episode = 7000

    
stats = []
for indx in range(number_runs):
    
    lengths = np.load("./../results/stats/"+path+str(indx+1)+"_"+str(max_episode)+"_episodes.npy")
    lengths = lengths[:max_episode-1]
    #print(lengths)
    step = steps_avg
    accum_length = 0.
    avg_length = []
    for i in range(len(lengths)):
        accum_length += lengths[i]
        if i%step == 0:
            avg_length.append(accum_length/step)
            accum_length = 0.
    stats.append(avg_length)
    
    
stats = np.asarray(stats)
mean_length = np.mean(stats, 0)
std_length = np.std(stats, 0)
print(mean_length.shape)
print(std_length.shape)
x = np.linspace(0, len(lengths), len(mean_length))
convergence = np.ones_like(x)*500

plt.fill_between(x, mean_length-std_length, mean_length+std_length,label="variance", alpha=0.2)
plt.plot(x, mean_length, label="mean")
plt.plot(x,convergence, '--', label="max length" )
plt.grid()
plt.legend()
plt.title('length episodes over Time')
plt.xlabel('episodes')
plt.ylabel("Length episode")
plt.savefig("./../results/plots/"+path+"episodes")
plt.show()


stats = []
for indx in range(number_runs):
    rewards = np.load("./../results/stats/"+path+str(indx+1)+"_"+str(max_episode)+"_reward.npy")
    rewards = rewards[:max_episode-1]
    step = steps_avg
    accum_reward = 0.
    avg_rewards = []
    for i in range(len(rewards)):
        accum_reward += rewards[i]
        if i%step == 0:
            avg_rewards.append(accum_reward/step)
            accum_reward = 0.

    stats.append(avg_rewards)
    
stats = np.asarray(stats)
mean_reward = np.mean(stats, 0)
std_rewards = np.std(stats, 0)
x = np.linspace(0, len(rewards), len(mean_reward))
convergence = np.ones_like(x)*2000

plt.fill_between(x, mean_reward-std_rewards, mean_reward+std_rewards,label="variance", alpha=0.2)
plt.plot(x, mean_reward, label="mean")
plt.plot(x,convergence, '--', label="convergency threshold" )
plt.grid()
plt.legend()
plt.title('Reward over Time')
plt.xlabel('episodes')
plt.ylabel("Reward")
plt.savefig("./../results/plots/"+path+"rewards")
plt.show()
