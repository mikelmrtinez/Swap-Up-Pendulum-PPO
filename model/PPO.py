# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 00:12:27 2020

@author: Mikel
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

import numpy as np
from collections import namedtuple
from model.ActorCritic import ActorCritic
from model.history import History


device = torch.device("cpu")#if torch.cuda.is_available() else "cpu")
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
      
        
class PPO:
  def __init__(self, state_dim, action_dim, gamma, lr, epochs, eps, a_std, c1, c2, path_model=None, timesteps_before_save=500):
    
    #Hyperparameters for PPO
    self._gamma = gamma
    self._epochs = epochs
    self._eps = eps
    self._state_dim = state_dim
    self._actions_dim = action_dim
    self._lr = lr
    self._sigma = a_std
    self. _c1 = c1
    self._c2 = c2
    #Call Networks and Setting them
    self._ac = ActorCritic(state_dim, action_dim).to(device)
    self._ac_old = ActorCritic(state_dim, action_dim).to(device)
    self._ac_old.load_state_dict(self._ac.state_dict())
    self._timesteps_before_save = timesteps_before_save

    self._opt = optim.Adam(self._ac.parameters(), lr=self._lr)
    self._loss_vf = nn.MSELoss()
    self.episode = 0
    self._print_timesteps = 50
    self._save_timestep = 1000
      
    #If available pretrained models
    
    if path_model != None:
        checkpoint_v = torch.load(path_model)
        self._ac.load_state_dict(checkpoint_v['model_state_dict'])
        self._opt.load_state_dict(checkpoint_v['optimizer_state_dict'])
        self.episode = checkpoint_v['episodes']
        self._loss_vf  = checkpoint_v['loss']
        self._ac_old.load_state_dict(self._ac.state_dict())
        
  def get_actor_critic(self):
    return self._ac

  
  def get_action(self, state):

    state = torch.FloatTensor(state.reshape(1, -1)).to(device)
    mu = self._ac_old.policy(state)
    
    dist = torch.distributions.Normal(mu, self._sigma**2)
    actions = dist.sample(mu.size())
    
    #We clip here the actions ! ! !
    action = torch.clamp(actions, -1., 1.) 
    log_prob = dist.log_prob(action)
    

    state = torch.squeeze(state)
    log_prob = torch.squeeze(log_prob)
    action = torch.squeeze(action)
    
    
    return action, log_prob, state

  def get_actions_training_policy(self, states, actions):
    
    mu = self._ac.policy(states)
    dist = torch.distributions.Normal(mu, self._sigma**2)
    
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy()
    state_values = self._ac.value_function(states)
    
    return state_values, log_prob, entropy

  def generateMonteCarloDescountedRewards(self, history):
      rewards_discounted = []
      discounted_reward = 0
      for indx in range(len(history.rewards)-1, -1, -1):
          if history.dones[indx]:
              discounted_reward = 0
          discounted_reward = history.rewards[indx] + discounted_reward*self._gamma
          rewards_discounted.insert(0, discounted_reward)
      return torch.FloatTensor(rewards_discounted).reshape(-1,1)
  
    
  def listToTensor(self, mylist):
      mytensor = torch.squeeze(torch.stack(mylist)).detach()
      return mytensor
      
  def update_actorcritic(self, history):      
      #Montecarlo estimation of rewards
      rewards_discounted= self.generateMonteCarloDescountedRewards(history)
      #Noramlization for stabillity and regularization
      rewards_discounted_normalized = F.batch_norm(rewards_discounted, 
                                                   rewards_discounted.mean(), 
                                                   rewards_discounted.std()+1e-6)
      #Create batch for trainning
      states_old = self.listToTensor(history.states)
      actions_old= self.listToTensor(history.actions).reshape(-1,1)
      log_probs_old = self.listToTensor(history.logprob_actions).reshape(-1,1)

      for epoch in range(self._epochs):
          
          state_values, log_probs, entropies = self.get_actions_training_policy(states_old, actions_old)
          ratios = torch.exp(log_probs.reshape(-1,1) - log_probs_old)
          #reshaping tensors for training
          ratios = torch.squeeze(ratios)
          state_values = torch.squeeze(state_values)
          entropies = torch.squeeze(entropies).to(device)
          rewards_discounted_normalized = torch.squeeze(rewards_discounted_normalized)
          As = torch.squeeze(rewards_discounted_normalized - state_values)
          
          #The unclipped Surrogate loss function
          loss_CPI = ratios*As
          #The clipped Surrogate loss fucntion
          loss_CLIP = torch.clamp(ratios, 1.-self._eps, 1.+self._eps)*As
          #The merged loss function
          self.loss_CLIP_VF_S = (-torch.min(loss_CPI, loss_CLIP) 
                            + self._c1*self._loss_vf(state_values, rewards_discounted_normalized.detach()) 
                            - self._c2*entropies)
          self._opt.zero_grad()
          self.loss_CLIP_VF_S.mean().backward()
          self._opt.step()
      
      self._ac_old.load_state_dict(self._ac.state_dict())
          

  def train(self, episodes, time_steps, update_steptime, env, path):
      path_save = './results/saved_models/'+path
   
      stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes)) 
      print("START!")     
      collector = History()
      avg_length = 0
      accum_reward = 0
      state = env.reset()
      max_length_episode = time_steps
      episode_reward = 0
      length_episode = 0
      i_episode = 1
      
      while i_episode < episodes:
          for _ in range(update_steptime):
            
            action , log_prob, state = self.get_action(state)
            next_state, reward, done, _ = env.step(action.cpu().data.numpy().flatten())
            
            #Store the enviroment's reaction to action taken by Policy_old
            collector.rewards.append(reward)
            collector.dones.append(done)            
            collector.states.append(state)
            collector.actions.append(action)
            collector.logprob_actions.append(log_prob)
            
            state = next_state
            episode_reward += reward
            length_episode+=1

            if done or length_episode == max_length_episode :
                state = env.reset()
                done = True
                
                avg_length += length_episode
                accum_reward += episode_reward
                
                stats.episode_lengths[i_episode-1] = length_episode
                stats.episode_rewards[i_episode-1] = episode_reward
                length_episode = 0
                episode_reward = 0
                i_episode += 1
                
              
                if i_episode%self._print_timesteps == 0:
                    print('Episode {}/{} \tAvg length: {} \tAvg reward: {}'.format(i_episode+self.episode, episodes,
                          int(avg_length/50), int(accum_reward/50)))
                    avg_length = 0.
                    accum_reward = 0.
                    
                if i_episode%self._save_timestep == 0:
                  torch.save({
                    'episodes': i_episode+self.episode,
                    'model_state_dict': self._ac.state_dict(),
                    'optimizer_state_dict': self._opt.state_dict(),
                    'loss': self.loss_CLIP_VF_S}, path_save+str(i_episode+self.episode)+".pkl")
                
                  print("created and saved "+path+str(i_episode+self.episode)+" models")
                if i_episode%self._save_timestep == 0:
                  np.save('./results/stats/'+path+str(i_episode+self.episode)+'_episodes',stats.episode_lengths)
                  np.save('./results/stats/'+path+str(i_episode+self.episode)+'_reward',stats.episode_rewards)

          self.update_actorcritic(collector)
          collector.reset()
         
