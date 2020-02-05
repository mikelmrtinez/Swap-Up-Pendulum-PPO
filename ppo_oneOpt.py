# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 00:12:27 2020

@author: Mikel
"""
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple


device = torch.device("cpu") #"cuda:0" if torch.cuda.is_available() else
EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])


def plot_episode_stats(stats, path, smoothing_window=10, noshow=False):
  # Plot the episode length over time
  fig1 = plt.figure(figsize=(10,5))
  plt.plot(stats.episode_lengths)
  plt.xlabel("Episode")
  plt.ylabel("Episode Length")
  plt.title("Episode Length over Time")
  fig1.savefig(path+'_episode_lengths.png')
  if noshow:
      plt.close(fig1)
  else:
      plt.show(fig1)

  # Plot the episode reward over time
  fig2 = plt.figure(figsize=(10,5))
  rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
  plt.plot(rewards_smoothed)
  plt.xlabel("Episode")
  plt.ylabel("Episode Reward (Smoothed)")
  plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
  fig2.savefig(path+'_reward.png')
  if noshow:
      plt.close(fig2)
  else:
      plt.show(fig2)
      


class V(nn.Module):
  def __init__(self, state_dim, hidden_dim=40):
    super(V, self).__init__()
    self.fc1 = nn.Linear(state_dim, 64)
    self.fc2 = nn.Linear(64, 32)
    self.fc3 = nn.Linear(32, 1) 
    self.tanh = nn.Tanh()
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.tanh(x)
    x = self.fc2(x)
    x = self.tanh(x)
    x = self.fc3(x)
    return x


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=40):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc2_mu = nn.Linear(32, action_dim)
        
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        mu = self.fc2_mu(x)
        mu = self.tanh(mu)

        
        return mu 
    

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.policy = Policy(state_dim, action_dim)
        self.value_function = V(state_dim)
        

class History():
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprob_actions = []
        self.rewards = []
        self.dones = []
        
    def clean(self):
        del self.states[:]
        del self.actions[:]
        del self.logprob_actions[:]
        del self.dones[:]
        del self.rewards[:]
        
        
class PPO:
  def __init__(self, state_dim, action_dim, gamma, lr, betas, K, eps, a_std, c1, c2, path_ac=None):
    
    #Hyperparameters for PPO
    self._gamma = gamma
    self._betas = betas
    self._k = K
    self._eps = eps
    self._state_dim = state_dim
    self._actions_dim = action_dim
    self._lr = lr
    self._betas = betas
    self._sigma = a_std
    self. _c1 = c1
    self._c2 = c2
    #Call Networks and Setting them
    self._ac = ActorCritic(state_dim, action_dim).to(device)
    self._ac_old = ActorCritic(state_dim, action_dim).to(device)
    self._ac_old.load_state_dict(self._ac.state_dict())
    
    print(self._ac)

    self._opt = optim.Adam(self._ac.parameters(), lr=self._lr, betas = self._betas)
    self._loss_vf = nn.MSELoss()
    self.episode = 0
      
    #If available pretrained models
    
    if path_ac != None:
        checkpoint_v = torch.load(path_ac)
        self._ac.load_state_dict(checkpoint_v['model_state_dict'])
        self._opt.load_state_dict(checkpoint_v['optimizer_state_dict'])
        self.episode = checkpoint_v['epoch']
        self._loss_vf  = checkpoint_v['loss']
        self._ac_old.load_state_dict(self._ac.state_dict())
        
        
  def get_action(self, state, history):

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
    
    history.states.append(state)
    history.actions.append(action)
    history.logprob_actions.append(log_prob)
    
    
    return action.cpu().data.numpy().flatten()

  def test_policy(self, states, actions):
    #print("Testing actions...")
    
    mu = self._ac.policy(states)
    dist = torch.distributions.Normal(mu, self._sigma**2)
    
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy()
    state_values = self._ac.value_function(states)
    
    return state_values, log_prob, entropy

  def generateMonteCarloDescountedRewards(self, history):
      rewards_discounted = []
      t = 0
      discounted_reward = 0
      for reward, done in zip(reversed(history.rewards), reversed(history.dones)):
          if done:
              discounted_reward = 0
          discounted_reward = discounted_reward*self._gamma + reward
          rewards_discounted.insert(0, discounted_reward)
          t+=1
      #Conver to tensor and normalize 
      rewards_discounted = torch.FloatTensor(rewards_discounted).reshape(-1,1)
      rewards_discounted_normalized = (rewards_discounted - rewards_discounted.mean())/(rewards_discounted.std()+1e-6)
      return rewards_discounted_normalized
  
    
  def listToTensor(self, history):
      
      states_old = torch.squeeze(torch.stack(history.states).to(device)).detach()
      actions_old = torch.squeeze(torch.stack(history.actions).to(device)).detach().reshape(-1,1)
      log_probs_old = torch.squeeze(torch.stack(history.logprob_actions)).to(device).detach().reshape(-1,1)
      return states_old, actions_old, log_probs_old
      
  def update_parameters(self, history):
#      print("Updating networks...")
      
      #MonteCarlo as estimation of rewards
      rewards_discounted_normalized = self.generateMonteCarloDescountedRewards(history)
      #Create batch for trainning
      states_old, actions_old, log_probs_old = self.listToTensor(history)
      for k in range(self._k):
          
          state_values, log_probs, entropies = self.test_policy(states_old, actions_old)
          ratios = torch.exp(log_probs.reshape(-1,1) - log_probs_old.detach())
          
          #reshaping tensors for training
          ratios = torch.squeeze(ratios)
          state_values = torch.squeeze(state_values)
          entropies = torch.squeeze(entropies)
          rewards_discounted_normalized = torch.squeeze(rewards_discounted_normalized)
          As = torch.squeeze(rewards_discounted_normalized - state_values.detach())
          
          #The unclipped Surrogate loss function
          loss_CPI = ratios*As
          #The clipped Surrogate loss fucntion
          loss_CLIP = torch.clamp(ratios, 1.-self._eps, 1.+self._eps)*As
          #The merged loss function
          self.loss_CLIP_VF_S = ( -torch.min(loss_CPI, loss_CLIP) 
                            + self._c1*self._loss_vf(state_values, rewards_discounted_normalized) 
                            - self._c2*entropies)
          self._opt.zero_grad()
          self.loss_CLIP_VF_S.mean().backward()
          self._opt.step()
      
      self._ac_old.load_state_dict(self._ac.state_dict())
          

  def train(self, episodes, time_steps, update_steptime, env, path):
      
      history = History()
      avg_length = 0
      accum_reward = 0
      stats = EpisodeStats(episode_lengths=np.zeros(episodes), episode_rewards=np.zeros(episodes)) 
      print("START!")
      steps_count = 0
      for i_episode in range(1, episodes+1):
          #print("Episode :", i_episode)
          state = env.reset()
          for t in range(time_steps):
            steps_count+=1
            #Run Policy_old
            action = self.get_action(state, history) 
            next_state, reward, done, _ = env.step(action)
            state = next_state
            #Store the enviroment's reaction to action taken by Policy_old
            history.rewards.append(reward)
            history.dones.append(done)
            
            #If TRUE we update the parameters
            if steps_count % update_steptime == 0 :
                self.update_parameters(history)
                history.clean()
                steps_count = 0
            accum_reward += reward   
            if done:
                break
          avg_length += t 
          
          #Printing stats of the taining proces    
          if i_episode%50 == 0:
              print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode+self.episode, 
                    int(avg_length/50), int(accum_reward/50)))
              stats.episode_rewards[i_episode-1] += int(accum_reward/50)
              stats.episode_lengths[i_episode-1] = int(avg_length/50)
              avg_length = 0.
              accum_reward = 0.
          #Saving the models  
          if i_episode%500 == 0:
             torch.save({
                'epoch': i_episode+self.episode,
                'model_state_dict': self._ac.state_dict(),
                'optimizer_state_dict': self._opt.state_dict(),
                'loss': self.loss_CLIP_VF_S}, path+str(i_episode+self.episode)+"_ac.pkl")
            
             print("created and saved "+path+str(i_episode+self.episode)+" models")
    
      return stats

