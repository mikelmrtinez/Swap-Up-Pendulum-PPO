# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:00:19 2020

@author: Mikel
"""
import torch.nn as nn

class V(nn.Module):
  def __init__(self, state_dim, hidden_dim=64):
    super(V, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
    self.fc3 = nn.Linear(hidden_dim//2, 1) 
    
    self.tanh = nn.Tanh()
    
  def forward(self, x):
    x = self.fc1(x)
    x = self.tanh(x)
    x = self.fc2(x)
    x = self.tanh(x)
    x = self.fc3(x)
    return x


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2_mu = nn.Linear(hidden_dim//2, action_dim)
        
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
        