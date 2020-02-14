# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:02:15 2020

@author: Mikel
"""

class History():
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprob_actions = []
        self.rewards = []
        self.dones = []
        
    def reset(self):
        del self.states[:]
        del self.actions[:]
        del self.logprob_actions[:]
        del self.dones[:]
        del self.rewards[:]
        