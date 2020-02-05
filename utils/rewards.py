# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:57:37 2020

@author: Mikel
"""
import os
import numpy as np

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

def reward_1(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    else:
        return  
    
def reward_2(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    else:
        return  -(theta_norm**2 + 0.02*x**2 + 0.002*v**2)
    
def reward_3(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -500
    elif (abs(theta_norm)<0.5):
        return  20*(np.pi-abs(theta_norm))**2 - w**2 - (5*x)**2
    else:
        return -((theta_norm)**4 + w**2 + (5*x)**2 + v**4)
    
def reward_4(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    else:
        return 2*np.cos(theta_norm) - 0.01*w**2 -  0.001*v
    
def reward_5(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -20
    else:
        return 5*np.cos(theta_norm) + np.cos(np.pi-theta_norm)*0.5*w**2 - v**2 - 0.1*x**2
    
def reward_6(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    else:
        return  np.cos(theta_norm) - 0.01*w**2