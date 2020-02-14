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
    
def reward_8(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.01*w**2 - 0.0001*v**2 + 1
    else:
        return  np.cos(theta_norm) - 0.001*w**2 -0.001*v**2 
    
def reward_11(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.01*w**2 + 1
    else:
        return  np.cos(theta_norm) - 0.001*w**2 -0.001*x**2 -0.001*v**2
    
def reward_12(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.1*w**2 + 2
    else:
        return  np.cos(theta_norm) + 0.01*w**2 - 0.01*x**2 
    
def reward_13(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.1*w**2 + 2
    else:
        return  np.cos(theta_norm) - 0.1*w**2*np.cos(theta_norm) - 0.01*x**2


    


def reward_15(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/8):
        return  2*np.cos(theta_norm) - w**2 + 4
    
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.1*w**2 + 2
    else:
        return  np.cos(theta_norm) - 0.1*w**2*np.cos(theta_norm) - 0.01*x**2
    
def reward_14(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/10):
        return  2*np.cos(theta_norm) - 0.1*w**2 + 4
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.1*w**2 + 2
    else:
        return np.cos(theta_norm) - 0.01*w**2*np.cos(theta_norm) -0.001*x**2 
    
def reward_10(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.01*w**2 + 1
    else:
        return  np.cos(theta_norm) - 0.001*w**2 -0.001*x**2 
    
def reward_9(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/2):
        return  2*np.cos(theta_norm) - 0.1*w**2 + 1
    else:
        return  np.cos(theta_norm) - 0.001*w**2 -0.001*v**2 
    
    
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
        return  np.cos(theta_norm) - 0.008*w**2
    
def reward_7(cart_pole):
    theta_norm = angle_normalize(cart_pole.state[2])
    w = cart_pole.state[3]
    x = cart_pole.state[0]
    v = cart_pole.state[1]
    
    if (x < -cart_pole.x_threshold or x > cart_pole.x_threshold):
        return -100
    elif(abs(theta_norm)<np.pi/2):
        return  np.cos(theta_norm) - 0.008*w**2 -0.001*v**2 + 5
    else:
        return  np.cos(theta_norm) - 0.008*w**2 -0.001*v**2 
        