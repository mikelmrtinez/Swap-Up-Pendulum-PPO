# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 18:25:23 2020

@author: Mikel
"""

    
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from env.continuous_cartpole import ContinuousCartPoleEnv
from mpl_toolkits.mplot3d import Axes3D
from utils.rewards import reward_15

reward = reward_15

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
if __name__ == '__main__':
    #interactive plotting in separate window
    
    env = ContinuousCartPoleEnv()
    x_threshold = env.x_threshold
    
    x_space = np.linspace(-x_threshold-2, x_threshold+2, 100)
    theta_space = np.linspace(-np.pi, np.pi, 100)
    w_space = np.linspace(-10, 10, 100)
    
    X, T = np.meshgrid(x_space, theta_space)
    
    coordinates = np.array([X, T])
    
    _, x_len, theta_len = coordinates.shape
    
    results_xt = []
    for x_index in range(x_len):
        for theta_index in range(theta_len):
            x, theta = coordinates[:, x_index, theta_index]
            env.state = [x, 0, theta, 0]
            results_xt.append(reward(env))
    
    results_xt = np.array(results_xt).reshape(x_len, theta_len)
    print('Max value:' , np.max(results_xt))
    print('Min value:' , np.min(results_xt))
    ax1.plot_surface(X, T, results_xt, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('x')
    ax1.set_ylabel('theta')
    ax1.set_zlabel('reward')
    ax1.set_title('X vs Theta')
    plt.savefig('./results/plots/reward_15_x_theta')

    plt.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    
    Z, W = np.meshgrid(theta_space, w_space)
    
    coordinates_2 = np.array([Z, W])
    
    _, theta_len, w_len = coordinates_2.shape
    
    results_tw = []
    for t_index in range(theta_len):
        for w_index in range(w_len):
            theta, w = coordinates_2[:, t_index, w_index]
            env.state = [0, 0, theta, w]
            results_tw.append(reward(env))
    
    results_tw = np.array(results_tw).reshape(theta_len, w_len)
    print('Max value:' , np.max(results_tw))
    print('Min value:' , np.min(results_tw))
    
    ax2.plot_surface(Z, W, results_tw, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax2.set_xlabel('theta')
    ax2.set_ylabel('w')
    ax2.set_zlabel('reward')
    ax2.set_title('theta vs w')
    plt.savefig('./results/plots/reward_15_theta_w')
    plt.show()