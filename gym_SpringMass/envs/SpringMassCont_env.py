#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:02:13 2021

@author: dsalarc
"""

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class SpringMassContinuous(gym.Env):
    """Custom Environment that follows gym interface"""
    #metadata = {'render.modes': ['human']}
    metadata = {'render.modes': ['console']}

    def __init__(self, K=1, M=1, C=0, g = 0, MaxForce = 3):
        # super(SpringMassContinuous, self).__init__() 
        
        #Define Constants
        self.K = K
        self.M = M
        self.C = C
        self.g = g
        self.MaxForce = MaxForce
        
        self.n_states = 2
        self.t_step = 0.05
        
        # Define action and observation space
           
        self.MaxState = np.array([20,20])
        
        self.action_space = spaces.Box(low=-self.MaxForce, 
                                       high=self.MaxForce,
                                       shape=(1,),
                                       dtype=np.float16)
        
        self.observation_space = spaces.Box(low=-self.MaxState,
                                            high=self.MaxState,
                                            dtype=np.float16)
  
    def Calc_Xdot(self,U,X,Xdot):
        Last_Xdot = Xdot
        
        Xdot = np.zeros(self.n_states);
        Xdot[0] = X[1]
        Xdot[1] = (U[0] -X[0]*self.K - X[1]*self.C)/self.M + self.g
        
        Xdotdot = (Xdot - Last_Xdot)/self.t_step
        return Xdot, Xdotdot
    
    def Euler_2nd(self,X,Xdot,Xdotdot,t_step):
        
        X = X + Xdot*t_step + (Xdotdot*t_step**2)/2
        return X

    def step(self, action):
      self.CurrentStep += 1
      
      Force = np.array([0],dtype = np.float32)
      Force[0] = min(max(action[0], -self.MaxForce), self.MaxForce)
      # Execute one time step within the environment
      
      # Force = np.array([self.ForceVec[action]])
      
      self.Xdot, self.Xdotdot = self.Calc_Xdot(Force,self.X,self.Xdot)
      self.X = self.Euler_2nd(self.X,self.Xdot,self.Xdotdot,self.t_step)
      self.AllStates = np.vstack((self.AllStates,self.X))

      
      # Calculate Reward
      self.LastReward = self.CalcReward()
      
      # Terminal State
      if self.LastReward > 0.99:
          done = True
      else:
          done = False
      
      done = False
      # Optionally we can pass additional info, we are not using that for now
      info = {'Force': Force}          
      
      return self.X, self.LastReward, done, info
      
    def CalcReward(self):
        # Reward = 1 - np.sqrt(np.sum((self.X / self.MaxState)**2))
        Reward = 10 - (self.X[0]**2 + self.X[1]**2)/10
        return Reward    
    
    def reset(self):
      # Reset the state of the environment to an initial state
      randomVec = np.random.random(self.n_states)
      # self.X = (self.observation_space.high + self.observation_space.low)/2 + (self.observation_space.high - self.observation_space.low)/np.sqrt(2) * (randomVec-.5) 
      self.X = np.array([10,0])
      self.Xdot    = self.X * 0
      
      self.CurrentStep = 0
      
      self.AllStates = self.X
      
      return self.X
      
    def render(self, mode='console', close=False):
        
        # Render the environment to the screen       
        plt.figure(1)
        plt.clf()
        plt.grid('on')
        plt.xlim((self.observation_space.low[0],self.observation_space.high[0]))
        plt.ylim((self.observation_space.low[1],self.observation_space.high[1]))
        if self.CurrentStep > 0:
            plt.plot(self.AllStates[:,0],self.AllStates[:,1],'tab:gray')
            plt.plot(self.AllStates[-1,0],self.AllStates[-1,1],'ob')
        else:
            plt.plot(self.AllStates[0],self.AllStates[1],'ob')
        plt.xlabel('Position X [m]')
        plt.ylabel('Velocity V [m/s]')
        plt.show()
        
    def close(self):
        pass
        
