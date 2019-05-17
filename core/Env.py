from mlagents.envs import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
from utils.logger import log


class EnvInfo():
    def __init__(self, env_name, is_visual, obs_shape, act_size):
        self.env_name= env_name
        self.is_visual= is_visual
        self.obs_shape= obs_shape
        self.act_size= act_size

class Discrete(EnvInfo):
    def __init__(self, env_name, is_visual, obs_shape, act_size):
        super().__init__(env_name, is_visual, obs_shape, act_size)

class Continuous(EnvInfo):
    def __init__(self, env_name, is_visual, obs_shape, act_size):
        super().__init__(env_name, is_visual, obs_shape, act_size)


class UnityEnv():

    def __init__(self, env_name= "", seed= 0):

        self._env_name = env_name
        self._bool_is_visual = False
        self._bool_is_grayscale = False

        self._default_brain_name = None
        self._default_brain = None

        self._shape = None

        # Start ML Agents Environment | Without filename in editor training is started
        log("ML AGENTS INFO")
        if self._env_name == "":
            self._env = UnityEnvironment(file_name= None, seed= seed)
        else:
            self._env = UnityEnvironment(file_name= self._env_name, seed= seed)
        log("END ML AGENTS INFO")
        
        self._default_brain_name = self._env.brain_names[0]

        # get the default brain
        self._default_brain= self._env.brains[self._default_brain_name]

        # Check if there are visual observations and set bool_is_visual
        if self._default_brain.number_visual_observations is not 0:
            self._bool_is_visual = True
        else:
            self._bool_is_visual = False

        # get infos about environment on first reset
        self._info = self._env.reset()[self._default_brain_name] 

        if self._bool_is_visual:

            self._shape = self._info.visual_observations[0][0].shape

            if self._shape[2] == 3:
                self._bool_is_grayscale = False
            else:
                self._bool_is_grayscale = True

            plt.ion()
            plt.show()

            if self._bool_is_grayscale:
                o = self._info.visual_observations[0][0][None, : , : , 0]
                plt.imshow(o[0], cmap='gray')
            else:
                o = self._info.visual_observations[0][0][None, : , : , :]
                plt.imshow(o[0])

            plt.pause(0.001)

        else:
            self._shape = (self.num_obs,)

        if self.action_space_type== 'discrete':
            self._env_info = Discrete(env_name, self._bool_is_visual, self._shape, self.num_actions) 
            
        elif self.action_space_type== 'continuous':
            self._env_info = Continuous(env_name, self._bool_is_visual, self._shape, self.num_actions) 
            
    
    @property
    def EnvInfo(self):
        return self._env_info

    @property
    def env(self):
        return self._env

    @property 
    def action_space_type(self):
        return self._default_brain.vector_action_space_type

    @property
    def default_brain(self):
        return self._default_brain

    @property
    def get_env_academy_name(self):
        return self._env.academy_name

    @property
    def default_brain_name(self):
        return self._default_brain_name

    @property
    def num_actions(self):
        return self._default_brain.vector_action_space_size[0]
        
    @property
    def is_visual(self):
        return self._bool_is_visual  
    
    @property
    def num_obs(self):
        return self._info.vector_observations.size

    @property
    def shape(self):
        return self._shape

    def reset(self):

        info = self._env.reset() 

        if self._bool_is_visual:
            o = info[self._default_brain_name].visual_observations[0][0][None, : , : , :]
        else:
            o = info[self._default_brain_name].vector_observations[0][None, :]

        r, d = 0, False
        return o, r, d


    action = dict()

    def step(self, a):
        
        if self.action_space_type == 'continuous':
            info = self._env.step(a)
        else:
            info = self._env.step([a]) # a is int here

        r = info[self._default_brain_name].rewards[0]
        d = info[self._default_brain_name].local_done[0]

        if self._bool_is_visual:
            o = info[self._default_brain_name].visual_observations[0][0][None, : , : , :]
        else:
            o = info[self._default_brain_name].vector_observations[0][None, :]
            
        return o, r, d


'''
    GYM from Open AI for fast testing
'''

import gym

class GymCartPole():
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.env.seed(0)

    @property 
    def action_space_type(self):
        return 'discrete'

    @property
    def num_actions(self):
        return 2
    
    @property
    def get_env_academy_name(self):
        return 'OpenAIGym_CartPole_v1'

    @property
    def num_obs(self):
        return 4

    def reset(self):
        o, r, d = self.env.reset(), 0, False
        return o[None, :], r, d

    def step(self, action):
        o, r, d, _ = self.env.step(action.numpy())
        return o[None, :], r, d




  
        
        

        