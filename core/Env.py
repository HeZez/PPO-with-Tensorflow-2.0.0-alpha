from mlagents.envs import UnityEnvironment
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.logger import log
from collections import deque


class EnvInfo():
    def __init__(self, env_name, is_behavioral_cloning, is_visual, obs_shape, act_size, is_frame_stacking):
        self.env_name= env_name
        self.is_behavioral_cloning= is_behavioral_cloning
        self.is_visual= is_visual
        self.is_frame_stacking = is_frame_stacking
        self.obs_shape= obs_shape
        self.act_size= act_size

class Discrete(EnvInfo):
    def __init__(self, env_name, is_behavioral_cloning, is_visual, obs_shape, act_size, is_frame_stacking):
        super().__init__(env_name, is_behavioral_cloning, is_visual, obs_shape, act_size, is_frame_stacking)

class Continuous(EnvInfo):
    def __init__(self, env_name, is_behavioral_cloning, is_visual, obs_shape, act_size, is_frame_stacking):
        super().__init__(env_name, is_behavioral_cloning, is_visual, obs_shape, act_size, is_frame_stacking)


class UnityEnv():

    def __init__(self, env_name= "", seed= 0, frame_stacking= False):

        self._env_name = env_name
        self._bool_is_behavioral_cloning = False
        self._bool_is_visual = False
        self._bool_is_grayscale = False

        self._default_brain_name = 'Student'
        self._default_brain = None

        self._teacher_brain_name = 'Teacher'
        self._teacher_brain = None

        self._shape = None

        self._bool_frame_stacking = frame_stacking
        self._stack_size = 4
        self._frames = deque([np.zeros((84,84,3), dtype=np.float) for i in range(self._stack_size)], maxlen= self._stack_size)
        

        # Start ML Agents Environment | Without filename in editor training is started
        log("ML AGENTS INFO")
        if self._env_name == "":
            self._env = UnityEnvironment(file_name= None, seed= seed)
        else:
            self._env = UnityEnvironment(file_name= self._env_name, seed= seed)
        log("END ML AGENTS INFO")


    	# Checks if is behaviroal cloning -> is true if Brain with name 'Teacher' is found
        # Gets the teacher brain
        for name in self._env.brain_names:
            if name == self._teacher_brain_name:
                self._bool_is_behavioral_cloning = True
                self._teacher_brain = self._env.brains[self._teacher_brain_name]
                break


        # default Brain name is Student if behaviroal Cloning is true otherwise takes the first Brain 
        # Take care of this in the Brain Acadmey and names of brains
        if not self._bool_is_behavioral_cloning:
            self._default_brain_name = self._env.brain_names[0]


        # get the default brain --> can be the first brain or Student brain if behavioral cloning is true
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
            plt.ion()
            plt.show()
            if self._shape[2] == 3:
                self._bool_is_grayscale = False
                o = self._info.visual_observations[0][0][None, : , : , :]
                plt.imshow(o[0])
            else:
                self._bool_is_grayscale = True
                o = self._info.visual_observations[0][0][None, : , : , 0]
                plt.imshow(o[0], cmap='gray')
            plt.pause(0.001)

            # if self._bool_frame_stacking:
            #     self._shape = (4,84,84,3)


        if not self._bool_is_visual:
            self._shape = (self.num_obs,)


        if self.action_space_type== 'discrete':
            self._env_info = Discrete(env_name, self._bool_is_behavioral_cloning, self._bool_is_visual, self._shape, self.num_actions, self._bool_frame_stacking) 
            
            
        elif self.action_space_type== 'continuous':
            self._env_info = Continuous(env_name, self._bool_is_behavioral_cloning, self._bool_is_visual, self._shape, self.num_actions, self._bool_frame_stacking) 
            
    
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
    def is_behavioral_cloning(self):
        return self._bool_is_behavioral_cloning

    @property
    def teacher_brain(self):
        if self._bool_is_behavioral_cloning:
            return self._teacher_brain

    @property
    def teacher_brain_name(self):
        if self._bool_is_behavioral_cloning:
            return self._teacher_brain_name  
    
    @property
    def num_obs(self):
        return self._info.vector_observations.size

    @property
    def shape(self):
        return self._shape

    def _get_obs_from_info(self, info):

        if self._bool_is_visual:
            o_student = info[self._default_brain_name].visual_observations[0][0]
            o_student = o_student[None, : , : , :]

            o_teacher = info[self._teacher_brain_name].visual_observations[0][0]
            o_teacher = o_teacher[None, : , : , :]

        else:
            o_student = info[self._default_brain_name].vector_observations[0][None, :]
            o_teacher = info[self._teacher_brain_name].vector_observations[0][None, :]
        
        return o_student, o_teacher


    def reset(self):

        info = self._env.reset() 

        if self._bool_is_behavioral_cloning:

            o_student, o_teacher = self._get_obs_from_info(info)
            act_teacher = info[self._teacher_brain_name].previous_vector_actions[0]
            r, d = 0, False
            return o_student, r, d, o_teacher, act_teacher

        else:

            if self._bool_is_visual:
                o = info[self._default_brain_name].visual_observations[0][0]
                o = o[None, : , : , :]

                # for _ in range(self._stack_size):
                #     self._frames.append(o[0])
                # o = tf.concat(list(self._frames), axis=-1)

                if self._bool_frame_stacking:
                    for _ in range(self._stack_size):
                        self._frames.append(o[0]) 
                    stacked_obs = np.stack(self._frames)
                    o = stacked_obs

            else:
                o = info[self._default_brain_name].vector_observations[0][None, :]

            r, d = 0, False
            return o, r, d


    action = dict()

    def step(self, a):

        if self.is_behavioral_cloning:

            if self.action_space_type == 'continuous':
                self.action[self._default_brain_name] = a
                info = self._env.step(self.action)
            else:
                self.action[self._default_brain_name] = [a]
                info = self._env.step(self.action)

            r = info[self._default_brain_name].rewards[0]
            d = info[self._default_brain_name].local_done[0]

            o_student, o_teacher = self._get_obs_from_info(info)
            act_teacher = info[self._teacher_brain_name].previous_vector_actions[0]

            return o_student, r, d, o_teacher, act_teacher

        else:

            if self.action_space_type == 'continuous':
                info = self._env.step(a)
            else:
                info = self._env.step([a]) # a is int here

            r = info[self._default_brain_name].rewards[0]
            d = info[self._default_brain_name].local_done[0]

            if self._bool_is_visual:
                o = info[self._default_brain_name].visual_observations[0][0]
                o = o[None, : , : , :]

                if self._bool_frame_stacking:
                    self._frames.append(o[0]) 
                    stacked_obs = np.stack(self._frames)
                    o = stacked_obs

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




  
        
        

        