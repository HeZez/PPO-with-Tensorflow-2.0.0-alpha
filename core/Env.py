from mlagents.envs import UnityEnvironment
from utils.logger import log
import gym


class UnityEnv():
    def __init__(self, env_name="", seed=0):

        self.env_name = env_name

        # Start ML Agents Environment | Without filename in editor training is started
        log("ML AGENTS INFO")
        if self.env_name=="":
            self.env = UnityEnvironment(file_name = None, seed=seed)
        else:
            self.env = UnityEnvironment(file_name = env_name, seed = seed)
        log("END ML AGENTS INFO")

        self.info = self.env.reset()[self.default_brain_name]
        # print(self.info)

    @property
    def _get_env(self):
        return self.env

    @property 
    def action_space_type(self):
        return self.default_brain.vector_action_space_type

    @property
    def default_brain(self):
        return self.env.brains[self.default_brain_name]

    @property
    def get_env_academy_name(self):
        return self.env.academy_name

    @property
    def default_brain_name(self):
        return self.env.brain_names[0]

    @property
    def num_actions(self):
        return self.default_brain.vector_action_space_size[0]
        
    @property
    def is_visual(self):

        if len(self.info.visual_observations[0]) is not 0:
            return True
        else:
            return False
    
    @property
    def num_obs(self):
        return self.info.vector_observations.size

    
    def reset(self):

        info = self.env.reset()[self.default_brain_name]

        if self.is_visual:
            o = info.visual_observations[0][0]
        else:
            o = info.vector_observations[0][None, :]

        r, d = 0, False
        return o, r, d


    def step(self, a):

        if self.action_space_type == 'continuous':
            info = self.env.step(a)[self.default_brain_name]
        else:
            info = self.env.step([a])[self.default_brain_name] # a is int here

        r = info.rewards[0]
        d = info.local_done[0]

        if self.is_visual:
            o = info.visual_observations[0][0]
        else:
            o = info.vector_observations[0][None, :]
        
        return o, r, d
        


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




  
        
        

        