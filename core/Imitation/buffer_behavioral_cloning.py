import numpy as np
from core.Env import Continuous, Discrete
from utils.misc import create_buffers


class Buffer_Imitation:

    def __init__(self, size, env_info=Discrete):

        self.size = size
        self.env_info = env_info
        self.obs_buf, self.act_buf = create_buffers(size, env_info= self.env_info)
        self.ptr, self.max_size = 0, size

    def store(self, obs, act):

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.ptr += 1

    def get(self):

        path_slice = slice(0, self.ptr)
        act_buf = self.act_buf[path_slice]
        obs_buf = self.obs_buf[path_slice]

        # rearrange so that obs matches previous_acts act(t+1) == obs(t)
        act = act_buf[1:]
        obs = obs_buf[:-1]

        # delete non acts
        result = np.where(act!=0)
        act = act[result]
        obs = obs[result]
        self.ptr = 0

        return obs, act

    def save(self):
        obs, act = self.get()
        np.savetxt('imitationObs', obs)
        np.savetxt('imitationAct', act)
    
    def load(self):
        obs = np.loadtxt('imitationObs')
        act = np.loadtxt('imitationAct')
        return obs, act

    def sample(self, batch_size):

        obs, act = self.get() 

        if len(act) == 0:
            return [], []
        if len(act) < batch_size:
            batch_size = len(act)
            
        idxs = np.random.choice(range(len(act)), size=batch_size, replace=False)
        self.ptr = 0
    
        return self._reformat(idxs, obs, act, batch_size)

    def _reformat(self, idxs, obs, act, batch_size):
        obs_buf, act_buf = create_buffers(batch_size, env_info= self.env_info)
        obs_buf = obs[idxs]
        act_buf = act[idxs]
        return obs_buf, act_buf



    