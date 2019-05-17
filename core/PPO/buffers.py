import numpy as np
from core.Env import Continuous, Discrete


def statistics_scalar(x):
    """
    Get mean/std  of scalar x.
    Args: x: An array containing samples of the scalar to produce statistics for.
    """
    mean = np.mean(x)       
    std = np.std(x)
    return mean, std

def create_buffers(size, env_info=Discrete):
    '''
    Creates Obs and Act buffers with desired shape and Action Type
    '''
    obs_buf = np.zeros((size,) + env_info.obs_shape, dtype= np.float32)
    if isinstance(env_info, Discrete):
        act_buf = np.zeros((size,), dtype= np.int32)
    elif isinstance(env_info, Continuous):
        act_buf = np.zeros((size, env_info.act_size), dtype= np.float32)  
    return obs_buf, act_buf

def discount_cum_sum(vetcor, discount):
    '''
    Retruns the Discounted Cum Sum --> 

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    '''
    vetcor_length = len(vetcor)
    dcs = np.zeros_like(vetcor, dtype= np.float32)
    for i in reversed(range(vetcor_length)):
        dcs[i] = vetcor[i] + (discount * dcs[i+1] if i+1 < vetcor_length else 0)
    return dcs

def gae_lambda_advantage(rews, vals, gamma, lam):
    '''
    GAE --> https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/

    1. Define the temporal difference residuals td(t) = r(t) + gamma * V(st+1) - V(st)
    2. GAE --> A(gae) = Sum(l= 0 to infinity)[(gamma*lam)**l * td(t + l)]
    '''
    # [:-1] means take all elements of array except the last
    # [1:] means take all elements of array except the first --> the second element is now the first V(st+1)
    td_deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
    gae = discount_cum_sum (td_deltas, gamma * lam)
    return gae


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
        # obs, act = self.delete_non_act()
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


class Buffer_PPO:
    '''
    This is the Buffer for the PPO Algorithm
    '''

    def __init__(self, size, env_info= Discrete, gamma= 0.99, lam= 0.95):

        self.obs_buf, self.act_buf = create_buffers(size, env_info)

        self.adv_buf = np.zeros((size,), dtype=np.float32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.ret_buf = np.zeros((size,), dtype=np.float32)
        self.val_buf = np.zeros((size,), dtype=np.float32)
        self.logp_buf = np.zeros((size,), dtype=np.float32)

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

        self.trajectory = None
 

    def store(self, obs, act, rew, val, logp):

        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    
    def finish_path(self, last_val=0):

        # Slices the path which to bootstrap
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE --> Advantages for PPO Update
        self.adv_buf[path_slice] = gae_lambda_advantage(rews, vals, self.gamma, self.lam) 

        # The next line computes Rewards-To-Go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cum_sum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

        # Save trajecory for SIL Episodes
        self.trajectory = [self.obs_buf[path_slice], self.act_buf[path_slice], self.ret_buf[path_slice]] 
        
    def get_trajectory(self):
        return self.trajectory

    def get(self):

        # Buffer has to be full before you can get and reset
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0

        # The next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]


    