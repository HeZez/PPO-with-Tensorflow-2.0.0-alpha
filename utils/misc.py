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
    if env_info.is_frame_stacking:
        # shape = (4,) + env_info.obs_shape
        shape = env_info.obs_shape
    else:
        shape = env_info.obs_shape

    obs_buf = np.zeros((size,) + shape, dtype= np.float32)
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




    