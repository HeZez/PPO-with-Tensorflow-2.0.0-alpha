import numpy as np
from core.Env import Continuous, Discrete
from utils.misc import create_buffers, statistics_scalar, gae_lambda_advantage, discount_cum_sum


class Buffer_PPO:
    '''
    This is the Buffer for the PPO Algorithm
    '''

    def __init__(self, size, env_info= Discrete, gamma= 0.99, lam= 0.95):

        self.obs_buf, self.act_buf = create_buffers(size, env_info)

        self.adv_buf = np.zeros((size,), dtype=np.float32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.intrinsic_rew_buf = np.zeros((size,), dtype=np.float32)
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
        # self.intrinsic_rew_buf[self.ptr] = intr_rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
    
    
    def finish_path(self, last_val=0, intrinsic_model=None):

        # Slices the path which to bootstrap
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)


        next_obs = self.obs_buf[path_slice][1:]
        obs = self.obs_buf[path_slice][:-1]
        actions = self.act_buf[path_slice][:-1]

        intrinsic_rewards = intrinsic_model.get_intrinsic_reward(obs, actions, next_obs)
        rews += np.append(intrinsic_rewards.numpy(), [0,0])

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


    