import random
import numpy as np
from core.SIL.SumTree import SumTree


class ReplayBuffer(object):

    def __init__(self, size):

        """
        Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. 
            When the buffer overflows the old memories are dropped.
        """

        self._storage = [] 
        self._maxsize = size
        self._next_idx = 0


    def __len__(self):
        return len(self._storage)
    

    def add(self, obs_t, action, R):

        data = (obs_t, action, R)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize


    def _encode_sample(self, idxes):

        obses_t, actions, returns= [], [], []

        for i in idxes:

            data = self._storage[i]
            obs_t, action, R = data

            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            returns.append(R)

        return [np.array(obses_t), np.array(actions), np.array(returns)]


    def sample_random_from_replay(self, batch_size):

        """
        Sample a Random batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):

    e = 0.01
    beta_increment_per_sampling = 0.001
    _max_priority = 1.0

    def __init__(self, size, alpha= 0.6, beta= 0.4):

        '''
        size: int --> Max Number of transitions. If overflows old memories are dropped.
        alpha: float 
            How much priorizaitaion is used
            (0 - no priorizatiaion, 1 - full priorizitation)

        '''

        super(PrioritizedReplayBuffer, self).__init__(size)
        
        self._alpha = alpha
        self._beta = beta
        self._tree = SumTree(size)
        self._size = size

    
    def add(self, obs, act, R):

        data_idx = self._next_idx
        super().add(obs, act, R)

        self._tree.add(data_idx, self._max_priority ** self._alpha)


    def _sample_proportional(self, batch_size):

        res = []

        for _ in range(batch_size):
       
            mass = random.random() * self._tree.total_sum
            idx = self._tree.find_prefixsum_idx(mass)
            res.append(idx)

        return res


    def _sample_rank(self, batch_size):

        res = []
        segment = self._tree.total_sum / batch_size

        for i in range(batch_size):

            a = segment * i
            b = segment * (i + 1)
            mass = random.uniform(a,b)

            idx = self._tree.find_prefixsum_idx(mass)
            res.append(idx)

        return res


    def sample(self, batch_size):
        '''
        Returns obs, acts, returns, idxs, is_weights
        Priorities are saved in the leaves of the binary SumTree
        '''

        idxs = self._sample_proportional(batch_size)

        priorities = [self._tree.get_priority(idx) for idx in idxs]

        self._beta = np.min([1., self._beta + self.beta_increment_per_sampling]) # max = 1.0

        # Sampling probability (for priority) = priority / SUM of priorities
        p_sample = priorities / self._tree.total_sum

        # Importance sampling weight is_weight = [(1/n * 1/p_sample) ** beta] / max_is_weight
        is_weight = np.power(self._tree.n_entries * p_sample, -self._beta)
        is_weight /= is_weight.max()

        # Get Samples (obs, acts, R) | must calc the right index of data
        data_Idxs = [idx - self._size + 1 for idx in idxs]
        encoded_samples = self._encode_sample(data_Idxs)

        return encoded_samples + [idxs, is_weight]
        

    
    def update_priorities(self, idxs, priorities):

        for idx, priority in zip(idxs, priorities):

            priority = max(priority, 1e-6)

            data_idx = idx - self._size + 1
            assert priority > 0
            assert 0 <= data_idx < len(self._storage)

            self._tree.update(idx, priority ** self._alpha)
            self._max_priority = max(self._max_priority, priority)