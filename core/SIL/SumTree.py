import numpy as np

class SumTree:

    '''
    SumTree is binary tree data structure where the parents value is the sum of its children priorities. 
    In the leaf nodes you can find priorities and the corresponding data-object (o,a,r)

    See: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/

    Goal is searching fast for a priority which is sampled by a random uniform distribution
    Priority is searched over SUM of priorities

    So you dont have to go through a sorted Array from max priority to min priority 
    Probability of being sampled is Pi = priority[i] / Sum od all priorities
    '''

    def __init__(self, capacity):

        self._capacity = capacity
        self._tree = np.zeros(2 * self._capacity - 1) # Because of Binary structure 
        self._n_entries = 0
        

    @property
    def n_entries(self):
        return self._n_entries


    def _propagate(self,idx,change):
        '''
        Updates everything from idx as start to the root node
        Propagates the changes all up from the leafes to the root node
        '''

        # Division without digits after the decimal point
        # Get the parent node of child node
        parent = (idx-1) // 2 

        self._tree[parent] +=change

        # Recursion until root node is reached
        if parent !=0:
            self._propagate(parent,change)


    def _retrieve(self, idx, sum):
        '''
        Find a sample on a leaf node; Left nodes are less than the root nodes which is less than the right nodes

                    N
                   / \
                 C1   C2

                 In such a Node trinagle values always ar getting bigger from left to the right  C1 < N < C2

                 Searching for a Value:

                 if Value is bigger on Node go to the right side 
                 if Value is smaller on Node go to the left side
        '''

        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx
        
        if sum <= self._tree[left]:
            return self._retrieve(left, sum)

        else: 
            return self._retrieve(right, sum - self._tree[left])


    @property
    def total_sum(self):
        return self._tree[0]
    

    def add(self, data_idx, priority):
        '''
        Stores a priority and the Date belonging to the priority
        '''

        # index for tree leafes begins at 
        # data_idx + capacity of tree Array --> tree Array size = 2 * capacity
        idx = data_idx + self._capacity - 1

        self.update(idx, priority)
        
        if self._n_entries < self._capacity:
            self._n_entries += 1


    def update(self, idx, priority):
        '''
        Update Priority in _tree
        '''

        change = priority - self._tree[idx]

        self._tree[idx] = priority
        self._propagate(idx, change)


    def get_priority(self, idx):
        # idx from sum tree
        return self._tree[idx]


    def find_prefixsum_idx(self, mass):
        '''
        Get Priority and Sample
        '''
        # idx from sum tree
        idx = self._retrieve(0, mass)
        return idx 
