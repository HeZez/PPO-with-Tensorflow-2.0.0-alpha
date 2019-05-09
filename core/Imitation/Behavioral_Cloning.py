import numpy as np
import tensorflow as tf
from utils.logger import log


class Behavioral_Cloning:

    '''
    BC Implementation
    '''

    def __init__(self,
                    use_bc= True,
                    bc_iters= 100, 

                    pi = None, 
                    v = None, 
                    optimizer_pi = None, 
                    optimizer_v = None, 

                    num_actions = None,  
                    **kwargs):

        # BC Arguments
        self.use_bc = use_bc
        self.bc_iters = bc_iters

        self.pi = pi
        self.v = v
        self.optimizer_pi = optimizer_pi
        self.optimizer_v = optimizer_v

        self.num_actions = num_actions

        # Im Buffer ??


    def update_BC(self, obs, acts): 
        

        for _ in range(self.bc_iters):
            loss = self.train_BC_one_step(obs, acts)
            

        print('loss BC ==> pi: ' + str(loss.numpy().mean()))
                
        return loss
    

    def train_BC_one_step(self, obs, acts):

        with tf.GradientTape() as tape:

            logits = self.pi(obs)

            loss = self.bc_loss(logits, acts)
            
        grads = tape.gradient(loss, self.pi.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer_pi.apply_gradients(zip(grads, self.pi.trainable_variables))

        return loss
    

    def bc_loss(self, logits, acts):
        '''
            BC Loss

        '''
        labels_one_hot = tf.one_hot(acts, self.num_actions)
        loss = tf.keras.losses.categorical_crossentropy(labels_one_hot, logits, from_logits=True)
        
        return loss




        