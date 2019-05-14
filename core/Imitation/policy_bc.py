import numpy as np
import tensorflow as tf
from utils.logger import log
from core.PPO.buffers import Buffer_Imitation


class Behavioral_Cloning:

    '''
    Behavioral Cloning Implementation --> Works only for categorical networks
    Maps Observations to actions in a supervised learning setting with Categorlical Cross Entropy
    Maybe Imitation Buffer can be integrated here!?

    '''

    def __init__(self,
                    use_bc= True,
                    batch_size_bc= 200,
                    iters_bc= 100, 

                    pi = None, 
                    v = None, 
                    optimizer_pi = None, 
                    optimizer_v = None, 

                    num_actions = None,  
                    **kwargs):

        # Behavioral Cloning Arguments
        self.use_bc = use_bc
        self.batch_size_bc = batch_size_bc
        self.iters_bc = iters_bc
        
        self.pi = pi
        self.v = v
        self.optimizer_pi = optimizer_pi
        self.optimizer_v = optimizer_v

        self.num_actions = num_actions


    def update_BC(self, obs, acts): 
        
        for _ in range(self.iters_bc):
            loss = self.train_BC_one_step(obs, acts)
                
        return loss.numpy().mean()
    

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




        