import numpy as np
import tensorflow as tf
from core.SIL.prioritized_experinece_replay import PrioritizedReplayBuffer 
from utils.logger import log

class SIL:
    '''
    SIL Implementation following (not exactly):
    https://arxiv.org/pdf/1806.05635.pdf

    '''

    def __init__(self,
                    use_sil= False,
                    batch_size_sil= 100,
                    sil_iters= 1, 

                    pi = None, 
                    v = None, 
                    optimizer_pi = None, 
                    optimizer_v = None, 

                    num_actions = None,  
                    **kwargs):

        # SIL Arguments
        self.use_sil = use_sil
        self.sil_iters = sil_iters
        self.w_value = 1

        self.batch_size_sil = batch_size_sil

        self.pi = pi
        self.v = v
        self.optimizer_pi = optimizer_pi
        self.optimizer_v = optimizer_v

        self.num_actions = num_actions

        self.per_buffer = PrioritizedReplayBuffer(500000)


    def update_SIL(self): 
        '''
        Update Cycle for SIL if SIL is activated
        '''
        o, a, R, idxs, is_weights = self.per_buffer.sample(self.batch_size_sil)

        for _ in range(self.sil_iters):
            loss_pi, adv, loss_v = self.train_sil_one_step(o, a, R, is_weights)
            
        self.per_buffer.update_priorities(idxs, adv)
                
        return loss_pi.numpy().mean(), loss_v.numpy().mean()
    

    def train_sil_policy_one_step(self, obs, act, R, is_weights):

        with tf.GradientTape() as tape:

            logits = self.pi(obs)
            V_Pred = self.v.get_value(obs)

            pi_loss, adv  = self.sil_policy_loss(logits, act, R, V_Pred, is_weights) 
            
        grads = tape.gradient(pi_loss, self.pi.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer_pi.apply_gradients(zip(grads, self.pi.trainable_variables))

        return pi_loss, adv

    
    def train_sil_value_one_step(self, obs, R, is_weights):

        with tf.GradientTape() as tape:

            V_Pred = self.v(obs)
            v_loss = self.sil_value_loss(R, V_Pred, is_weights) 

        grads = tape.gradient(v_loss, self.v.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer_v.apply_gradients(zip(grads, self.v.trainable_variables))

        return v_loss

    def train_sil_one_step(self, obs, act, R, is_weights):

        pi_loss, adv = self.train_sil_policy_one_step(obs, act, R, is_weights)
        v_loss = self.train_sil_value_one_step(obs, R, is_weights)

        return pi_loss, adv, v_loss


    def sil_policy_loss(self, logits_or_mu, act, R, V_Pred, is_weights):
        '''
            sil_policy_loss = -log_prob * max(R - V_Pred, 0)
            sil_val_loss = 0.5 * max(R - V_Pred, 0) ** 2
            Called on Batch sampled from PER

        '''
        # logp = self.log_probs(logits, act)
        logp = self.pi.logp(logits_or_mu, act)

        clipped_advs2 = tf.math.maximum(R - tf.squeeze(V_Pred), 0)
        clipped_advs = tf.clip_by_value(R - tf.squeeze(V_Pred), 0, 1)

        sil_policy_loss = -self.w_value * tf.reduce_mean(is_weights * clipped_advs * logp)

        return sil_policy_loss, clipped_advs2
    

    def sil_value_loss(self, R, V_Pred, is_weights):
        '''
            sil_policy_loss = -log_prob * max(R - V_Pred, 0)
            sil_val_loss = 0.5 * max(R - V_Pred, 0) ** 2
            Called on Batch sampled from PER

        '''
        # clipped_advs2 = tf.math.maximum(R - tf.squeeze(V_Pred), 0)
        clipped_advs = tf.clip_by_value(R - tf.squeeze(V_Pred), 0, 1)
        sil_val_loss = 0.5 * self.w_value * tf.reduce_mean(is_weights * clipped_advs**2)
        
        return sil_val_loss

    
    def log_probs(self, logits, act):

        # make logp for policy_gradient update
        logp_all = tf.nn.log_softmax(logits)
        logp = tf.reduce_sum( tf.one_hot(act, self.num_actions) * logp_all, axis=1)

        return logp


    def add_episode_to_per(self, trajectory):

        o, a, R = trajectory 

        # Only add good trajectories
        R_mean = np.mean(R)

        if R_mean > 0:
            for idx in range(len(o)):
                if R[idx] >= 0:
                    self.per_buffer.add(o[idx],a[idx],R[idx])

        