import tensorflow as tf
import numpy as np

from core.PPO.models_categorical import pi_model, v_model, pi_model_with_conv, v_model_with_conv
from core.PPO.policy_base import PolicyBase

from utils.logger import log


class Policy_PPO_Categorical(PolicyBase):

    def __init__(self,
                 policy_params=dict(), 
                 num_actions = None, 
                 is_visual = False):

        super().__init__(**policy_params, num_actions= num_actions)

        if is_visual:
            self.pi = pi_model_with_conv(self.hidden_sizes_pi, self.num_actions)
            self.v = v_model_with_conv(self.hidden_sizes_v)
        else:
            self.pi = pi_model(self.hidden_sizes_pi, self.num_actions)
            self.v = v_model(self.hidden_sizes_v)


    def update(self, observations, actions, advs, returns, logp_t):
        '''
        Update the Policy Gradient and the Value Network
        '''

        # Policy Update Cycle for iter steps
        for i in range(self.train_pi_iters):
            loss_pi, loss_entropy, approx_ent, kl = self.train_pi_one_step(observations, actions, advs, logp_t)
            if kl > 1.5 * self.target_kl:
                log("Early stopping at step %d due to reaching max kl." %i)
                break

        # Value Update Cycle for iter steps
        for _ in range(self.train_v_iters):
            loss_v = self.train_v_one_step(observations, returns)
            
        # Return Metrics
        return loss_pi.numpy().mean(), loss_entropy.numpy().mean(), approx_ent.numpy().mean(), kl.numpy().mean(), loss_v.numpy().mean()
        

    def _value_loss(self, returns, values):

        # Mean Squared Error
        loss = tf.reduce_mean(tf.square(returns - values))
        return loss 


    def _pi_loss(self, logits, logp_old, act, adv):

        # PPO Objective 
        logp_all = tf.nn.log_softmax(logits)
        logp = tf.reduce_sum( tf.one_hot(act, self.num_actions) * logp_all, axis=1)
        ratio = tf.exp(logp-logp_old)
        min_adv = tf.where(adv > 0, (1+ self.clip_ratio) * adv, (1-self.clip_ratio) * adv)

        # Policy Gradient Loss
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))

        # Entropy loss 
        entropy_loss = tf.reduce_mean(self.entropy(logits))
        # entropy_loss = tf.reduce_mean(self.entropy2(logits)) 

        # Total Loss
        pi_loss -= self.ent_coef * entropy_loss

        # Approximated  Kullback Leibler Divergence from OLD and NEW Policy
        approx_kl = tf.reduce_mean(logp_old-logp)
        approx_ent = tf.reduce_mean(-logp) 

        return pi_loss, entropy_loss, approx_ent, approx_kl

    
    @tf.function
    def train_pi_one_step(self, obs, act, adv, logp_old):

        with tf.GradientTape() as tape:

            logits = self.pi(obs)
            pi_loss, entropy_loss, approx_ent, approx_kl  = self._pi_loss(logits, logp_old, act, adv)
            
        grads = tape.gradient(pi_loss, self.pi.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer_pi.apply_gradients(zip(grads, self.pi.trainable_variables))

        return pi_loss, entropy_loss, approx_ent, approx_kl


    @tf.function
    def train_v_one_step(self, obs, returns):

        with tf.GradientTape() as tape:

            values = self.v(obs)
            v_loss = self._value_loss(returns, values)

        grads = tape.gradient(v_loss, self.v.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer_v.apply_gradients(zip(grads, self.v.trainable_variables))

        return v_loss


    def entropy(self, logits):
        '''
        Entropy term for more randomness which means more exploration in ppo -> 
        
        Due to machine precission error -> 
        entropy = - tf.reduce_sum (logp_all * tf.log(logp_all) + 1E-12, axis=-1, keepdims=True) 
        cannot be calculated this way
        '''
        
        a0 = logits - tf.reduce_max(logits, axis= -1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis= -1, keepdims=True)
        p0 = exp_a0 / z0
        entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis= -1)

        return entropy


    def entropy2(self, logits):
        '''
        Entropy calc with keras over logits ??
        # calculated with Cross Entropy over logits with itself ??
        '''
        entropy = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        return entropy

    def cat_entropy_softmax(self, p0):
        return -tf.reduce_sum(p0 * tf.math.log(p0 + 1e-6), axis = 1)

    def mse(self, pred, target):
        return tf.square(pred-target)/2.0









