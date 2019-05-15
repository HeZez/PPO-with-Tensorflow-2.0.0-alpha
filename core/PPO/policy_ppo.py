import tensorflow as tf
import numpy as np
from core.PPO.models import pi_categorical_model, pi_gaussian_model, v_model
from core.PPO.policy_base import PolicyBase
from core.Env import Discrete, Continuous
from utils.logger import log


class Policy_PPO(PolicyBase):

    def __init__(self,
                 policy_params=dict(), 
                 env_info = Discrete):

        super().__init__(**policy_params, env_info= env_info)

        # Decide which model to choose
        if isinstance(self.env_info, Discrete):
            self.pi = pi_categorical_model(self.hidden_sizes_pi, self.env_info)
        elif isinstance(self.env_info, Continuous):
            self.pi = pi_gaussian_model(self.hidden_sizes_pi, self.env_info)

        self.v = v_model(self.hidden_sizes_v, self.env_info)


    def update(self, observations, actions, advs, returns, logp_t):
        '''
        Update the Policy Gradient and the Value Network
        '''
        for i in range(self.train_pi_iters):
            loss_pi, loss_entropy, approx_ent, kl = self.train_pi_one_step(observations, actions, advs, logp_t)
            if kl > 1.5 * self.target_kl:
                log("Early stopping at step %d due to reaching max kl." %i)
                break

        for _ in range(self.train_v_iters):
            loss_v = self.train_v_one_step(observations, returns)
            
        # Return Metrics
        return loss_pi.numpy().mean(), loss_entropy.numpy().mean(), approx_ent.numpy().mean(), kl.numpy().mean(), loss_v.numpy().mean()
        

    def _value_loss(self, returns, values):

        # Mean Squared Error
        loss = tf.reduce_mean(tf.square(returns - values))
        return loss 


    def _pi_loss(self, logits_or_mu, logp_old, act, adv):

        # PPO Objective 
        logp = self.pi.logp(logits_or_mu, act)
        ratio = tf.exp(logp-logp_old)
        min_adv = tf.where(adv > 0, (1+ self.clip_ratio) * adv, (1-self.clip_ratio) * adv)

        # Policy Gradient Loss
        pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))

        # Entropy loss | Gaussian Policy --> returns Entropy based on log_std | Categorical Policy --> returns entropy based on logits
        entropy = self.pi.entropy(logits_or_mu)
        entropy_loss = tf.reduce_mean(entropy)

        # Total Loss
        pi_loss -= self.ent_coef * entropy_loss

        # Approximated  Kullback Leibler Divergence from OLD and NEW Policy
        approx_kl = tf.reduce_mean(logp_old-logp)
        approx_ent = tf.reduce_mean(-logp) 

        return pi_loss, entropy_loss, approx_ent, approx_kl

    
    @tf.function
    def train_pi_one_step(self, obs, act, adv, logp_old):

        with tf.GradientTape() as tape:

            logits_or_mu = self.pi(obs)
            pi_loss, entropy_loss, approx_ent, approx_kl  = self._pi_loss(logits_or_mu, logp_old, act, adv)
            
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

    









