import tensorflow as tf
import numpy as np
from core.SAC.models import pi_gaussian_model, vf_model
from core.PPO.policy_base import PolicyBase
from core.Env import Discrete, Continuous
from utils.logger import log


class Policy_SAC(PolicyBase):

    def __init__(self,
                 policy_params=dict(), 
                 env_info = Discrete):

        super().__init__(**policy_params, env_info= env_info)

        self.alpha = 0.1
        
        self.pi = pi_gaussian_model(self.hidden_sizes_pi, self.env_info)

        self.q1 = vf_model(self.hidden_sizes_v, self.env_info)
        self.q1_pi = vf_model(self.hidden_sizes_v, self.env_info)
        self.q2 = vf_model(self.hidden_sizes_v, self.env_info)
        self.q2_pi = vf_model(self.hidden_sizes_v, self.env_info)

        self.v = vf_model(self.hidden_sizes_v, self.env_info)

    # @tf.function not working here
    def update(self, observations, actions, advs, returns, logp_t):
        '''
        Update 
        '''
        for i in range(self.train_pi_iters):
            self.train_pi_one_step(observations, actions)
        # Return Metrics
        
        
        
    def _mean_squared_error(self, targets, values):
        # Mean Squared Error
        loss = 0.5 * tf.reduce_mean(tf.square(targets - values))
        return loss 


    def _pi_loss(self, mu, log_std, act, ):
        logp_pi = self.pi.gaussian_likelihood(act, mu, log_std)
        pi_loss = tf.reduce_mean(self.alpha * logp_pi - self.q1_pi)
        return pi_loss

    def train_pi_one_step(self, obs, act):
        with tf.GradientTape() as tape:
            mu, log_std= self.pi(obs)
            pi_loss = self._pi_loss(mu, log_std, act)
        grads = tape.gradient(pi_loss, self.pi.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer_pi.apply_gradients(zip(grads, self.pi.trainable_variables))
        return pi_loss

    def train_q1_one_step(self, obs, obs2, returns, dones):

        q_backup = tf.stop_gradient(returns + 0.99*(1-dones)*self.v.predict(obs2))

        with tf.GradientTape() as tape:

            values = self.q1(obs)
            v_loss = self._mean_squared_error(q_backup, values)

        grads = tape.gradient(v_loss, self.v.trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer_v.apply_gradients(zip(grads, self.v.trainable_variables))

        return v_loss

    









