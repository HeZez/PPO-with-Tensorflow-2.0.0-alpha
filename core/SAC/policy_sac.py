import tensorflow as tf
import numpy as np
from core.SAC.models_sac import pi_gaussian_model, vf_model
from core.PPO.policy_base import PolicyBase
from core.Env import Discrete, Continuous
from utils.logger import log


class Policy_SAC(PolicyBase):

    def __init__(self,
                 policy_params=dict(), 
                 env_info = Discrete):

        super().__init__(**policy_params, env_info= env_info)

        self.alpha = 0.01
        self.polyak = 0.995
        self.gamma = 0.99
        
        self.pi = pi_gaussian_model(self.hidden_sizes_pi, self.env_info)
        self.q1 = vf_model(self.hidden_sizes_v, self.env_info)
        self.q2 = vf_model(self.hidden_sizes_v, self.env_info)
        self.v = vf_model(self.hidden_sizes_v, self.env_info)
        self.v_target = vf_model(self.hidden_sizes_v, self.env_info)

        # init v_target with v weights
        self.v_target.set_weights(self.v.get_weights())

    # @tf.function
    def update(self, batch, iters):
        '''
        Update 
        '''
        obs, obs_next, actions, returns, dones = batch['obs1'], batch['obs2'], batch['acts'], batch['rews'], batch['done']

        q1_loss, q2_loss = self.train_values_one_step(obs, obs_next, actions, returns, dones)
        v_loss = self.train_v_one_step(obs)
        pi_loss = self.train_pi_one_step(obs)
        self.target_update()
        
        # Return Metrics
        return pi_loss, q1_loss, q2_loss, v_loss
        
        
    def _mean_squared_error(self, targets, values):
        loss = 0.5 * tf.reduce_mean(tf.square(targets - values))
        return loss 


    def _pi_loss(self, mu, log_std, obs):
        '''
        maximize q1_pi by backpropagating through mu, log_std via mu, pi, logp_pi
        no backpropagation through q1_pi, just for error calculation which ist used for backpropagation through mu and log_std networks
        '''
        mu, pi, logp_pi = self.pi.mu_pi_logp_pi (mu, log_std)
        q1_pi = self.q1(tf.concat([obs, pi], axis = -1))
        pi_loss = tf.reduce_mean(self.alpha * logp_pi - q1_pi)

        return pi_loss


    def train_pi_one_step(self, obs):
        with tf.GradientTape() as tape:
            mu, log_std= self.pi(obs)
            pi_loss = self._pi_loss(mu, log_std, obs)
        grads = tape.gradient(pi_loss, self.pi.trainable_variables) # pi_loss is differentiated against pi.trainable_variables
        self.optimizer_pi.apply_gradients(zip(grads, self.pi.trainable_variables))

        return pi_loss


    def train_values_one_step(self, obs, obs_next, act, returns, dones):
        '''
        Off policy samples because act is sampled from buffer_sac
        '''
        v_target = self.v_target(obs_next)
        q_backup = tf.stop_gradient(returns + self.gamma * (1 - dones) * v_target)

        with tf.GradientTape() as tape:
            q1 = self.q1(tf.concat([obs, act], axis = -1))
            q1_loss = self._mean_squared_error(q_backup, q1)
        grads = tape.gradient(q1_loss, self.q1.trainable_variables)
        self.optimizer_q1.apply_gradients(zip(grads, self.q1.trainable_variables))

        with tf.GradientTape() as tape:
            q2 = self.q2(tf.concat([obs, act], axis = -1))
            q2_loss = self._mean_squared_error(q_backup, q2)
        grads = tape.gradient(q2_loss, self.q2.trainable_variables)
        self.optimizer_q2.apply_gradients(zip(grads, self.q2.trainable_variables))

        return q1_loss, q2_loss

    
    def train_v_one_step(self, obs):
        '''
        On Policy samples because pi is calculated on obs
        calculate q1 and q2 on last Policy pi
        Uses Double Q network trick

        '''
        # mu, log_std = self.pi(obs)
        # _, pi, logp_pi = self.pi.mu_pi_logp_pi(mu, log_std)

        _, pi, logp_pi = self.pi.get_mu_pi_logp_pi(obs)

        q1_pi = self.q1(tf.concat([obs,pi], axis = -1))
        q2_pi = self.q2(tf.concat([obs,pi], axis = -1))
        min_q_pi = tf.minimum(q1_pi, q2_pi)
        v_backup = tf.stop_gradient(min_q_pi - self.alpha * logp_pi)

        with tf.GradientTape() as tape:
            values = self.v(obs)
            v_loss = self._mean_squared_error(v_backup, values)
        grads = tape.gradient(v_loss, self.v.trainable_variables)
        self.optimizer_v.apply_gradients(zip(grads, self.v.trainable_variables))

        return v_loss

    
    def target_update(self):
        '''
        copy weights from v network to target network
        '''
        zip_vars = zip(self.v.trainable_variables, self.v_target.trainable_variables)
        polyaked_vars = [self.polyak * v_targ + (1 - self.polyak) * v_main for v_main, v_targ in zip_vars]
        self.v_target.set_weights(polyaked_vars)
        








