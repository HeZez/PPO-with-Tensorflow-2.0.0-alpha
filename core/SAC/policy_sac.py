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
        self.polyak = 0.995
        
        self.pi = pi_gaussian_model(self.hidden_sizes_pi, self.env_info)
        self.q1 = vf_model(self.hidden_sizes_v, self.env_info)
        self.q2 = vf_model(self.hidden_sizes_v, self.env_info)
        self.v = vf_model(self.hidden_sizes_v, self.env_info)
        self.v_target = vf_model(self.hidden_sizes_v, self.env_info)

        self.target_update()

        #self.target_init()

    # @tf.function not working here
    def update(self, obs, obs_next, actions, returns, dones):
        '''
        Update 
        '''
        pi_loss = self.train_pi_one_step(obs)
        q1_loss, q2_loss = self.train_values_one_step(obs, obs_next, actions, returns, dones)
        v_loss = self.train_v_one_step(obs)
        self.target_update()
        
        # Return Metrics
        return pi_loss, q1_loss, q2_loss, v_loss
        
        
    def _mean_squared_error(self, targets, values):
        loss = 0.5 * tf.reduce_mean(tf.square(targets - values))
        return loss 


    def _pi_loss(self, mu, log_std, obs):
        mu, pi, logp_pi = self.pi.get_mu_pi_logp_pi_2(mu, log_std)
        q1_pi = self.q1(tf.concat([obs,pi], axis = -1))
        pi_loss = tf.reduce_mean(self.alpha * logp_pi - q1_pi)
        return pi_loss

    def train_pi_one_step(self, obs):
        with tf.GradientTape() as tape:
            mu, log_std= self.pi(obs)
            pi_loss = self._pi_loss(mu, log_std, obs)
        grads = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.optimizer_pi.apply_gradients(zip(grads, self.pi.trainable_variables))
        return pi_loss

    def train_values_one_step(self, obs, obs_next, act, returns, dones):
        # Off policy samples
        q_backup = tf.stop_gradient(returns + 0.99*(1-dones)*self.v_target.predict(obs_next))

        with tf.GradientTape() as tape:
            q1 = self.q1(tf.concat([obs,act], axis = -1))
            q1_loss = self._mean_squared_error(q_backup, q1)
            # q2 = self.q2(tf.concat([obs,act], axis = -1))
            # q2_loss = self._mean_squared_error(q_backup, q2)

        grads = tape.gradient(q1_loss, self.q1.trainable_variables)
        self.optimizer_q1.apply_gradients(zip(grads, self.q1.trainable_variables))

        with tf.GradientTape() as tape:
            # q1 = self.q1(tf.concat([obs,act], axis = -1))
            # q1_loss = self._mean_squared_error(q_backup, q1)
            q2 = self.q2(tf.concat([obs,act], axis = -1))
            q2_loss = self._mean_squared_error(q_backup, q2)

        grads = tape.gradient(q2_loss, self.q2.trainable_variables)
        self.optimizer_q2.apply_gradients(zip(grads, self.q2.trainable_variables))

        return q1_loss, q2_loss
    
    # def train_q2_one_step(self, obs, obs_next, act, returns, dones):
    #     q_backup = tf.stop_gradient(returns + 0.99*(1-dones)*self.v_target.predict(obs_next))

    #     with tf.GradientTape() as tape:
    #         values = self.q2(tf.concat([obs,act], axis = -1))
    #         q2_loss = self._mean_squared_error(q_backup, values)
    #     grads = tape.gradient(q2_loss, self.q1.trainable_variables)
    #     self.optimizer_q2.apply_gradients(zip(grads, self.q2.trainable_variables))
    #     return q2_loss

    
    def train_v_one_step(self, obs):
        # On Policy samples
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
        

        #self.v_target = tf.keras.models.clone_model(self.v)
        self.v_target.set_weights(self.v.get_weights())
        #target_update = self.v_target.assign(self.polyak * self.v_target +(1- self.polyak)*self.v)
        #tf.group([tf.assign(v_targ, self.polyak * v_targ + (1-self.polyak)*v_main) for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
        #return target_update


    # def target_init(self):
        
    #     target_init = self.v_target.assign(self.v)
    #     # tf.group([tf..assign(v_targ, v_main) for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])
    #     return target_init







