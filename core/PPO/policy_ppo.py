import tensorflow as tf
import numpy as np
from core.PPO.models_ppo import pi_categorical_model, pi_gaussian_model, v_model, FwdDyn
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
        self.fwd_dyn = FwdDyn(env_info= self.env_info)

    # @tf.function not working here
    def update(self, observations, actions, advs, returns, logp_t): #, io, ia, ion):
        '''
        Update the Policy Gradient and the Value Network
        '''

        # next_obs = ion # observations[1:]
        # obs = io # observations[:-1]
        # acts = ia # actions[:-1]

        for i in range(self.train_pi_iters):
            loss_pi, loss_entropy, approx_ent, kl = self.train_pi_one_step(observations, actions, advs, logp_t)
            if kl > 1.5 * self.target_kl:
                log("Early stopping at step %d due to reaching max kl." %i)
                break

        for _ in range(self.train_v_iters):
            loss_v = self.train_v_one_step(observations, returns)
        
        # for _ in range(80):
        #     loss_dyn = self.train_fwd_dyn_one_step(obs, acts, next_obs)
            
        # Return Metrics
        return loss_pi, loss_entropy, approx_ent, kl, loss_v
        
    def _dyn_loss(self,next_obs, next_obs_preds):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(next_obs-next_obs_preds), axis=1))

    # @tf.function
    def train_fwd_dyn_one_step(self, obs, actions, next_obs):
        a_arr = []
        for a in actions:
            a_arr.append([a])
        inputs = tf.concat((obs,a_arr), axis=-1)

        with tf.GradientTape() as tape:

            next_obs_preds = self.fwd_dyn(inputs)
            dyn_loss = self._dyn_loss(next_obs, next_obs_preds)

        grads = tape.gradient(dyn_loss, self.fwd_dyn.trainable_variables)
        # grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
        self.optimizer_fwd_dyn.apply_gradients(zip(grads, self.fwd_dyn.trainable_variables))

        return dyn_loss
        
    def _value_loss(self, returns, values):
        # Mean Squared Error
        # dif = returns - values
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

        if isinstance(self.env_info, Discrete):
            entropy = self.pi.entropy(logits_or_mu)
        else:
            entropy = self.pi.entropy()

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

    









