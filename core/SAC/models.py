import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layer
from core.Env import Discrete, Continuous

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-8

class pi_gaussian_model(tf.keras.Model):

    def __init__(self, hidden_sizes=(32,32), env_info= Continuous, activation='relu'):

        super().__init__('pi_gaussian_model')

        self.env_info = env_info
        self.num_outputs = self.env_info.act_size

        self.hidden_layers = tf.keras.Sequential([layer.Dense(h, activation= activation) for h in hidden_sizes])

        self.mu = layer.Dense(self.num_outputs, name='policy_mu')
        self.log_std = layer.Dense(self.num_outputs, name='policy_log_std')
        
    
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden_layers = self.hidden_layers(x)
        mu = self.mu(hidden_layers)
        log_std = self.log_std(hidden_layers)
        return mu, log_std
    
    

    def get_mu_pi_logp_pi(self, obs):
        # mu
        mu, log_std = self.predict(obs)
        # std deviation
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = tf.exp(log_std)
        # sample action
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        # clip actions in range of -1,1 
        # action = tf.clip_by_value(action, -1, 1)
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)

        mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)

        # Action scale

        return np.squeeze(mu, axis=-1), np.squeeze(pi, axis=-1), np.squeeze(logp_pi, axis=-1)

    def get_mu_pi_logp_pi_2(self, mu, log_std):
        # std deviation
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = tf.exp(log_std)
        # sample action
        pi = mu + tf.random.normal(tf.shape(mu)) * std
        # clip actions in range of -1,1 
        # action = tf.clip_by_value(action, -1, 1)
        logp_pi = self.gaussian_likelihood(pi, mu, log_std)

        mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)

        # Action scale

        return mu, pi, logp_pi


    def get_action(self, obs, deterministic=False):
        mu, pi, _ = self.get_mu_pi_logp_pi(obs)
        return mu if deterministic else pi


    def gaussian_likelihood(self, x, mu, log_std):
        '''
        calculate the liklihood logp of a gaussian distribution for parameters x given the variables mu and log_std
        '''
        pre_sum = -0.5 * (((x-mu) / (tf.exp(log_std)+ EPS))**2 + 2 * log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)
    

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


    def apply_squashing_func(self, mu, pi, logp_pi):
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= tf.reduce_sum(tf.math.log(self.clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
        return mu, pi, logp_pi



class vf_model(tf.keras.Model):

    def __init__(self, hidden_sizes_v= (32,32), env_info= Discrete):

        super().__init__('vf_model')

        self.env_info= env_info
        self.hidden_v_layers = tf.keras.Sequential([layer.Dense(h, activation='relu') for h in hidden_sizes_v])
        self.value= layer.Dense(1, name='values')

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        hidden_vals = self.hidden_v_layers(x)
        values = self.value(hidden_vals)
        return  values
    
    def get_value(self, obs):
        value = self.predict(obs)
        return  np.squeeze(value, axis=-1)