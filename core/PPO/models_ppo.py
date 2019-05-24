import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as layer
from core.Env import Discrete, Continuous

EPS = 1e-8

def lstm_model(shape= None, model =None):
    '''
    lstm model ????
    '''
    input_sequences = tf.keras.Input(shape=(4,) + shape)
    # returns sequence of 4 (Timesteps) of size outputs flattened from conv model
    lstm = tf.keras.layers.TimeDistributed(model)(input_sequences)
    # returns the last output of the lstm layer with output size 32
    lstm = layer.LSTM(32, return_sequences=False)(lstm) 
    lstm_model = tf.keras.Model(inputs= input_sequences, outputs= lstm)
    return lstm_model


def conv_model_functional_API(shape=(84,84,3), activation='elu', use_lstm= True):
    '''
    filters = feature maps where each has an dimensionality d x d 

    kernel_size = kernel matrix m x m sliding over input image

    strides =   step size with which moving the kernel matrix over the input image ==> 
                increase size of strides to reduce output dimensionality d x d of feature maps

    max pool = scale down size of input from convolution layer

    always go from less to more feature maps

    then flatten layers and connect do mlp for classification
    '''
    inputs = tf.keras.Input(shape= shape)
    x = layer.Conv2D(filters= 32, kernel_size= (5, 5), strides= 1, padding= 'valid', activation= activation, name='Functional_API')(inputs)
    x = layer.MaxPooling2D((2, 2))(x)
    x = layer.Conv2D(filters= 64, kernel_size= (3, 3), strides= 2, padding= 'valid', activation= activation)(x)
    x = layer.MaxPooling2D((2, 2))(x)
    outputs = layer.Flatten()(x)
    model = tf.keras.Model(inputs= inputs, outputs= outputs)

    # if use_lstm:
    #     lstm_mod = lstm_model(shape, model)
    #     return lstm_mod
    # else:
    #     return model
    
    return model


class ProbabilityDistribution(tf.keras.Model):

    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

    
class pi_categorical_model(tf.keras.Model):
    '''
    logits --> log of an odd
    odd = p/1-p --> chance of an event to ocure
    logp --> log probabilities = softmax of logits
    '''

    def __init__(self, hidden_sizes_pi= (32, 32), env_info= Discrete, activation='relu'):

        super().__init__('pi_categorical_model')

        self.env_info = env_info
        self.num_actions = self.env_info.act_size

        if self.env_info.is_visual:
            self.model = conv_model_functional_API(shape= self.env_info.obs_shape, use_lstm= self.env_info.is_frame_stacking) 

        self.hidden_pi_layers = tf.keras.Sequential([layer.Dense(h, activation= activation) for h in hidden_sizes_pi])
        self.logits = layer.Dense(self.num_actions, name='policy_logits')
        
        self.dist = ProbabilityDistribution()

    @tf.function
    def call(self, inputs):

        x = tf.convert_to_tensor(inputs)

        if self.env_info.is_visual:
            x = self.model(x)

        hidden_logs = self.hidden_pi_layers(x)
        logits = self.logits(hidden_logs)

        return logits
    
    def get_action_logp(self, obs):

        if self.env_info.is_frame_stacking:
            obs = obs[None, :, :, :] #  :]
        logits = self.predict(obs)
        action = self.dist.predict(logits)
        logp_t = self.logp(logits, action) 

        return tf.squeeze(action, axis=-1), np.squeeze(logp_t, axis=-1)

    def logp(self, logits, action):

        logp_all = tf.nn.log_softmax(logits)
        logp = tf.reduce_sum(tf.one_hot(action, depth= self.num_actions) * logp_all, axis= 1)

        return logp

    def entropy(self, logits= None):
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

    def entropy_cat_with_keras(self, logits= None):
        '''
        Entropy calc with keras over logits ??
        # calculated with Cross Entropy over logits with itself ??
        '''
        entropy = tf.keras.losses.categorical_crossentropy(logits, logits, from_logits=True)
        return entropy

    def entropy_cat_with_softmax(self, p0 = None):
        return -tf.reduce_sum(p0 * tf.math.log(p0 + 1e-6), axis = 1)



class v_model(tf.keras.Model):

    def __init__(self, hidden_sizes_v= (32,32), env_info= Discrete):

        super().__init__('v_model')

        self.env_info= env_info

        if self.env_info.is_visual:
            self.model = conv_model_functional_API(shape= self.env_info.obs_shape, use_lstm= self.env_info.is_frame_stacking)

        self.hidden_v_layers = tf.keras.Sequential([layer.Dense(h, activation='relu') for h in hidden_sizes_v])
        self.value= layer.Dense(1, name='values')

    @tf.function
    def call(self, inputs):

        x = tf.convert_to_tensor(inputs)

        if self.env_info.is_visual:
            x = self.model(x)

        hidden_vals = self.hidden_v_layers(x)
        values = self.value(hidden_vals)

        return  tf.squeeze (values, axis=-1)
    
    def get_value(self, obs):
        # for Frame Stacking
        if self.env_info.is_frame_stacking:
            obs = obs[None, :, :, :] # :]
        value = self.predict(obs)
        return  np.squeeze(value, axis=-1)



class pi_gaussian_model(tf.keras.Model):

    def __init__(self, hidden_sizes=(32,32), env_info= Continuous, activation='relu'):

        super().__init__('pi_gaussian_model')

        self.env_info = env_info
        self.num_outputs = self.env_info.act_size

        if self.env_info.is_visual:
            self.model = conv_model_functional_API(shape= self.env_info.obs_shape)

        self.hidden_layers = tf.keras.Sequential([layer.Dense(h, activation= activation) for h in hidden_sizes])
        self.mu = layer.Dense(self.num_outputs, name='policy_mu')

        # std deviation is a trainable variable and is updated by the pi optimizer
        self.log_std = tf.Variable(name= 'log_std', initial_value= -0.5 * np.ones(self.num_outputs, dtype=np.float32))
        
    
    @tf.function
    def call(self, inputs):

        x = tf.convert_to_tensor(inputs)

        if self.env_info.is_visual:
            x = self.model(x)

        hidden_mu = self.hidden_layers(x)
        mu = self.mu(hidden_mu)
        return mu
    
    def get_action_logp(self, obs):

        '''
        Get Action and logarithmic probability on action at Environment-Step t
        
        Approximate mu from a Neural Network
        
        Model a Gaussian distrubution with mu and standard deviation (std) where action 
        is sampled from a Random Normal distribution which is mu + random_normal * std

        Last calculate log_p at step t which is logp_old for PPO Update --> For Importance Sampling and kl approx  
        '''

        # mu
        mu = self.predict(obs)
        # std deviation
        std = tf.exp(self.log_std)
        # sample action
        action = mu + tf.random.normal(tf.shape(mu)) * std
        # clip actions in range of -1,1 
        action = tf.clip_by_value(action, -1, 1)
        # calculate logp_old
        logp_t = self.logp(action, mu)

        return np.squeeze(action, axis=-1), np.squeeze(logp_t, axis=-1)

    def logp(self, mu, action):
        return self.gaussian_likelihood(action, mu, self.log_std)

    def gaussian_likelihood(self, x, mu, log_std):
        '''
        calculate the liklihood logp of a gaussian distribution for parameters x given the variables mu and log_std
        '''
        pre_sum = -0.5 * (((x-mu) / (tf.exp(log_std)+EPS))**2 + 2 * log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def entropy(self):
        '''
        Entropy term for more randomness which means more exploration in ppo -> 
        '''
        entropy = tf.reduce_sum(self.log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return entropy