import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl


EPS = 1e-8

class ProbabilityDistribution(tf.keras.Model):

    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
    

class convolutional_net(tf.keras.Model):

    def __init__(self):

        super().__init__('convolutionapassl_net')

        self.model = tf.keras.models.Sequential()
        self.model.add(kl.Conv2D(32, (3, 3), activation='relu'))
        self.model.add(kl.MaxPooling2D((2, 2)))
        self.model.add(kl.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(kl.MaxPooling2D((2, 2)))
        self.model.add(kl.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(kl.Flatten())

        
    def call(self, inputs):

        tensor_input = tf.convert_to_tensor(inputs)
        output = self.model(tensor_input)
        return output

class pi_model_with_conv(tf.keras.Model):

    def __init__(self, hidden_sizes_pi=(32, 32), num_actions=None):
        
        super().__init__('pi_with_conv')
        self.conv = convolutional_net()
        self.pi = pi_model(hidden_sizes_pi, num_actions)

    def call(self, inputs):
        conv_output = self.conv.predict(inputs)
        logits = self.pi.predict(conv_output)
        return logits

    def get_action_logp(self,obs):
        conv_output = self.conv.predict(obs)
        action, logp_t = self.pi.get_action_logp(conv_output)
        return tf.squeeze(action, axis=-1), np.squeeze(logp_t, axis=-1)

        

class pi_model(tf.keras.Model):

    def __init__(self, hidden_sizes_pi=(32, 32), num_actions=None):

        super().__init__('pi_model')

        self.num_actions = num_actions
        self.hidden_pi_layers = tf.keras.Sequential([kl.Dense(h, activation='relu') for h in hidden_sizes_pi])
        self.logits = kl.Dense(num_actions, name='policy_logits')
        self.dist = ProbabilityDistribution()

    def call(self, inputs):

        tensor_input = tf.convert_to_tensor(inputs)
        hidden_logs = self.hidden_pi_layers(tensor_input)
        return self.logits(hidden_logs)
    
    def get_action_logp(self, obs):

        logits = self.predict(obs)
        logp_all = tf.nn.log_softmax(logits)
        action = self.dist.predict(logits)
        logp_t = tf.reduce_sum(tf.one_hot(action, depth=self.num_actions) * logp_all, axis=1)
        return tf.squeeze(action, axis=-1), np.squeeze(logp_t, axis=-1)



class pi_gaussian_model(tf.keras.Model):

    def __init__(self, hidden_sizes=(32,32), activation='relu', num_outputs=None):

        super().__init__('pi_gaussian_model')

        self.num_outputs = num_outputs
        self.hidden_layers = tf.keras.Sequential([kl.Dense(h, activation= activation) for h in hidden_sizes])
        self.mu = kl.Dense(num_outputs, name='policy_mu')
        
        
    def call(self, inputs):

        tensor_input = tf.convert_to_tensor(inputs)
        hidden = self.hidden_layers(tensor_input)
        mu = self.mu(hidden)
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
        log_std = tf.Variable(name= 'log_std', initial_value= -0.5 * np.ones(self.num_outputs, dtype=np.float32))
        std = tf.exp(log_std)
        # sample action
        action = mu + tf.random.normal(tf.shape(mu)) * std
        # calculate logp_old
        logp_t = self.gaussian_likelihood(action, mu, log_std)

        return np.squeeze(action, axis=-1), np.squeeze(logp_t, axis=-1)


    def gaussian_likelihood(self, x, mu, log_std):

        '''
        calculate the liklihood of a gaussian distribution for parameters x given the variables mu and log_std
        '''
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2 * log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)



class v_model(tf.keras.Model):

    def __init__(self, hidden_sizes_v=(32,32)):

        super().__init__('v_model')

        self.hidden_v_layers = tf.keras.Sequential([kl.Dense(h, activation='relu') for h in hidden_sizes_v])
        self.value= kl.Dense(1, name='value')

    def call(self, inputs):

        tensor_input = tf.convert_to_tensor(inputs)
        hidden_vals = self.hidden_v_layers(tensor_input)
        return  self.value(hidden_vals)
    
    def get_value(self, obs):

        value = self.predict(obs)
        return  np.squeeze(value, axis=-1)


