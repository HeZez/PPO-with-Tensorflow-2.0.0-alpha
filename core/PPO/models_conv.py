import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl

EPS = 1e-8

class ProbabilityDistribution(tf.keras.Model):

    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
    

def conv_model_sequential_API(shape=(84,84,3), activation='relu'):
    '''
        filters = feature maps where each has an dimensionality d x d 

        kernel_size = kernel matrix m x m sliding over input image

        strides =   step size with which moving the kernel matrix over the input image ==> 
                    increase size of strides to reduce output dimensionality d x d of feature maps

        max pool = scale down size of input from convolution layer

        always go from less to more feature maps

        then flatten layers and connect do mlp for classification

    '''
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape= shape))
    model.add(kl.Conv2D(filters = 32, kernel_size = (5, 5), strides = 4, padding='valid', activation= activation))
    model.add(kl.Conv2D(filters = 64, kernel_size = (3, 3), strides = 4, padding='valid', activation= activation))
    model.add(kl.Flatten())

    return model

def conv_model_functional_API(shape=(84,84,3), activation='elu'):

    inputs = tf.keras.Input(shape=shape)
    x = kl.Conv2D(filters= 32, kernel_size= (5, 5), strides= 1, padding= 'valid', activation= activation, name='Functional_API')(inputs)
    x = kl.MaxPooling2D((2, 2))(x)
    x = kl.Conv2D(filters= 64, kernel_size= (3, 3), strides= 2, padding= 'valid', activation= activation)(x)
    x = kl.MaxPooling2D((2, 2))(x)
    outputs = kl.Flatten()(x)

    model = tf.keras.Model(inputs= inputs, outputs= outputs)

    return model

def conv_model(shape=(84,84,3), activation='elu'):

    conv1 = kl.Conv2D(filters= 32, kernel_size= (5,5), strides= 1, padding= 'valid', activation= activation, input_shape= (84,84,3))
    max_pool_1 = kl.MaxPooling2D((2,2))
    conv2 = kl.Conv2D(filters= 64, kernel_size= (3,3), strides= 2, padding= 'valid', activation= activation)
    max_pool_2 = kl.MaxPooling2D((2,2))
    flat = kl.Flatten()
    
    return conv1, max_pool_1, conv2, max_pool_2, flat


class pi_model_with_conv(tf.keras.Model):

    def __init__(self, hidden_sizes_pi= (32,32), num_actions=None):
        
        super().__init__('pi_with_conv')
        
        self.num_actions = num_actions

        self.model = conv_model_functional_API()
        
        self.hidden_pi_layers = tf.keras.Sequential([kl.Dense(h, activation='relu') for h in hidden_sizes_pi])
        self.logits = kl.Dense(num_actions, name='policy_logits')

        self.dist = ProbabilityDistribution()

    @tf.function
    def call(self, inputs):
        
        tensor_input = tf.convert_to_tensor(inputs)

        x = self.model(tensor_input)

        hidden_logs = self.hidden_pi_layers(x)
        logits = self.logits(hidden_logs)

        return logits

    #@tf.function
    def get_action_logp(self,obs):

        logits = self.predict(obs)
        logp_all = tf.nn.log_softmax(logits)
        action = self.dist.predict(logits)
        logp_t = tf.reduce_sum(tf.one_hot(action, depth=self.num_actions) * logp_all, axis=1)
        return tf.squeeze(action, axis=-1), np.squeeze(logp_t, axis=-1)



class v_model_with_conv(tf.keras.Model):

    def __init__(self, hidden_sizes_v= (32,32)):

        super().__init__('v_with_conv')

        # self.conv1, self.max_pool_1, self.conv2,self.max_pool_2, self.flat = conv_model()
        self.model = conv_model_functional_API()

        self.hidden_v_layers = tf.keras.Sequential([kl.Dense(h, activation='relu') for h in hidden_sizes_v])
        self.value= kl.Dense(1, name='value')

    @tf.function
    def call(self, inputs):

        tensor_input = tf.convert_to_tensor(inputs)

        # x = self.conv1(tensor_input)
        # x = self.max_pool_1(x)
        # x = self.conv2(x)
        # x = self.max_pool_2(x)
        # x = self.flat(x)

        x = self.model(tensor_input)

        hidden_vals = self.hidden_v_layers(x)
        vals = self.value(hidden_vals)

        return vals
    
    #@tf.function
    def get_value(self, obs):

        value = self.predict(obs)
        return  np.squeeze(value, axis=-1)
        





