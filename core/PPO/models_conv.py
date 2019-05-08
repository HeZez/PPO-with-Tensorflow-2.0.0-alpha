import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as kl

EPS = 1e-8

class ProbabilityDistribution(tf.keras.Model):

    def call(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
    

def conv_model_1(shape=(84,84,3), activation='relu'):
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

def conv_model_2(shape=(84,84,3), activation='relu'):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=shape))
    model.add(kl.Conv2D(32, (3, 3), activation=activation))
    model.add(kl.MaxPooling2D((2, 2)))
    model.add(kl.Conv2D(64, (3, 3), activation=activation))
    model.add(kl.MaxPooling2D((2, 2)))
    model.add(kl.Conv2D(64, (3, 3), activation=activation))
    model.add(kl.Flatten())
    return model

def conv_model(shape=(84,84,3), activation='elu'):

    conv1 = kl.Conv2D(filters= 32, kernel_size= (5,5), strides= 4, padding= 'valid', activation= activation, input_shape= (84,84,3))
    conv2 = kl.Conv2D(filters= 64, kernel_size= (3,3), strides= 4, padding= 'valid', activation= activation)
    flat = kl.Flatten()
    
    return conv1, conv2, flat


class pi_model_with_conv(tf.keras.Model):

    def __init__(self, hidden_sizes_pi= (32,32), num_actions=None):
        
        super().__init__('pi_with_conv')
        
        self.num_actions = num_actions
        
        # self.conv1 = kl.Conv2D(32, input_shape=(84,84,3), kernel_size=(5,5),strides= 4,padding='valid', activation='elu')
        # self.conv2 = kl.Conv2D(64, kernel_size=(3,3),strides= 4,padding='valid', activation='elu')
        # self.flat = kl.Flatten()

        self.conv1, self.conv2, self.flat = conv_model()
        
        self.hidden_pi_layers = tf.keras.Sequential([kl.Dense(h, activation='relu') for h in hidden_sizes_pi])
        self.logits = kl.Dense(num_actions, name='policy_logits')

        self.dist = ProbabilityDistribution()

    def call(self, inputs):
        
        tensor_input = tf.convert_to_tensor(inputs)

        x = self.conv1(tensor_input)
        x = self.conv2(x)
        x = self.flat(x)

        hidden_logs = self.hidden_pi_layers(x)
        logits = self.logits(hidden_logs)

        return logits

    def get_action_logp(self,obs):

        logits = self.predict(obs)
        logp_all = tf.nn.log_softmax(logits)
        action = self.dist.predict(logits)
        logp_t = tf.reduce_sum(tf.one_hot(action, depth=self.num_actions) * logp_all, axis=1)
        return tf.squeeze(action, axis=-1), np.squeeze(logp_t, axis=-1)



class v_model_with_conv(tf.keras.Model):

    def __init__(self, hidden_sizes_v= (32,32)):

        super().__init__('v_with_conv')

        # self.conv1 = kl.Conv2D(32,input_shape=(84,84,3), kernel_size=(5,5),strides= 4,padding='valid', activation='elu')
        # self.conv2 = kl.Conv2D(64, kernel_size=(3,3),strides= 4,padding='valid', activation='elu')
        # self.flat = kl.Flatten()

        self.conv1, self.conv2, self.flat = conv_model()

        self.hidden_v_layers = tf.keras.Sequential([kl.Dense(h, activation='relu') for h in hidden_sizes_v])
        self.value= kl.Dense(1, name='value')

    def call(self, inputs):

        tensor_input = tf.convert_to_tensor(inputs)
        
        x = self.conv1(tensor_input)
        x = self.conv2(x)
        x = self.flat(x)

        hidden_vals = self.hidden_v_layers(x)
        vals = self.value(hidden_vals)

        return vals
    
    def get_value(self, obs):

        value = self.predict(obs)
        return  np.squeeze(value, axis=-1)
        





