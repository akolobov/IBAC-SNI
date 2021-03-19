import numpy as np
import tensorflow.keras as ks
import tensorflow.keras.backend as K

def get_activation(tns=None, activation='relu'):
    '''
    Adds an activation layer to a graph.

    Args :
        tns :
            *Keras tensor or None*

            Input tensor. If not None, then the graph will be connected through
            it, and a tensor will be returned. If None, the activation layer
            will be returned.
        activation :
            *str, optional (default='relu')*

            The name of an activation function.
            One of 'relu', 'leakyrelu', 'prelu', 'elu', 'mrelu' or 'swish',
            or anything that Keras will recognize as an activation function
            name.

    Returns :
        *Keras tensor or layer instance* (see tns argument)
    '''

    if activation == 'relu':
        act = ks.layers.ReLU()

    elif activation == 'leakyrelu':
        act = ks.layers.LeakyReLU()

    elif activation == 'prelu':
        act = ks.layers.PReLU()

    elif activation == 'elu':
        act = ks.layers.ELU()

    elif activation == 'swish':
        def swish(x):
            return K.sigmoid(x) * x
        act = ks.layers.Activation(swish)

    elif activation == 'mrelu':
        def mrelu(x):
            return K.minimum(K.maximum(1-x, 0), K.maximum(1+x, 0))
        act = ks.layers.Activation(mrelu)

    elif activation == 'gaussian':
        def gaussian(x):
            return K.exp(-x**2)
        act = ks.layers.Activation(gaussian)

    elif activation == 'flipped_gaussian':
        def flipped_gaussian(x):
            return 1 - K.exp(-x**2)
        act = ks.layers.Activation(flipped_gaussian)

    else:
        act = ks.layers.Activation(activation)

    if tns is not None:
        return act(tns)
    else:
        return act


class FiLM(ks.layers.Layer):

    def __init__(self, widths=[64,64], activation='leakyrelu',
                 initialization='glorot_uniform', **kwargs):
        self.widths = widths
        self.activation = activation
        self.initialization = initialization
        super(FiLM, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_vars_shape = input_shape
        self.n_feature_maps = feature_map_shape[-1]
        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]

        # Collect trainable weights
        trainable_weights = []

        # Create weights for hidden layers
        self.hidden_dense_layers = []
        for i,width in enumerate(self.widths):
            dense = ks.layers.Dense(width,
                                    kernel_initializer=self.initialization,
                                    name=f'FiLM_dense_{i}')
            if i==0:
                build_shape = FiLM_vars_shape[:2]
            else:
                build_shape = (None,self.widths[i-1])
            dense.build(build_shape)
            trainable_weights += dense.trainable_weights
            self.hidden_dense_layers.append(dense)

        # Create weights for output layer
        self.output_dense = ks.layers.Dense(2 * self.n_feature_maps, # assumes channel_last
                                            kernel_initializer=self.initialization,
                                            name=f'FiLM_dense_output')
        self.output_dense.build((None,self.widths[-1]))
        trainable_weights += self.output_dense.trainable_weights

        # Pass on all collected trainable weights
        self._trainable_weights = trainable_weights

        super(FiLM, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        conv_output, FiLM_vars = x

        # Generate FiLM outputs
        tns = FiLM_vars
        for i in range(len(self.widths)):
            tns = self.hidden_dense_layers[i](tns)
            tns = get_activation(activation=self.activation)(tns)
            
        FiLM_output = self.output_dense(tns)

        # Duplicate in order to apply to entire feature maps
        # Taken from https://github.com/GuessWhatGame/neural_toolbox/blob/master/film_layer.py
        FiLM_output = K.expand_dims(FiLM_output, axis=[1])
        FiLM_output = K.expand_dims(FiLM_output, axis=[1])
        FiLM_output = K.tile(FiLM_output, [1, self.height, self.width, 1])

        # Split into gammas and betas
        gammas = FiLM_output[:, :, :, :self.n_feature_maps]
        betas = FiLM_output[:, :, :, self.n_feature_maps:]

        # Apply affine transformation
        return (1 + gammas) * conv_output + betas

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]