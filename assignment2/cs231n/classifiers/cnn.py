from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        C, H, W = input_dim
        
        # Calculate output shape of the first conv layer.
        # According to conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}        
        conv1_stride = 1
        conv1_pad = (filter_size - 1) // 2
        conv1_H_out = int(1 + (H + 2 * conv1_pad - filter_size) / conv1_stride)
        conv1_W_out = int(1 + (W + 2 * conv1_pad - filter_size) / conv1_stride)
        
        # Calculate output shape of the first pool layer.
        # pass pool_param to the forward pass for the max-pooling layer
        # pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        pool1_H_out = int((conv1_H_out - 2 + 0)/2 + 1)
        pool1_W_out = int((conv1_W_out - 2 + 0)/2 + 1)
        
        self.params.update({
            'W1': np.random.normal(scale=weight_scale, size=(num_filters, C, filter_size, filter_size)),
            'b1': np.zeros((num_filters, )),
            'W2': np.random.normal(scale=weight_scale, size=(pool1_H_out * pool1_W_out * num_filters, hidden_dim)),
            'b2': np.zeros((hidden_dim, )),
            'W3': np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes)),
            'b3': np.zeros((num_classes, ))
        })
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None, use_fast=True):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        N, C, H, W = X.shape
        
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        # First layer.
        if use_fast:
            layer1_conv_out, layer1_conv_cache = conv_forward_fast(X, W1, b1, conv_param)
            layer1_relu_out, layer1_relu_cache = relu_forward(layer1_conv_out)
            layer1_max_pool_out, layer1_max_pool_cache = max_pool_forward_fast(layer1_relu_out, pool_param)
        else:
            layer1_conv_out, layer1_conv_cache = conv_forward_naive(X, W1, b1, conv_param)
            layer1_relu_out, layer1_relu_cache = relu_forward(layer1_conv_out)
            layer1_max_pool_out, layer1_max_pool_cache = max_pool_forward_naive(layer1_relu_out, pool_param)

        # Second layer.
        layer2_affine_input = layer1_max_pool_out.reshape(N, -1)
        layer2_affine_out, layer2_affine_cache = affine_forward(layer2_affine_input, W2, b2)
        layer2_relu_out, layer2_relu_cache = relu_forward(layer2_affine_out)

        # Third layer.
        layer3_affine_out, layer3_affine_cache = affine_forward(layer2_relu_out, W3, b3)
        
        scores = layer3_affine_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        # L2 regularization.
        l2_reg = self.reg * (
            np.sum(np.square(self.params['W1'])) + \
            np.sum(np.square(self.params['W2'])) + \
            np.sum(np.square(self.params['W3']))
        ) * 0.5
        
        loss, dlayer3_affine_out = softmax_loss(layer3_affine_out, y)
        loss += l2_reg
        
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        # Gradient of second layer.
        dlayer2_relu_out, dW3, db3 = affine_backward(dlayer3_affine_out, layer3_affine_cache)
        dlayer2_affine_out = relu_backward(dlayer2_relu_out, layer2_relu_cache)
        dlayer2_affine_input, dW2, db2 =  affine_backward(dlayer2_affine_out, layer2_affine_cache)
        
        dlayer2_affine_input = dlayer2_affine_input.reshape(layer1_max_pool_out.shape)
        
        # Gradient of first layer.
        if use_fast:
            dlayer1_max_pool_out = max_pool_backward_fast(dlayer2_affine_input, layer1_max_pool_cache)
            dlayer1_relu_out = relu_backward(dlayer1_max_pool_out, layer1_relu_cache)
            dlayer1_conv_out, dW1, db1 = conv_backward_fast(dlayer1_relu_out, layer1_conv_cache)
        else:
            dlayer1_max_pool_out = max_pool_backward_naive(dlayer2_affine_input, layer1_max_pool_cache)
            dlayer1_relu_out = relu_backward(dlayer1_max_pool_out, layer1_relu_cache)
            dlayer1_conv_out, dW1, db1 = conv_backward_naive(dlayer1_relu_out, layer1_conv_cache)
            
        grads.update({
            'W1': dW1 + 2 * self.reg * self.params['W1'] * 0.5, 
            'b1': db1, 
            'W2': dW2 + 2 * self.reg * self.params['W2'] * 0.5, 
            'b2': db2,
            'W3': dW3 + 2 * self.reg * self.params['W3'] * 0.5, 
            'b3': db3
        })
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
