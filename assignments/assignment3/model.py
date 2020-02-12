import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """    
        # TODO Create necessary layers
        conv_kernel = 3
        maxpool_kernel = 4
        
        self.layers = [ConvolutionalLayer(input_shape[2], conv1_channels,\
                                        conv_kernel,  int(np.floor((conv_kernel - 1) / 2))), 
                        ReLULayer(),                  
                        MaxPoolingLayer(maxpool_kernel, stride=maxpool_kernel),
                        ConvolutionalLayer(conv1_channels, conv2_channels,\
                                        conv_kernel,  int(np.floor((conv_kernel - 1) / 2))),
                        ReLULayer(),
                        MaxPoolingLayer(maxpool_kernel, stride=maxpool_kernel),
                        Flattener(),
#                        ReLULayer(),
                        FullyConnectedLayer(int(input_shape[0]/(maxpool_kernel**2)\
                                            * input_shape[1]/(maxpool_kernel**2)\
                                            * conv2_channels), n_output_classes)]       

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        parameters = self.params()
        for param in parameters.values():
            param.grad = np.zeros_like(param.grad)
        # TODO Compute loss and fill param gradients
        data = X
        for i in range(len(self.layers)):
            data = self.layers[i].forward(data)
            
        loss, dprediction = softmax_with_cross_entropy(data, y.copy())
        
        gradients = dprediction
        for i in reversed(range(len(self.layers))):
            gradients = self.layers[i].backward(gradients)
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
#        reg = 1e-6
#        for param_key, param in parameters.items():
#            reg_loss, reg_grad = l2_regularization(param.value, reg)
#            loss += reg_loss
#            #print(param_key, np.mean(np.square(param.grad)), np.mean(np.square(reg_grad)))
#            param.grad += reg_grad
#        
        return loss

    def predict(self, X):
        data = X
        for i in range(len(self.layers)):
            data = self.layers[i].forward(data)
        # classes predictions on the set
        y_pred = np.argmax(data, axis=1)

        return y_pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        layers_with_params = [self.layers[0], self.layers[3], self.layers[7]]
        
        for i in range(3):
            parameters = layers_with_params[i].params()
            for param_key in parameters:
                result[param_key + '_%i' %i] = parameters[param_key]

        return result
