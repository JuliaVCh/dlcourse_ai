import numpy as np
from copy import deepcopy

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # implement softmax
    predictions = deepcopy(predictions)
    
    if np.ndim(predictions) == 1:
        predictions -= np.max(predictions)
        predictions[predictions < - 700] = -700
        probs = np.exp(predictions)
        exp_sum = np.sum(probs)
    else:
        predictions = (predictions - np.max(predictions, axis=1, keepdims=True))
        predictions[predictions < - 700] = -700
        probs = np.exp(predictions)
        exp_sum = np.sum(probs, axis=1, keepdims=True)
        assert np.all(exp_sum)
    probs = probs / exp_sum
    
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # implement cross-entropy
    if np.ndim(target_index) == 0:
        loss = -np.log(probs[target_index])
    else:
        row_count = np.size(target_index)
        row_idx = np.arange(row_count)
        likelihood = probs[row_idx, target_index]
        log_likelihood = -np.log(likelihood)
        loss = np.mean(log_likelihood)
    
    return loss

def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = 0.5 * reg_strength * np.sum(np.square(W))
    grad = reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # fix argument dimentions so that they are exactly as expected
    if np.ndim(target_index) > 1:
        target_index = target_index.flatten()
    # implement softmax with cross-entropy
    probs = softmax(predictions.copy())
    loss = cross_entropy_loss(probs, target_index)
    # calculate gradient
    dprediction = probs
    if np.ndim(target_index) == 0:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(np.size(target_index)), target_index] -= 1
        dprediction = dprediction / np.size(target_index)

    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = deepcopy(X)
        self.X[self.X < 0] = 0

        return self.X

    def backward(self, d_out):
        if self.X is None:
            raise Exception("ReLULayer forward pass was not implemented yet!")
        else:
            d_result = deepcopy(d_out)
            d_result[self.X == 0] = 0
        
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        # a = 0.001 does not work at fucking all!
        a = np.sqrt(2 / n_input)
        self.W = Param(np.random.normal(loc=0.0, scale=a, size=(n_input, n_output))) 
        self.B = Param(np.random.normal(loc=0.0, scale=a, size=(1, n_output)))
        self.X = None

    def forward(self, X):
        self.X = deepcopy(X)
        return np.dot(self.X, self.W.value) + self.B.value

    def backward(self, d_out):
        # Compute gradient with respect to input
        if self.X is None:
            raise Exception("FullyConnectedLayer forward pass was not implemented yet!")
        else:
            d_result = np.dot(d_out, self.W.value.T)
        # and gradients with respect to W and B
        dW = np.dot(self.X.T, d_out)
        #dB = np.dot(np.ones((1, d_out.shape[0])), d_out) 
        dB = np.sum(d_out, axis=0, keepdims=True)
        # Add gradients of W and B to their `grad` attribute
        self.W.grad += dW
        self.B.grad += dB

        return d_result

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        a = np.sqrt(1 / (self.in_channels * self.filter_size * self.filter_size))
        self.W = Param(
            np.random.uniform(-a, a, size=(filter_size, filter_size,
                            in_channels, out_channels))
        )
        # казалось бы размеры должны быть такими:
#        self.W = Param(
#            np.random.randn(in_channels, out_channels,
#                            filter_size, filter_size)
#        )

        self.B = Param(np.zeros(out_channels))
#        self.B = Param(np.random.uniform(-a, a, size=(out_channels)))

        self.padding = padding
        
        self.X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # assuming that stride == 1
        out_height = int(2 * self.padding + height - self.filter_size + 1)
        out_width = int(2 * self.padding + width - self.filter_size + 1)
        out = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        W = self.W.value.reshape(-1, self.out_channels)
        self.X = X.copy()
        npad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
#        X_padded = np.pad(self.X, npad, mode='edge')
        X_padded = np.pad(self.X, npad, mode='constant', constant_values=0) 
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                filter_y = slice(y, y + self.filter_size)
                filter_x = slice(x, x + self.filter_size)
                out[:, y, x] = np.dot(X_padded[:, filter_y, filter_x]\
                                .reshape(batch_size, -1), W) + self.B.value
        
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        if self.X is None:
            raise Exception("ConvolutionalLayer forward pass not implemented!")
        else:
            batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        npad = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
#        X_padded = np.pad(self.X, npad, mode='edge')
        X_padded = np.pad(self.X, npad, mode='constant', constant_values=0) 
        d_X = np.pad(np.zeros_like(self.X), npad, mode='constant', constant_values=0)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                filter_y = slice(y, y + self.filter_size)
                filter_x = slice(x, x + self.filter_size)
                # input grad
                d_X[:, filter_y, filter_x] += np.dot(d_out[:, y, x],\
                   self.W.value.reshape(-1, self.out_channels).T)\
                   .reshape(batch_size, self.filter_size, self.filter_size, channels)
                # weights grad
                self.W.grad += np.dot(X_padded[:, filter_y, filter_x]\
                                .reshape(batch_size, -1).T, d_out[:, y, x])\
                                .reshape(self.W.grad.shape)
                # bias grad
                self.B.grad += np.dot(np.ones((1, d_out.shape[0])), d_out[:, y, x]).flatten()

        if self.padding > 0:
            d_X = d_X[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        return d_X

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        out_height = int(np.floor((height - self.pool_size) / self.stride) + 1)
        out_width = int(np.floor((width - self.pool_size) / self.stride) + 1)
        out = np.zeros((batch_size, out_height, out_width, channels))
        self.max_indices = np.zeros_like(out, dtype='int32')
        
        self.X = X.copy()
        
        for y in range(out_height):
            for x in range(out_width):
                filter_y = slice(y * self.stride, y * self.stride + self.pool_size)
                filter_x = slice(x * self.stride, x * self.stride + self.pool_size)
                section = self.X[:, filter_y, filter_x].reshape(batch_size, -1, channels)
                out[:, y, x] = np.amax(section, axis=1)
                # indices of X sectrion max arguments along flattened 2nd and 3rd dimentions
                self.max_indices[:, y, x] = np.argmax(section, axis=1)
        
        return out

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        if self.X is None:
            raise Exception("MaxPoolingLayer forward pass not implemented!")
        else:
            batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
            
        d_X = np.zeros_like(self.X)
        
        for y in range(out_height):
            for x in range(out_width):
                # indices for each dimention of X section max arguments
                i, j = np.unravel_index(self.max_indices[:, y, x], (self.pool_size, self.pool_size))
                b, c = np.unravel_index(np.arange(batch_size * channels), (batch_size, channels))
                # selecting section of X
                filter_y = slice(y * self.stride, y * self.stride + self.pool_size)
                filter_x = slice(x * self.stride, x * self.stride + self.pool_size)
                section = d_X[:, filter_y, filter_x]
                # add grad so that next step would not overwrite previous step result
                section[b.flatten(), i.flatten(), j.flatten(), c.flatten()] += d_out[:, y, x].flatten()
        
        return d_X

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = X.shape
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, height*width*channels]
        return X.copy().reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.copy().reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
