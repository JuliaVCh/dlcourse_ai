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
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
#    loss = reg_strength * np.sum(np.square(W))
#    grad = 2 * reg_strength * W
    loss = 0.5 * reg_strength * np.sum(np.square(W))
    grad = reg_strength * W

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
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
    """
    # fix argument dimentions so that they are exactly as expected
    if np.ndim(target_index) > 1:
        target_index = target_index.flatten()
    # implement softmax with cross-entropy
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    # calculate gradient
    dprediction = probs
    if np.ndim(target_index) == 0:
        dprediction[target_index] -= 1
    else:
        #print(np.all(dprediction[np.arange(np.size(target_index)), target_index] > 0))
        dprediction[np.arange(np.size(target_index)), target_index] -= 1
        # does or does not grad calculation need averaging?
        dprediction = dprediction / np.size(target_index)

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # Implement forward pass
        self.X = deepcopy(X)
        self.X[self.X < 0] = 0

        return self.X

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # Implement backward pass
        if self.X is None:
            raise Exception("ReLULayer forward pass was not implemented yet!")
        else:
            #d_result = np.where(self.X == 0, 0, d_out)
            d_result = deepcopy(d_out)
            d_result[self.X == 0] = 0
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        # a = 0.001 does not work at fucking all!
        a = 1 / np.sqrt(n_input)
        self.W = Param(a * np.random.randn(n_input, n_output)) 
        self.B = Param(a * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Implement forward pass
        self.X = deepcopy(X)
        
        return np.dot(self.X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Implement backward pass
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
        return {'W': self.W, 'B': self.B}
