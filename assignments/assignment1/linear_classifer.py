import numpy as np


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
    predictions = predictions.copy()
    
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
    #print(probs[probs <= 0].shape[0])
    if np.ndim(target_index) == 0:
        loss = -np.log(probs[target_index])
    else:
        row_count = np.size(target_index)
        row_idx = np.arange(row_count)
        likelihood = probs[row_idx, target_index]
        log_likelihood = -np.log(likelihood)
        loss = np.mean(log_likelihood)
    
    return loss


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
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    # calculate gradient
    dprediction = probs
    if np.ndim(target_index) == 0:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(np.size(target_index)), target_index] -= 1
        # does or does not grad calculation need averaging?
        dprediction = dprediction / np.size(target_index)
    
    return loss, dprediction

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

    # implement l2 regularization and gradient
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    target_index = target_index.flatten()
    predictions = np.dot(X, W)

    # implement prediction and gradient over W
    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dprediction)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            # a = 0.001 does not work at fucking all!
            self.W = 1 / np.sqrt(num_features) * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            
            for batch_indices in batches_indices:
                # implement generating batches from indices
                batch = X[batch_indices]
                # Compute loss and gradients
                CE_loss, CE_dW = linear_softmax(batch, self.W, y[batch_indices])
                reg_loss, reg_dW = l2_regularization(self.W, reg)
                # Apply gradient to weights using learning rate
                self.W -= learning_rate * (CE_dW + reg_dW)
                loss = CE_loss + reg_loss
            
            # end of training epoch
            #print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)
        
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # Implement class prediction
        predictions = np.dot(X, self.W)
        y_pred = np.argmax(predictions, axis=1)

        return y_pred



                
                                                          

            

                
