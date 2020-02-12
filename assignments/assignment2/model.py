import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # Create necessary layers
        self.input_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.reLU = ReLULayer()
        self.output_layer = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        params = self.params()
        # set parameter gradient to zeros
        for param_key, param in params.items():
            param.grad = np.zeros_like(param.grad)
        # Compute loss and fill param gradients
        # forward pass through the model
        out1 = self.input_layer.forward(X)
        out2 = self.reLU.forward(out1)
        predictions = self.output_layer.forward(out2)
        loss, dprediction = softmax_with_cross_entropy(predictions, y)
        # backward pass through the model
        d_out2 = self.output_layer.backward(dprediction)
        d_out1 = self.reLU.backward(d_out2)
        self.input_layer.backward(d_out1)
        # implement l2 regularization on all params
        for param_key, param in params.items():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            #print(param_key, np.mean(np.square(param.grad)), np.mean(np.square(reg_grad)))
            param.grad += reg_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # forward pass through the model
        out1 = self.input_layer.forward(X)
        out2 = self.reLU.forward(out1)
        predictions = self.output_layer.forward(out2)
        # classes predictions on the set
        y_pred = np.argmax(predictions, axis=1)

        return y_pred

    def params(self):
        result = {}

        # Implement aggregating all of the params
        params1 = self.input_layer.params()
        params2 = self.output_layer.params()
        for param_key in params1:
            result[param_key + '_1'] = params1[param_key]
            result[param_key + '_2'] = params2[param_key]
        
        return result
