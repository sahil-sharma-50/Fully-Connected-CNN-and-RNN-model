import numpy as np
from Layers.Base import BaseLayer
import copy


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(size=(input_size, output_size))
        self.bias = np.random.uniform(size=(1, output_size))
        self.input_tensor = None
        self._optimizer = None
        self.gradient_weights = None
        self.gradient_bias = None

    def forward(self, input_tensor):
        # Perform the forward pass of the fully connected layer.
        self.input_tensor = input_tensor
        return np.dot(input_tensor, self.weights) + self.bias

    def backward(self, error_tensor):
        # Perform the backward pass of the fully connected layer.
        gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        gradient_bias = np.sum(error_tensor, axis=0)
        if self._optimizer:
            self.weights = self._optimizer.weight.calculate_update(self.weights, gradient_weights)
            self.bias = self._optimizer.bias.calculate_update(self.bias, gradient_bias)
        self.gradient_bias = error_tensor
        self.gradient_weights = gradient_weights
        gradient_input = np.dot(error_tensor, self.weights.T)
        return gradient_input

    def initialize(self, weights_initializer, bias_initializer):
        # Initialize the weights and biases of the fully connected layer.
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.output_size)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        # Set the optimizer for the fully connected layer.
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights
