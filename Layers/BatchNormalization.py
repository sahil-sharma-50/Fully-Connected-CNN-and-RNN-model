import numpy as np
from Layers.Base import BaseLayer
from Layers.Helpers import compute_bn_gradients


"""Batch normalization helps in improving training stability and accelerating convergence 
by normalizing the input data and adjusting its scale and shift."""


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weights = np.ones(channels)
        self.bias = np.zeros(channels)
        self.trainable = True
        self.testing_phase = False
        self.mean = 0
        self.variance = 1
        self.test_mean = 0
        self.test_variance = 1
        self.input_tensor = None
        self.input_shape = None
        self.normalized_input = None  # normalized_input (to store the normalized input tensor)
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None
        self._bias_optimizer = None

    def forward(self, input_tensor, alpha=0.8):
        # The method supports both 2D and 4D input tensors (for fully connected and convolutional layers, respectively).
        epsilon = 1e-10
        self.input_tensor = input_tensor

        # 2D input vector for fully connected layers
        if len(input_tensor.shape) == 2:
            self.mean = np.mean(input_tensor, axis=0)  # mean of the input tensor are computed
            self.variance = np.var(input_tensor, axis=0)  # variance of the input tensor are computed
            if not self.testing_phase:
                new_mean = np.mean(input_tensor, axis=0)
                new_variance = np.var(input_tensor, axis=0)
                self.test_mean = alpha * self.mean + (1 - alpha) * new_mean
                self.test_variance = alpha * self.variance + (1 - alpha) * new_variance
                self.mean = new_mean
                self.variance = new_variance
                normalized_input = (input_tensor - self.mean) / np.sqrt(self.variance + epsilon)  # normalized input tensor (normalized_input)
            else:
                normalized_input = (input_tensor - self.test_mean) / np.sqrt(self.test_variance + epsilon)  # normalized input tensor (normalized_input)
            self.normalized_input = normalized_input
            output = self.weights * normalized_input + self.bias  # The scaling & shifting operations are applied using the learned weights and biases.

        # 4D input vector for convolutional layers
        elif len(input_tensor.shape) == 4:
            batch_size, height, width, num_channels = input_tensor.shape
            self.mean = np.mean(input_tensor, axis=(0, 2, 3))  # mean of the input tensor are computed
            self.variance = np.var(input_tensor, axis=(0, 2, 3))  # variance of the input tensor are computed
            if not self.testing_phase:
                new_mean = np.mean(input_tensor, axis=(0, 2, 3))
                new_variance = np.var(input_tensor, axis=(0, 2, 3))
                self.test_mean = alpha * self.mean.reshape((1, height, 1, 1)) + \
                                 (1 - alpha) * new_mean.reshape((1, height, 1, 1))

                self.test_variance = alpha * self.variance.reshape((1, height, 1, 1)) + \
                                     (1 - alpha) * new_variance.reshape(
                    (1, height, 1, 1))
                self.mean = new_mean
                self.variance = new_variance
                # normalized input tensor (normalized_input)
                normalized_input = (input_tensor - self.mean.reshape((1, height, 1, 1))) \
                        / np.sqrt(self.variance.reshape((1, height, 1, 1)) + epsilon)
            else:
                # normalized input tensor (normalized_input)
                normalized_input = (input_tensor - self.test_mean.reshape((1, height, 1, 1))) \
                        / np.sqrt(self.test_variance.reshape((1, height, 1, 1)) + epsilon)
            self.normalized_input = normalized_input
            # The scaling & shifting operations are applied using the learned weights and biases.
            output = self.weights.reshape((1, height, 1, 1)) * normalized_input + self.bias.reshape((1, height, 1, 1))

        return output

    def backward(self, error_tensor):
        # For 2D
        if len(error_tensor.shape) == 2:
            self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=0)
            self.gradient_bias = np.sum(error_tensor, axis=0)
            output = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.variance, 1e-15)

        # For 4D
        elif len(error_tensor.shape) == 4:
            self.gradient_weights = np.sum(error_tensor * self.normalized_input, axis=(0, 2, 3))
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))
            output = compute_bn_gradients(self.reformat(error_tensor), self.reformat(self.input_tensor),
                                          self.weights, self.mean, self.variance, 1e-15)
            output = self.reformat(output)

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return output

    """The reformat method is a utility function used to reshape the input tensor for gradient computation.
     It reshapes tensor based on its dimensions & rearranges the axes needed to match the computation requirements."""

    def reformat(self, tensor):
        output = np.zeros_like(tensor)
        # For 2D
        if len(tensor.shape) == 2:
            try:
                batch_size, height, width, num_channels = self.input_shape
            except:
                batch_size, height, width, num_channels = self.input_tensor.shape
            output = tensor.reshape((batch_size, width * num_channels, height))
            output = np.transpose(output, (0, 2, 1))
            output = output.reshape((batch_size, height, width, num_channels))

        # For 4D
        if len(tensor.shape) == 4:
            batch_size, height, width, num_channels = tensor.shape
            output = tensor.reshape((batch_size, height, width * num_channels))
            output = np.transpose(output, (0, 2, 1))
            batch_size, MN, height = output.shape
            output = output.reshape((batch_size * MN, height))
        return output

    """The initialize method initializes the weights and biases of the batch normalization layer. 
    In this case, the weights are set to ones and the biases to zeros."""
    def initialize(self, weight_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @gradient_weights.deleter
    def gradient_weights(self):
        del self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @gradient_bias.deleter
    def gradient_bias(self):
        del self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @optimizer.deleter
    def optimizer(self):
        del self._optimizer

    @property
    def bias_optimizer(self):
        return self._bias_optimizer

    @bias_optimizer.setter
    def bias_optimizer(self, value):
        self._bias_optimizer = value

    @bias_optimizer.deleter
    def bias_optimizer(self):
        del self._bias_optimizer
