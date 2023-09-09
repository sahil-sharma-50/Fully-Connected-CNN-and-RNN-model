import numpy as np
import copy
from Layers.Base import BaseLayer
from Layers.FullyConnected import FullyConnected
from Layers.TanH import TanH

"""Recurrent Neural Networks (RNNs) are a type of neural network architecture designed for processing sequential data.
RNNs are especially useful when there is a dependency on previous information in the sequence to make predictions or decisions."""


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.FC_h = FullyConnected(hidden_size + input_size, hidden_size)
        self.FC_y = FullyConnected(hidden_size, output_size)
        self.gradient_weights_n = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        self.weights = self.FC_h.weights
        self.tanH = TanH()
        self.num_time_steps = 0  # bptt
        self.h_mem = []
        self.trainable = True
        self._memorize = False
        self.gradient_weights_y = None
        self.gradient_weights_h = None
        self.input_tensor = None
        self.output_error = None
        self.weights_y = None
        self.weights_h = None
        self.hidden_state = None
        self.prev_hidden_state = None
        self.batch_size = None
        self.optimizer = None

    def forward(self, input_tensor):
        """The forward pass calculates the hidden state for each time step based on the input tensor and the previous hidden state.
        The hidden state is then passed through the output layer to generate the output tensor."""
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        if self._memorize:
            if self.hidden_state is None:
                self.hidden_state = np.zeros((self.batch_size + 1, self.hidden_size))
            else:
                self.hidden_state[0] = self.prev_hidden_state
        else:
            self.hidden_state = np.zeros((self.batch_size + 1, self.hidden_size))
        output_tensor = np.zeros((self.batch_size, self.output_size))
        for batch in range(self.batch_size):
            hidden_ax = self.hidden_state[batch][np.newaxis, :]  # Extract hidden state for the current batch
            input_ax = input_tensor[batch][np.newaxis, :]  # Extract input tensor for the current batch
            # Concatenate hidden state & input tensors along the feature axis.
            input_new = np.concatenate((hidden_ax, input_ax), axis=1)
            self.h_mem.append(input_new)  # Store the new input tensor in the h_mem list.
            # Pass new input through FC_h to compute the input tanH activation function.
            hidden_state_input = self.FC_h.forward(input_new)
            self.hidden_state[batch + 1] = TanH().forward(hidden_state_input)  # Apply TanH().forward() to the hidden state input
            # Pass next hidden state through the output layer (FC_y) to compute the output tensor for the current batch.
            # Store the output tensor in the corresponding row of the output_tensor placeholder.
            output_tensor[batch] = self.FC_y.forward(self.hidden_state[batch + 1][np.newaxis, :])
            # Update prev_hidden_state with last hidden state from current forward pass.
        self.prev_hidden_state = self.hidden_state[-1]
        return output_tensor

    def backward(self, error_tensor):
        self.output_error = np.zeros((self.batch_size, self.input_size))
        self.gradient_weights_y = np.zeros((self.hidden_size + 1, self.output_size))
        self.gradient_weights_h = np.zeros((self.hidden_size + self.input_size + 1, self.hidden_size))
        count = 0
        gradient_tanh = 1 - self.hidden_state[1:] ** 2
        hidden_error = np.zeros((1, self.hidden_size))
        for batch in reversed(range(self.batch_size)):
            # Back-propagate the error through the output layer (FC_y.backward()) and store the resulting error tensor.
            output_layer_error = self.FC_y.backward(error_tensor[batch][np.newaxis, :])

            # Set the input tensor of FC_y to be the concatenation of the current hidden state and a bias term.
            self.FC_y.input_tensor = np.hstack((self.hidden_state[batch + 1], 1))[np.newaxis, :]

            # Compute the gradient of the hidden state error by adding the hidden error and output layer error.
            hidden_state_gradient_error = hidden_error + output_layer_error

            # Compute gradient for hidden layer by multiplying the gradient of tanh activation function (gradient_tanh)
            # with the hidden state gradient error.
            gradient_hidden = gradient_tanh[batch] * hidden_state_gradient_error
            hidden_layer_error = self.FC_h.backward(gradient_hidden)

            # Separate the hidden layer error into the hidden error and input error components.
            hidden_error = hidden_layer_error[:, :self.hidden_size]
            input_error = hidden_layer_error[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]

            # Store the input error in the corresponding row of the output error tensor.
            self.output_error[batch] = input_error

            # Compute the input tensor for FC_h by concatenating the hidden state, input tensor, and a bias term.
            conn = np.hstack((self.hidden_state[batch], self.input_tensor[batch], 1))
            self.FC_h.input_tensor = conn[np.newaxis, :]

            # If the current count is less than or equal to the number of time steps,
            # update the weights, gradients, and the corresponding weights and gradients of FC_y and FC_h.
            if count <= self.num_time_steps:
                self.weights_y = self.FC_y.weights
                self.weights_h = self.FC_h.weights
                self.gradient_weights_y = self.FC_y.gradient_weights
                self.gradient_weights_h = self.FC_h.gradient_weights

            count += 1  # Increment the count.

        # If an optimizer is specified, update the weights of FC_y and FC_h using the calculated weight gradients.
        if self.optimizer:
            self.weights_y = self.optimizer.calculate_update(self.weights_y, self.gradient_weights_y)
            self.weights_h = self.optimizer.calculate_update(self.weights_h, self.gradient_weights_h)
            self.FC_y.weights = self.weights_y
            self.FC_h.weights = self.weights_h
        return self.output_error

    # initialize method is used to initialize the weights and biases of the fully connected layers within the RNN class.
    def initialize(self, weights_initializer, bias_initializer):
        self.FC_h.initialize(weights_initializer, bias_initializer)
        self.FC_y.initialize(weights_initializer, bias_initializer)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self.FC_h.weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    @property
    def gradient_weights(self):
        return self.gradient_weights_n

    @gradient_weights.setter
    def gradient_weights(self, gradient_weights):
        self.FC_y.gradient_weights = gradient_weights
