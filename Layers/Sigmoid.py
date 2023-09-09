import numpy as np
from Layers.Base import BaseLayer


"""The sigmoid activation function squashes the input values between 0 and 1, providing non-linear transformations to the input.
Used for binary classification
Problem: Vanishing Gradient"""


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, input_tensor):
        # Adding 1 to the exponential negative input tensor ensures that the denominator of the sigmoid function is always positive.
        self.output = 1 / (1 + np.exp(-1 * input_tensor))
        return self.output

    def backward(self, error_tensor):
        # self.output * (1 - self.output) -> is the derivative of the sigmoid function with respect to its input.
        gradient = error_tensor * self.output * (1 - self.output)
        return gradient
