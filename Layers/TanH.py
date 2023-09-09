import numpy as np
from Layers.Base import BaseLayer


"""The tanh activation function squashes the input values between -1 and 1, providing non-linear transformations to the input.
Applied in Hidden layers
Problem: Vanishing Gradient"""


class TanH(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, input_tensor):
        self.output = np.tanh(input_tensor)  # Compute the element-wise hyperbolic tangent of the input_tensor.
        return self.output

    def backward(self, error_tensor):
        # (1 - self.output ** 2) -> is the derivative of the tanh function with respect to its input.
        # This derivative is also known as the Jacobian matrix of the tanh function.
        gradient = error_tensor * (1 - self.output ** 2)
        return gradient
