from Layers.Base import BaseLayer
import numpy as np

""" Transform the output of the network into a probability distribution, normalizes the results by dividing each exponential element by the
    sum of all exponential elements. As a result, the output vector represents probabilities that sum up to 1, indicating the likelihood 
    of each class.
    
    softmax(x_i) = exp(x_i) / sum(exp(x_j))"""


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None
        self.error_tensor = None
        self.output = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        input_tensor = input_tensor - np.max(input_tensor)  # It prevents from numerical overflow [large values] during exponentiation.
        n = np.exp(input_tensor)
        self.output = n / np.sum(n, axis=1, keepdims=True)  # axis = 1 -> Row[corresponding features], axis = 0 -> Column,
        return self.output

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        # [:, None] = Reshape into column vector by Adding extra dimension => (batch_size,1)
        vector_sum = (self.output * self.error_tensor).sum(axis=1)[:, None]  # (batch_size, 1)
        return self.output * (self.error_tensor - vector_sum)
