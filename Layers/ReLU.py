from Layers.Base import BaseLayer
import numpy as np

# Rectified Linear Unit(ReLU)
""" ReLU replace negative values to 0, Helps to make model non-linear """


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.where(self.input_tensor >= 0, self.input_tensor, 0)  # Where input_tensor value >= 0, input_tensor, otherwise 0.

    def backward(self, error_tensor):
        return np.where(self.input_tensor >= 0, error_tensor, 0)  # Where input_tensor value >= 0, error_tensor, otherwise 0.
