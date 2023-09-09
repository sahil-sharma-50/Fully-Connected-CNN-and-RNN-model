import numpy as np
from Layers.Base import BaseLayer

"""During training, the dropout layer randomly sets a fraction of the input units to zero, 
encouraging the network to learn more robust and generalizable features.

It helps prevent over-fitting and improves generalization by randomly dropping units during training."""


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability  # Represents the probability of retaining a unit in the dropout layer. For each input element,
        # there is a 1 - probability chance of setting it to zero during training.
        self.testing_phase = False
        self.mask = 0  # A binary mask that is randomly generated during training.
        # It has the same shape as the input tensor and is used to mask out units during the forward and backward passes.

    def forward(self, input_tensor):
        if not self.testing_phase:
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape)
            # This step ensures that the expected value of each unit remains the same, even though some units are dropped.
            input_tensor = input_tensor * self.mask / self.probability
        return input_tensor

    def backward(self, error_tensor):
        # This step applies the same dropout mask used during the forward pass to the error tensor.
        error_tensor = error_tensor * self.mask / self.probability
        return error_tensor  # During backpropagation, the dropout layer passes the error only through the active units
        # (those that were not dropped) to the previous layer, ensuring that the gradients are correctly propagated and scaled.
