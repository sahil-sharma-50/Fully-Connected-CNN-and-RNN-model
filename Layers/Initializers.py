import numpy as np

"""These initializers help to ensure that the weights of the neural network are initialized with appropriate values,
which can help in faster convergence, better generalization, and avoiding problems such as vanishing gradients or exploding gradients."""

# Fan_in : Input dimension of weights (No. of inputs)
# Fan_out : Output dimension of weights (No. of outputs)


class Constant:
    # Typically for Bias initialization
    def __init__(self, weight_initialization=0.1):
        self.weight_initialization = weight_initialization

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.zeros((fan_in, fan_out)) + self.weight_initialization


class UniformRandom:
    def __init__(self):
        pass

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        return np.random.uniform(0.0, 1.0, weights_shape)


class Xavier:
    """ Initializer for Weights with normal distribution with
    mean 0 and standard deviation of = sqrt(2/(fan_in+fan_out))"""
    def __init__(self):
        pass

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        # square_root(2/fan_out+fan_in)
        return np.random.normal(0, (2 / (fan_out + fan_in)) ** (1 / 2), weights_shape)


class He:
    """ Same as Xavier but it uses fan_in parameter only to compute
     the standard deviation for weight initialization, i.e., sqrt(2/fan_in)."""
    def __init__(self):
        pass

    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        # square_root(2/fan_in)
        return np.random.normal(0, (2 / fan_in) ** (1 / 2), weights_shape)
