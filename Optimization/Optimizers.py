import numpy as np

"""Optimizers: These are the algorithms or methods to change the weights to reduce loss."""


class base_optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


# Stochastic Gradient Descent (SGD)

class Sgd(base_optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # (new weight = old weight - learning rate * gradient tensor)
        # Gradient Tensor is the partial derivatives of the loss function with respect to the weights

        """Ex 3"""
        if self.regularizer:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * shrinkage
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(base_optimizer):
    # Momentum : Steps of getting minima
    # help avoid getting stuck in local minima
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0  # Initialize velocity mask = 0, keeps track of previous interation

    def calculate_update(self, weight_tensor, gradient_tensor):
        # mask(k) = momentum * mask(k-1) - learning rate * gradient

        # self.momentum_rate * self.mask --> updates the velocity, scaled by the momentum rate
        # '- self.learning_rate * gradient_tensor' --> adjusts the velocity based on the current gradient and the learning rate.

        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        # return weight_tensor + self.mask  # updated weights by adding the velocity

        'Ex 3'
        if self.regularizer:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * shrinkage

        return weight_tensor + self.v


# Adaptive Moment Estimation

class Adam(base_optimizer):
    # It uses estimations of the first and second moments of the gradient
    # to adapt the learning rate for each weights of the neural network.

    # Parameter update based on current and past gradients
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu  # Momentum
        self.rho = rho  # Gradient moving average
        self.v = 0  # First oder moment estimate
        self.r = 0  # Second order moment estimate
        self.t = 1  # Time step ; T iteration

    def calculate_update(self, weight_tensor, gradient_tensor):
        # mask(k) = mu*v_(k-1) + (1-mu)g_k
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        # r^k = rho*r_(k-1) + (1-rho)*g**2
        self.r = self.rho * self.r + (1 - self.rho) * (gradient_tensor ** 2)

        '''Correction:'''
        new_v = self.v / (1 - self.mu ** self.t)
        new_r = self.r / (1 - self.rho ** self.t)
        self.t = self.t + 1

        'Ex 3'
        if self.regularizer:
            shrinkage = self.regularizer.calculate_gradient(weight_tensor)
            weight_tensor = weight_tensor - self.learning_rate * shrinkage

        #  Update rule for the weights in the Adam optimize
        weight_tensor = weight_tensor - (self.learning_rate * new_v) / (np.sqrt(new_r) + np.finfo(float).eps)
        return weight_tensor
