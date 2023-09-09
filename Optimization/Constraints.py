import numpy as np

"""Regularization is a technique commonly used in machine learning to prevent over-fitting and improve the generalization of models."""


# L2 regularization decrease the model's parameter values.
# It encourages the model to assign higher weights to more important features.
class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha  # represents the regularization strength. The value of alpha determines the amount of regularization applied.

    def calculate_gradient(self, weights):  # Calculate gradient with respect to weights
        return self.alpha * weights

    def norm(self, weights):  # calculates the regularization norm of the weights.
        sum_of_squares = np.sum(np.square(weights))  # Square of parameter weights
        return self.alpha * sum_of_squares  # It quantifies overall magnitude of weights and used as a penalty term in the loss function.


# It encourages the model to select a subset of the most important features.
# L1 regularization force model's parameter values to zero
class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        # This approach promotes sparsity in the weights by encouraging many weights to become zero.
        return self.alpha * np.sign(weights)  # compute the sign of each weight element

    def norm(self, weights):
        sum_of_abs = np.sum(np.abs(weights))  # absolute values of the weights -> np.abs()
        return self.alpha * sum_of_abs  # It contributes to the sparsity-inducing effect of L1 regularization.
