import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.label_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor

        '''Equation: -sum(true_class * log(predicted_class))'''
        # np.finfo(float).eps -> Adds a small epsilon value, to prevents undefined or infinite results i.e log(0) -> undefined
        return np.sum(label_tensor * -np.log(prediction_tensor + np.finfo(float).eps))

    def backward(self, label_tensor):
        self.label_tensor = label_tensor
        # self.label_tensor == 1 -> identifies the the correct class.
        # -1 / self.prediction_tensor -> derivative of the Cross-Entropy Loss with respect to the inputs. [Gradient of Loss]
        # 0 -> This ensures that the gradient remains 0 for incorrect class predictions.
        return np.where(self.label_tensor == 1, -1 / self.prediction_tensor, 0)
