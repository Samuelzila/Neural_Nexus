import numpy as np


class CategoricalCrossEntropy():
    def __call__(self, y_pred, y_true):
        samples = len(y_pred)
        epsilon = 1e-7  # pour éviter log(0)
        # Clip des prédictions pour éviter des valeurs exactement 0 ou 1
        y_pred_clipped = np.clip(y_pred, epsilon, 1. - epsilon)

        # Loss function for categorical labels.
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Loss function using one-hot labels.
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backpropagation(self, y_pred, y_true):
        # Prevent division by zero
        epsilon = 1e-7
        y_pred[np.abs(y_pred) < epsilon] = epsilon

        # Number of samples
        samples = len(y_pred)

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(len(y_pred[0]))[y_true]

        # Calculate gradient
        dinputs = -y_true / y_pred
        # Normalize gradient
        dinputs = dinputs / (samples)
        return dinputs


class MeanSquaredError():
    def __call__(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)

    def backpropagation(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.shape[0]
