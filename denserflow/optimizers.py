import numpy as np


class SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_layer(self, layer):
        """
        Will apply stochastic gradient descent on the layer being passed in.
        The layers backpropagation function must have been called for the derivatives to be calculated.
        """
        # If momentum is used
        if self.momentum:
            # Create momentum matrix for weights and biaises in the layer if they don't exist.
            if not hasattr(layer, "weights_momentum"):
                layer.weights_momentum = np.zeros_like(layer.weights)
                layer.biaises_momentum = np.zeros_like(layer.biaises)

        weight_gradient = self.momentum * layer.weights_momentum - \
            self.learning_rate * layer.dweights
        layer.weights_momentum = weight_gradient

        biais_gradient = self.momentum * layer.biaises_momentum - \
            self.learning_rate * layer.dbiaises[0]
        layer.biaises_momentum = biais_gradient

        layer.weights += weight_gradient
        layer.biaises += biais_gradient
