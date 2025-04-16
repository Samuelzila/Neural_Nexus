import numpy as np


class SGD:
    def __init__(self, learning_rate=1, decay=0., momentum=0.):
        # The initial learning rate
        self.learning_rate_0 = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.momentum = momentum
        self.iteration = 0

    def pre_update(self):
        """
        Call only once before updating all layers in the model
        """
        if self.decay:
            self.learning_rate = self.learning_rate_0 * \
                (1./(1.+self.decay*self.iteration))

    def update_layer(self, layer):
        """
        Will apply stochastic gradient descent on the layer being passed in.
        The layers backpropagation function must have been called for the derivatives to be calculated.

        Some functionnalities like decaying need pre_update() and post_update().
        """
        # If momentum is used
        if self.momentum:
            # Create momentum matrix for weights and biaises in the layer if they don't exist.
            if not hasattr(layer, "weights_momentum"):
                layer.weights_momentum = np.zeros_like(layer.weights)
                layer.biaises_momentum = np.zeros_like(layer.biaises)

            weight_neg_gradient = self.momentum * layer.weights_momentum - \
                self.learning_rate * layer.dweights
            layer.weights_momentum = weight_neg_gradient

            biais_neg_gradient = self.momentum * layer.biaises_momentum - \
                self.learning_rate * layer.dbiaises
            layer.biaises_momentum = biais_neg_gradient
        else:
            # Doing this in an else statement saves computation time,
            # even if mathematically equivalent to the above.
            weight_neg_gradient = -self.learning_rate * layer.dweights
            biais_neg_gradient = -self.learning_rate * layer.dbiaises

        layer.weights += weight_neg_gradient
        layer.biaises += biais_neg_gradient

    def post_update(self):
        """
        Call only once after updating all layers in the model
        """
        self.iteration += 1


class AdaGrad:
    def __init__(self, learning_rate=1, decay=0., epsilon=1e-7):
        # The initial learning rate
        self.learning_rate_0 = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iteration = 0

    def pre_update(self):
        """
        Call only once before updating all layers in the model
        """
        if self.decay:
            self.learning_rate = self.learning_rate_0 * \
                (1./(1.+self.decay*self.iteration))

    def update_layer(self, layer):
        """
        Will apply adagrad on the layer being passed in.
        The layers backpropagation function must have been called for the derivatives to be calculated.

        Some functionnalities like decaying need pre_update() and post_update().
        """
        # Create cache matrix for weights and biaises in the layer if they don't exist.
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biais_cache = np.zeros_like(layer.biaises)

        layer.weight_cache += layer.dweights**2
        layer.biais += layer.dbiaises**2

        weight_neg_gradient = -self.learning_rate * \
            layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        biais_neg_gradient = -self.learning_rate * \
            layer.dbiaises/(np.sqrt(layer.biais_cache)+self.epsilon)

        layer.weights += weight_neg_gradient
        layer.biaises += biais_neg_gradient

    def post_update(self):
        """
        Call only once after updating all layers in the model
        """
        self.iteration += 1


class RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        # The initial learning rate
        self.learning_rate_0 = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iteration = 0
        self.rho = rho

    def pre_update(self):
        """
        Call only once before updating all layers in the model
        """
        if self.decay:
            self.learning_rate = self.learning_rate_0 * \
                (1./(1.+self.decay*self.iteration))

    def update_layer(self, layer):
        """
        Will apply RMSprop on the layer being passed in.
        The layers backpropagation function must have been called for the derivatives to be calculated.

        Some functionnalities like decaying need pre_update() and post_update().
        """
        # Create cache matrix for weights and biaises in the layer if they don't exist.
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biais_cache = np.zeros_like(layer.biaises)

        layer.weight_cache = self.rho*layer.weight_cache + \
            (1-self.rho)*layer.dweights**2
        layer.biais_cache = self.rho*layer.biais_cache + \
            (1-self.rho)*layer.dbiaises**2

        weight_neg_gradient = -self.learning_rate * \
            layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        biais_neg_gradient = -self.learning_rate * \
            layer.dbiaises/(np.sqrt(layer.biais_cache)+self.epsilon)

        layer.weights += weight_neg_gradient
        layer.biaises += biais_neg_gradient

    def post_update(self):
        """
        Call only once after updating all layers in the model
        """
        self.iteration += 1


class Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        # The initial learning rate
        self.learning_rate_0 = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iteration = 0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update(self):
        """
        Call only once before updating all layers in the model
        """
        if self.decay:
            self.learning_rate = self.learning_rate_0 * \
                (1./(1.+self.decay*self.iteration))

    def update_layer(self, layer):
        """
        Will apply adam on the layer being passed in.
        The layers backpropagation function must have been called for the derivatives to be calculated.

        Some functionnalities like decaying need pre_update() and post_update().
        """
        # Create cache and momentum matrices for weights and biaises in the layer if they don't exist.
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.biais_momentum = np.zeros_like(layer.biaises)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.biais_cache = np.zeros_like(layer.biaises)

        # Update momentum (similar to caching in RMSprop)
        layer.weight_momentum = self.beta_1 * \
            layer.weight_momentum+(1-self.beta_1)*layer.dweights
        layer.biais_momentum = self.beta_1 * \
            layer.biais_momentum+(1-self.beta_1)*layer.dbiases

        # Corrected momentum
        weight_momentum_corrected = layer.weight_momentum / \
            (1-self.beta_1**(self.iteration+1))
        biais_momentum_corrected = layer.biais_momentum / \
            (1-self.beta_1**(self.iteration+1))

        layer.weight_cache = self.beta_2*layer.weight_cache + \
            (1-self.beta_2)*layer.dweights**2
        layer.biais_cache = self.beta_2*layer.biais_cache + \
            (1-self.beta_2)*layer.dbiaises**2

        # Corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1-self.beta_1**(self.iteration+1))
        biais_cache_corrected = layer.biais_cache / \
            (1-self.beta_1**(self.iteration+1))

        weight_neg_gradient = -self.learning_rate * \
            weight_momentum_corrected / \
            (np.sqrt(weight_cache_corrected)+self.epsilon)
        biais_neg_gradient = -self.learning_rate * \
            biais_momentum_corrected / \
            (np.sqrt(biais_cache_corrected)+self.epsilon)

        layer.weights += weight_neg_gradient
        layer.biaises += biais_neg_gradient

    def post_update(self):
        """
        Call only once after updating all layers in the model
        """
        self.iteration += 1


optimizer_type_dict = {
    "sgd": SGD,
    "adagrad": AdaGrad,
    "rmsprop": RMSprop,
    "adam": Adam
}
