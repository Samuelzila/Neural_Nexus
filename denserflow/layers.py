import numpy as np


def neuron_activation(name):
    """
    This function returns the specific activation function associated with the string.
    """
    match name:
        case "relu":
            return relu
        case "softmax":
            return softmax
        case "sigmoid":
            return sigmoid
        case None:
            return identity


def identity(inputs):
    return inputs


def relu(inputs):
    return np.maximum(0, inputs)


def softmax(inputs):
    exp = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
    return exp/np.sum(exp, axis=1, keepdims=True)


def sigmoid(inputs):
    return 1/(1+np.exp(-inputs))


class Layer:
    """
    Abstract layer class, to be used as a parent.
    """

    def forward(self, inputs):
        """
        The forward pass is the calculation that is done on the data.
        It takes the inputs of the previous layer and outputs the values of its own, in a numpy array.
        """
        raise NotImplementedError


class Dense(Layer):
    """
    Also know as a fully connected layer.
    """

    def __init__(self, nb_neurons, activation=None):
        self.biaises = np.random.rand(1, nb_neurons)[0]
        self.weights = None
        self.nb_neurons = nb_neurons
        self.activation = activation

    def __call__(self, inputs):
        if self.weights is None:
            self.weights = np.random.rand(inputs.shape[1], self.nb_neurons)

        output = np.dot(inputs, self.weights) + self.biaises
        return neuron_activation(self.activation)(output)
