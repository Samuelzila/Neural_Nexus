import numpy as np


class IdentityFunction:
    def __call__(self, inputs):
        return inputs

    def type(self):
        return None


class Relu:
    def __call__(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backpropagation(self, dvalues):
        dReLU = dvalues.copy()
        dReLU[self.inputs <= 0] = 0

        return dReLU

    def type(self):
        return "relu"


class sigmoid:
    def __init__(self):
        self.output = 0
        pass

    def __call__(self, inputs):
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backpropagation(self, dvalues):
        # La dérivée est.output * (1 -.output)
        return dvalues * self.output * (1 - self.output)

    def type(self):
        return "sigmoid"


class Softmax:
    def __call__(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backpropagation(self, dvalues):
        # Pour chaque échantillon, on calcule le jacobien et on l'applique au gradient reçu.
        dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Construire la matrice jacobienne pour la softmax
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(
                single_output) - np.dot(single_output, single_output.T)
            # Propager le gradient pour cet échantillon
            dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
        return dinputs

    def type(self):
        return "softmax"


# A dictionary associating names of activation functions as strings with their class.
activation_type_dict = {
    "relu": Relu,
    "sigmoid": sigmoid,
    "softmax": Softmax,
    None: IdentityFunction
}


def neuron_activation(activation):
    """
    Given the name of an activation function as a string, returns the associated class.
    """
    return activation_type_dict.get(activation)
