import numpy as np
from . import optimizers


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


class Layer:
    """
    Abstract layer class, to be used as a parent.
    """

    def __call__(self, inputs):
        """
        The forward pass is the calculation that is done on the data.
        It takes the inputs of the previous layer and outputs the values of its own, in a numpy array.
        """
        raise NotImplementedError


def mse_loss(y_true, y_pred):
    """
    Calcule l'erreur quadratique moyenne entre les valeurs réelles et les prédictions.

    Paramètres:
    y_true -- valeurs réelles (numpy array)
    y_pred -- valeurs prédites par le modèle (numpy array)

    Retourne:
    La perte moyenne sur tous les exemples.
    """
    return np.mean((y_true - y_pred) ** 2)


def mse_loss_derivative(y_true, y_pred):
    """
    Calcule la dérivée de l'erreur quadratique moyenne par rapport aux prédictions.

    Paramètres:
    y_true -- valeurs réelles (numpy array)
    y_pred -- valeurs prédites par le modèle (numpy array)

    Retourne:
    Le gradient de la perte par rapport aux prédictions.
    """
    # On divise par le nombre d'exemples (taille du batch) pour obtenir la moyenne.
    return 2 * (y_pred - y_true) / y_true.shape[0]


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


class Dense(Layer):
    """
    Dense layer (fully connected).
    All neurons in the input are connected to all neurons of this layer.
    """

    def __init__(self, nb_neurons, activation=None):
        self.nb_neurons = nb_neurons
        self.biases = np.zeros((1, nb_neurons))
        self.weights = None
        self.activation = neuron_activation(activation)()

    def __call__(self, inputs):
        """
        The forward pass. Apply the neuronal layer to the input data.
        """
        self.inputs = inputs
        if self.weights is None:
            self.weights = 0.01 * \
                np.random.randn(inputs.shape[1], self.nb_neurons)
        # Calcul de la sortie linéaire
        output = np.dot(inputs, self.weights) + self.biases
        output = self.activation(output)
        return output

    def backpropagation(self, dvalues, optimizer=optimizers.SGD()):
        """
        During the backward pass, this function updates the layers according to the optimizer and returns the impact ("drivatives") of the inputs.
        """
        # Calcul du gradient via la dérivée de la fonction d'activation.
        # Ici, on utilise la méthode backpropagation définie pour l'activation choisie.
        dactivation = self.activation.backpropagation(dvalues)
        # Calcul des gradients pour les poids et bias.
        self.dweights = np.dot(self.inputs.T, dactivation)
        # Comme self.biases est un tableau 1D, on le met à jour en le convertissant si besoin.
        self.dbiases = np.sum(dactivation, axis=0, keepdims=True)[0]

        # Propagation du gradient vers la couche précédente.
        dinputs = np.dot(dactivation, self.weights.T)

        # Optimize this layer
        optimizer.update_layer(self)

        # Free memory
        del self.dbiases
        del self.dweights

        return dinputs

    def to_dict(self):
        """
        Returns the object as a dict
        """
        ret_dict = {
            "type": "dense",
            "biases": self.biases.tolist(),
            "weights": self.weights.tolist(),
            "nb_neurons": self.nb_neurons,
            "activation": self.activation.type(),
        }
        optional_params_numpy = ["weight_momentum",
                                 "bias_momentum", "weight_cache", "bias_cache"]
        for param in optional_params_numpy:
            if hasattr(self, param):
                ret_dict[param] = getattr(self, param).tolist()

        return ret_dict

    @classmethod
    def from_dict(cls, layer_dict):
        """
        Layer constructor from dictionary
        """
        layer = cls(layer_dict["nb_neurons"],
                    activation=layer_dict["activation"])

        layer.biases = np.array(layer_dict["biases"])
        layer.weights = np.array(layer_dict["weights"])

        # Initialize optional parameters
        optional_params_numpy = ["weight_momentum",
                                 "bias_momentum", "weight_cache", "bias_cache"]
        for param in optional_params_numpy:
            if param in layer_dict:
                setattr(layer, param, np.array(layer_dict[param]))

        return layer


# A dictionary associating names of layers as strings with their class.
layer_type_dict = {
    "dense": Dense
}


def from_dict(layer_dict):
    """
    Converts a dict into a layer object
    """
    layer = layer_type_dict.get(layer_dict["type"])

    return layer.from_dict(layer_dict)
