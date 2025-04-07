import numpy as np

# [] est-ce que je devrais ajouter self à chacun des classes


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
        # Number of samples
        samples = len(y_pred)

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(len(y_pred[0]))[y_true]

        # Calculate gradient
        dinputs = -y_true / y_pred
        # Normalize gradient
        dinputs = dinputs / samples
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
        # TODO: Store data?
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


# class Softmax_CCE:
#     # Creates activation and loss function objects
#     def __init__(self):
#         self.activation = Softmax()
#         self.loss = cross_entropy_loss
#
#     # Forward pass
#     def __call__(self, inputs):
#         # Output layer's activation function
#         self.activation(inputs)
#         # Set the output
#         self.output = self.activation.output
#         # Calculate and return loss value
#         return self.output
#     # Backward pass
#
#     def backpropagation(self, dvalues, y_true):
#         # Number of samples
#         samples = len(dvalues)
#         # If labels are one-hot encoded,
#         # turn them into discrete values
#         # FIXME:
#         # if len(y_true.shape) == 3:
#         #     y_true = np.argmax(y_true, axis=1)
#         y_true = y_true.flatten()
#
#         # Copy so we can safely modify
#         self.dinputs = dvalues.copy()
#         # Calculate gradient
#         self.dinputs[range(samples), y_true] -= 1
#         # Normalize gradient
#         self.dinputs = self.dinputs / samples


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


activation_type_dict = {
    "relu": Relu,
    "sigmoid": sigmoid,
    "softmax": Softmax,
    None: IdentityFunction
}


def neuron_activation(activation):
    return activation_type_dict.get(activation)


class Dense(Layer):
    """
    Couche dense (fully connected).
    """

    def __init__(self, nb_neurons, activation=None):
        self.biaises = np.zeros((1, nb_neurons))
        self.weights = None
        self.nb_neurons = nb_neurons
        self.activation = neuron_activation(activation)()

    def __call__(self, inputs):
        self.inputs = inputs
        if self.weights is None:
            self.weights = 0.01 * \
                np.random.randn(inputs.shape[1], self.nb_neurons)
        # Calcul de la sortie linéaire
        output = np.dot(inputs, self.weights) + self.biaises
        output = self.activation(output)
        return output

    def backpropagation(self, dvalues, learning_rate):
        # Calcul du gradient via la dérivée de la fonction d'activation.
        # Ici, on utilise la méthode backpropagation définie pour l'activation choisie.
        dactivation = self.activation.backpropagation(dvalues)
        # Calcul des gradients pour les poids et biais.
        dweights = np.dot(self.inputs.T, dactivation)
        dbiaises = np.sum(dactivation, axis=0, keepdims=True)

        # Propagation du gradient vers la couche précédente.
        dinputs = np.dot(dactivation, self.weights.T)

        # Mise à jour des paramètres de la couche.
        self.weights -= learning_rate * dweights

        # Comme self.biaises est un tableau 1D, on le met à jour en le convertissant si besoin.
        self.biaises -= learning_rate * dbiaises[0]

        return dinputs

    def to_dict(self):
        """
        Returns the object as a dict
        """
        return {
            "type": "dense",
            "biaises": self.biaises.tolist(),
            "weights": self.weights.tolist(),
            "nb_neurons": self.nb_neurons,
            "activation": self.activation.type()
        }

    @classmethod
    def from_dict(cls, layer_dict):
        """
        Layer constructor from dictionary
        """
        layer = cls(layer_dict["nb_neurons"],
                    activation=layer_dict["activation"])

        layer.biaises = np.array(layer_dict["biaises"])
        layer.weights = np.array(layer_dict["weights"])

        return layer


layer_type_dict = {
    "dense": Dense
}


def from_dict(layer_dict):
    """
    Converts a dict into a layer object
    """
    layer = layer_type_dict.get(layer_dict["type"])

    return layer.from_dict(layer_dict)
