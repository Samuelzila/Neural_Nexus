import numpy as np
from . import optimizers, activations


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


class Dense(Layer):
    """
    Dense layer (fully connected).
    All neurons in the input are connected to all neurons of this layer.
    """

    def __init__(self, nb_neurons, activation=None):
        self.nb_neurons = nb_neurons
        self.biases = np.zeros((1, nb_neurons))
        self.weights = None
        self.activation = activations.neuron_activation(activation)()

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
