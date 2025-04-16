from . import layers, optimizers
import json


class Sequential:
    """
    A sequential model takes an array of layers as a parameter. The layers are going to be used one after the other.
    """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)

        return inputs

    def compile(self, optimizer='sgd', loss="categorical_crossentropy"):
        if isinstance(optimizer, str):
            self.optimizer = optimizers.optimizer_type_dict.get(optimizer)()
        else:
            self.optimizer = optimizer

    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Passe avant (forward pass)

            y_pred = self(X)

            loss_function = layers.CategoricalCrossEntropy()

            loss = loss_function(y_pred, y)

            grad = loss_function.backpropagation(y_pred, y)

            self.optimizer.pre_update()
            # Parcours des couches en sens inverse pour rétropropager l'erreur
            for layer in reversed(self.layers):
                grad = layer.backpropagation(grad, optimizer=self.optimizer)
            self.optimizer.post_update()

    def save(self, path):
        """
        Save the model to the path using a custom json format
        """
        with open(path, 'w') as file:
            file.write(json.dumps(self.to_dict()))

    def to_dict(self):
        """
        Returns the model as a dictionary
        """
        return {
            "type": "sequential",
            "layers": [layer.to_dict() for layer in self.layers]
        }

    @classmethod
    def from_dict(cls, layer_dicts):
        layers_arr = []
        for layer_dict in layer_dicts:
            layers_arr.append(layers.from_dict(layer_dict))
        return cls(layers_arr)

    def predict(self, X):
        return self(X)


model_type_dict = {
    "sequential": Sequential
}


def load_model(path):
    """
    Loads a model from a json file
    """
    with open(path) as file:
        obj = json.load(file)

    model_class = model_type_dict.get(obj["type"])

    return model_class.from_dict(obj["layers"])
