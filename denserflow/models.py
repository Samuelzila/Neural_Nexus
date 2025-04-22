from . import layers, optimizers
import os
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            file.write(json.dumps(self.to_dict()))

    def to_dict(self):
        """
        Returns the model as a dictionary
        """
        ret_dict = {
            "type": "sequential",
            "layers": [layer.to_dict() for layer in self.layers]
        }

        if hasattr(self, "optimizer"):
            ret_dict["optimizer"] = self.optimizer.to_dict()

        return ret_dict

    @classmethod
    def from_dict(cls, obj_dict):
        """
        Create a sequential model by passing in its dictionnary representation
        """
        layers_arr = []

        for layer_dict in obj_dict["layers"]:
            layers_arr.append(layers.from_dict(layer_dict))

        obj = cls(layers_arr)

        if "optimizer" in obj_dict:
            obj.optimizer = optimizers.from_dict(obj_dict["optimizer"])

        return obj

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

    return model_class.from_dict(obj)
