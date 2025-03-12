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
