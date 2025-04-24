"""
This code is used to train a model continuously, and is meant to be executed by itself.
It saves a model every 10 iterations.
If you press CTRL-C, it stops and usualy saves the model in its current state.
When you execute this script again, it starts back where it left.
"""

from denserflow import models, layers, optimizers
import numpy as np
import emnist
import os


def cleanup(model):
    """
    Save the model, test its accuracy.
    """
    model.save("./NeuralNexus.json")
    X, y = emnist.test()
    pred = np.argmax(model(X), axis=1)
    print(np.mean(pred == y))


# Load the model if it exists, create it otherwise
if os.path.exists("./NeuralNexus.json"):
    model = models.load_model("./NeuralNexus.json")

else:
    model = models.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(47, activation="softmax")
    ])

    optimizer = optimizers.Adam(decay=1e-4)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy")

# Train until interrupted
try:
    i = 0
    while True:
        for X, y in emnist.training_batched(50000):
            model.fit(X, y, epochs=1)

        if i % 10 == 0:
            print(f"{i} epochs done.")
            model.save(f"./models/continuous_{i}.json")
            cleanup(model)

        i += 1

except KeyboardInterrupt:
    cleanup(model)
