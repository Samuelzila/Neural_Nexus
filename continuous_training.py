from denserflow import models, layers, optimizers
import numpy as np
import emnist


def cleanup(model):
    model.save("./NeuralNexus.json")
    X, y = emnist.test()
    pred = np.argmax(model(X), axis=1)
    print(np.mean(pred == y))


NeuralNexus = models.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(47, activation="softmax")
])

# NeuralNexus = models.load_model("./NeuralNexus.json")

optimizer = optimizers.Adam(decay=1e-4)
NeuralNexus.compile(optimizer=optimizer)
try:
    i = 0
    while True:
        for X, y in emnist.training_batched(50000):
            NeuralNexus.fit(X, y, epochs=1)

        if i % 10 == 0:
            print(f"{i} epochs done.")
            NeuralNexus.save(f"./overnight/model_{i}.json")
            cleanup(NeuralNexus)

        i += 1

except KeyboardInterrupt:
    cleanup(NeuralNexus)

cleanup(NeuralNexus)
