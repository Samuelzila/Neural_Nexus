from denserflow import layers, models
import numpy as np
import emnist
np.random.seed(40)

NeuralNexus = models.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(47, activation="softmax")
])

NeuralNexus.compile("adam")

for X, y in emnist.training_batched(10000):
    NeuralNexus.fit(X, y,  epochs=50)
    break

X, y = emnist.test()
pred = np.argmax(NeuralNexus(X), axis=1)
print(np.mean(pred == y))
