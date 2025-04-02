from denserflow import layers, models
import numpy as np
import emnist

for X, y in emnist.training_batched(100):
    break


NeuralNexus = models.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(47, activation="softmax")
])

NeuralNexus.fit(X, y, learning_rate=0.01, epochs=100)

for X, y in emnist.test_batched(100):
    break
pred = np.argmax(NeuralNexus(X), axis=1)
print(np.mean(pred == y))
