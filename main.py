from denserflow import layers, models
import numpy as np
import emnist
np.random.seed(40)

NeuralNexus = models.Sequential([
    layers.Dense(128, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(47, activation="softmax")
])

for X, y in emnist.training_batched(100):
    NeuralNexus.fit(X, y, learning_rate=0.01, epochs=50)


NeuralNexus.save("./NeuralNexus.json")

NeuralNexus = models.load_model("./NeuralNexus.json")

X, y = emnist.test()
pred = np.argmax(NeuralNexus(X), axis=1)
print(np.mean(pred == y))
