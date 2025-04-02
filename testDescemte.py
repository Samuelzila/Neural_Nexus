from denserflow import layers, models
import numpy as np
import emnist
np.random.seed(40)
α = 0.01

for X, y in emnist.training_batched(100):
    break


# print("Weights avant entraînement :", model.layers[0].weights)
# print("Biases avant entraînement :", model.layers[0].biaises)

#

# Après entraînement
# print("Weights après entraînement :", model.layers[0].weights)
# print("Biases après entraînement :", model.layers[0].biaises)
#
dense1 = layers.Dense(128, activation="relu")
dense2 = layers.Dense(128, activation="relu")
dense3 = layers.Dense(47, activation="softmax")
NeuralNexus = models.Sequential([dense1, dense2, dense3])
NeuralNexus.fit(X, y, learning_rate=0.01, epochs=100)

CCE = layers.CategoricalCrossEntropy()


# for _ in range(100):
#     dense1(X)
#     dense2(dense1.output)
#     dense3(dense2.output)
#
#     loss = CCE(dense3.output, y)
#     print("loss = ", loss.mean())
#
#     grad_lo = CCE.backpropagation(dense3.output, y)
#
#     ac = dense3.activation.backpropagation(grad_lo)
#     dense3.backpropagation(ac, α)
#
#     dactivation2 = dense2.activation.backpropagation(dense3.dinputs)
#
#     dense2.backpropagation(dactivation2, α)
#
#     dactivation1 = dense1.activation.backpropagation(dense2.dinputs)
#
#     dense1.backpropagation(dactivation1, α)
#
# model = models.Sequential([dense1, dense2, dense3])

for X, y in emnist.test_batched(100):
    break
pred = np.argmax(NeuralNexus(X), axis=1)
print(np.mean(pred == y))
