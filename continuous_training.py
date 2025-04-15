from denserflow import models
import numpy as np
import emnist


def cleanup(model):
    model.save("./NeuralNexus.json")
    X, y = emnist.test()
    pred = np.argmax(model(X), axis=1)
    print(np.mean(pred == y))


NeuralNexus = models.load_model("./NeuralNexus.json")
try:
    α = 0.001
    i = 0
    while True:
        for X, y in emnist.training_batched(50000):
            NeuralNexus.fit(X, y, learning_rate=α, epochs=1)

        if i % 10 == 0:
            print(f"{i} epochs done.")
            NeuralNexus.save(f"./overnight/model_{i}.json")
            cleanup(NeuralNexus)

        i += 1

except KeyboardInterrupt:
    cleanup(NeuralNexus)

cleanup(NeuralNexus)
