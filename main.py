from denserflow import layers, models
import emnist

for X, y in emnist.training_batched(50000):

    model = models.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(47, activation="softmax")
    ])

    print(model(X).sum())
