import tensorflow as tf
from tensorflow.keras import datasets, layers, models # type: ignore

# Charger le dataset MNIST
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Redimensionner les images pour ajouter la dimension "canal" et normaliser
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Définir l'architecture du modèle CNN
model = models.Sequential([
    # Première couche convolutionnelle avec 32 filtres de taille 3x3
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Deuxième couche convolutionnelle avec 64 filtres
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # Troisième couche convolutionnelle
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    
    # Aplatir les caractéristiques pour les passer à une couche dense
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes pour les chiffres de 0 à 9
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Précision sur le test :', test_acc)

# Sauvegarder le modèle en format HDF5
model.save('mon_modele.h5')

