import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf

class DrawingApp(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.title("Reconnaissance de Chiffres")
        self.model = model
        self.canvas_width = 200
        self.canvas_height = 200

        # Création du canevas pour dessiner
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(padx=10, pady=10)

        # Bouton pour prédire le chiffre
        self.btn_predict = tk.Button(self, text="Prédire", command=self.predict_digit)
        self.btn_predict.pack(side='left', padx=10)

        # Bouton pour effacer le dessin
        self.btn_clear = tk.Button(self, text="Effacer", command=self.clear_canvas)
        self.btn_clear.pack(side='right', padx=10)

        # Liaison de l'événement de dessin
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Création d'une image PIL pour pouvoir la traiter ensuite
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw1 = ImageDraw.Draw(self.image1)
        self.last_x, self.last_y = None, None

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            # Dessiner sur le canevas Tkinter
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=8)
            # Dessiner sur l'image PIL
            self.draw1.line([self.last_x, self.last_y, x, y], fill=0, width=8)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw1 = ImageDraw.Draw(self.image1)

    def predict_digit(self):
        # Prétraitement de l'image :
        # Redimensionner l'image en 28x28 pixels
        img = self.image1.resize((28, 28))
        # Convertir en tableau numpy
        img_array = np.array(img)
        # Inverser les couleurs pour correspondre aux images MNIST (fond noir, chiffre blanc)
        img_array = 255 - img_array
        # Normaliser les valeurs entre 0 et 1
        img_array = img_array / 255.0
        # Reshape pour le modèle (1, 28, 28)
        img_array = img_array.reshape(1, 28, 28)

        # Faire la prédiction avec le modèle
        prediction = self.model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Afficher le résultat
        messagebox.showinfo("Résultat", f"Le chiffre prédit est : {predicted_digit}")

def main():
    # Charger le modèle sauvegardé (par exemple 'mon_modele.h5')
    model = tf.keras.models.load_model("mon_modele.h5")
    app = DrawingApp(model)
    app.mainloop()

if __name__ == "__main__":
    main()
