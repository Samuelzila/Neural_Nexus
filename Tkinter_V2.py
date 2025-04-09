import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import os
import string
import time

import emnist
import denserflow as tf
from denserflow import models

# === Configuration de la fenêtre principale === #
fenetre = tk.Tk()
fenetre.title("Reconnaissance de chiffres manuscrits")

main_frame = tk.Frame(fenetre, bg="white", borderwidth=2, relief="ridge", width=768, height=576)
main_frame.pack(padx=10, pady=10)

# === Résultats affichés === #
result_frame = tk.Frame(main_frame, bg="white", borderwidth=2, relief="groove", width=200, height=350)
result_frame.pack(side='right', fill='y', expand=True, padx=10, pady=10)

tk.Label(result_frame, text="Prédiction", font=("Arial", 36), bg="white", fg="black").pack(side='top', pady=10)
result_label = tk.Label(result_frame, text=" ?", font=("Arial", 116), bg="white", fg="black")
result_label.pack(expand=True, pady=10)

# === Classe pour le dessin et prédiction === #
class DrawingApp(tk.Frame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model
        self.canvas_width = 500
        self.canvas_height = 350

        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.bind_events()

    def bind_events(self):
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def draw_on_canvas(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="black", width=32)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=32)
        self.last_x, self.last_y = event.x, event.y

    def on_release(self, event):
        self.last_x, self.last_y = None, None
        try: self.predict()
        except: 
            self.canvas.after(1000, self.clear_canvas)
            result_label.config(text="?")    # Ignore les erreurs de prédiction si l'image est vide
        self.canvas.after(1000, self.clear_canvas)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        import image_processing  # Import local pour éviter les imports inutiles au lancement
        matrix = image_processing.format_matrix(np.array(self.image)).reshape(1, 784)
        prediction = self.model(matrix)
        digit = emnist.label_to_char(np.argmax(prediction))
        result_label.config(text=f"{digit}")

# === Point d'entrée principal === #
def main():
    model = models.load_model("NeuralNexus0,87.json")
    app = DrawingApp(main_frame, model)
    app.pack(side='right')

if __name__ == "__main__":
    main()
    fenetre.mainloop()
