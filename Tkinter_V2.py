from tkinter import * 
import string
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from tkinter import messagebox
import tensorflow as tf
import numpy as np

fenetre = Tk()

MainFrame = Frame(fenetre, bg="white", borderwidth=2, relief=GROOVE, width=768, height=576)
MainFrame.pack(padx=10, pady=10)


#canvas pour dessiner 
class DrawingApp(tk.Frame):  # Utiliser Toplevel au lieu de Tk
    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model
        self.canvas_width = 500
        self.canvas_height = 350

        # Création du canevas pour dessiner
        self.canvas = tk.Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()
        
        #bouton_utile
        self.btn_Fermer = Button(self, 
            text="❌ Fermer", 
            command=self.quit,
            bg="white",             
            fg="gray",         
            font=("Arial", 12, "bold"),
            relief="flat",          
            width=12,
            height=2)
        self.btn_Fermer.pack(side='left', expand=True, padx=10, pady=10)
        
        # Effet de surbrillance (néon)
        def on_enter(e):
            self.btn_Fermer.config(bg="#e0f7ff", highlightbackground="#00bfff", highlightthickness=3)
        def on_leave(e):
            self.btn_Fermer.config(bg="white", highlightthickness=0)

        self.btn_Fermer.bind("<Enter>", on_enter)
        self.btn_Fermer.bind("<Leave>", on_leave)

        # Bouton pour prédire le chiffre
        self.btn_predict = tk.Button(self, 
            text="🔎 Prédire", 
            command=self.predict_digit,
            bg="white",             
            fg="gray",           
            font=("Arial", 12, "bold"),
            relief="flat",          
            width=12,
            height=2)
        self.btn_predict.pack(side='left', expand=True, padx=10)
        
        # Effet de surbrillance (néon)
        def on_enter(e):
            self.btn_predict.config(bg="#e0f7ff", highlightbackground="#00bfff", highlightthickness=3)
        def on_leave(e):
            self.btn_predict.config(bg="white", highlightthickness=0)

        self.btn_predict.bind("<Enter>", on_enter)
        self.btn_predict.bind("<Leave>", on_leave)

        # Bouton pour effacer le dessin
        self.btn_clear = tk.Button(self, 
            text="🧹 Effacer", 
            command=self.clear_canvas,
            bg="white",             
            fg="gray",           
            font=("Arial", 12, "bold"),
            relief="flat",
            width=12,
            height=2)
        self.btn_clear.pack(side='left', expand=True, padx=10)
        
        # Effet de surbrillance (néon)
        def on_enter_clear(e):
            self.btn_clear.config(bg="#e0f7ff", highlightbackground="#00bfff", highlightthickness=3)
        def on_leave_clear(e):
            self.btn_clear.config(bg="white", highlightthickness=0)

        self.btn_clear.bind("<Enter>", on_enter_clear)
        self.btn_clear.bind("<Leave>", on_leave_clear)

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
    # Charger le modèle sauvegardé
    model = tf.keras.models.load_model("mon_modele.h5")
    app = DrawingApp(MainFrame, model)  # Passer fenetre comme parent
    app.pack(side=RIGHT)


if __name__ == "__main__":
    main()

fenetre.mainloop()

