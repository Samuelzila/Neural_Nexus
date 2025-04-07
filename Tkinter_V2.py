from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import tensorflow as tf
import numpy as np
import os
import string

# Configuration de la fenêtre principale
fenetre = Tk()
fenetre.title("Reconnaissance de chiffres manuscrits")

MainFrame = Frame(fenetre, bg="white", borderwidth=2,
                  relief="ridge", width=768, height=576)
MainFrame.pack(padx=10, pady=10)

# Charger et redimensionner les images


def charger_images():
    images = {}
    for i in range(-15, 13):
        chemin = f"C:/Users/seben/OneDrive/Bureau/Tout/École/ia/projet-d-int-gration-equipe-a/image uncanny/Phase({
            i}).png"
        if os.path.exists(chemin):
            img = Image.open(chemin)
            img.thumbnail((200, 200))
            images[i] = ImageTk.PhotoImage(img)
        else:
            print(f"Image manquante : {chemin}")
    return images


images = charger_images()

# Variables globales
chiffre = 0

# Fonctions principales


def choix(valeur):
    global chiffre
    chiffre += valeur
    print(f"Chiffre actuel : {chiffre}")
    actualiser_image()


def actualiser_image():
    if chiffre in images:
        label_image.config(image=images[chiffre])
    else:
        print(f"Aucune image pour la valeur : {chiffre}")


# Création des widgets
label_image = tk.Label(MainFrame)
label_image.pack(side='right', expand=True, padx=10, pady=10)

result_frame = Frame(MainFrame, bg="white", borderwidth=2,
                     relief=GROOVE, width=200, height=350)
result_frame.pack(side='right', fill=Y, expand=True, padx=10, pady=10)

result_label1 = Label(result_frame, text="Prédiction",
                      font=("Arial", 36), bg="white", fg="black")
result_label1.pack(side='top', expand=True, pady=10, padx=10)

result_label2 = Label(result_frame, text=" ? ", font=(
    "Arial", 116), bg="white", fg="black")
result_label2.pack(expand=True, pady=10, padx=10)

# Création des boutons Vrai/Faux


def creer_bouton_choix(texte, couleur, valeur, parent_frame):
    bouton = Button(parent_frame, text=texte, command=lambda: choix(
        valeur), bg="white", fg="black", font=(12), relief="ridge", width=6, height=2)
    bouton.pack(side='left' if valeur == -1 else 'right',
                expand=True, padx=10, pady=10)

    def on_enter(e):
        bouton.config(fg="black" if valeur == -1 else "black",
                      bg=couleur, highlightbackground="black")

    def on_leave(e):
        bouton.config(fg="black", bg="white", highlightthickness=0)

    bouton.bind("<Enter>", on_enter)
    bouton.bind("<Leave>", on_leave)


creer_bouton_choix("Vrai", "lime", -1, result_frame)
creer_bouton_choix("Faux", "red", 1, result_frame)

# Classe pour le canvas


class DrawingApp(tk.Frame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model
        self.canvas_width = 500
        self.canvas_height = 350

        self.canvas = tk.Canvas(
            self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        self.creer_boutons()
        self.lier_evenements()

        self.image1 = Image.new(
            "L", (self.canvas_width, self.canvas_height), color=255)
        self.draw1 = ImageDraw.Draw(self.image1)
        self.last_x, self.last_y = None, None

    def creer_boutons(self):
        self.creer_bouton("❌ Fermer", self.quit, "red", "black")
        self.creer_bouton("🔎 Prédire", self.predict_digit, "lime", "white")
        self.creer_bouton("🧹 Effacer", self.clear_canvas, "blue", "yellow")

    def creer_bouton(self, texte, commande, couleur_bg, couleur_fg):
        bouton = Button(self, text=texte, command=commande, bg="white",
                        fg="black", font=(12), relief="ridge", width=12, height=2)
        bouton.pack(side='left', expand=True, padx=10, pady=10)

        def on_enter(e):
            bouton.config(
                bg=couleur_bg, highlightbackground="black", fg=couleur_fg)

        def on_leave(e):
            bouton.config(bg="white", highlightthickness=0, fg="black")

        bouton.bind("<Enter>", on_enter)
        bouton.bind("<Leave>", on_leave)

    def lier_evenements(self):
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def draw(self, event):
        x, y = event.x, event.y
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y, fill="black", width=8)
            self.draw1.line([self.last_x, self.last_y, x, y], fill=0, width=8)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new(
            "L", (self.canvas_width, self.canvas_height), color=255)
        self.draw1 = ImageDraw.Draw(self.image1)
        result_label2.config(text="?")

    def predict_digit(self):
        import image_processing

        matrix = image_processing.format_matrix(np.array(self.image1))
        matrix = matrix.reshape(1, 28, 28)

        # Faire la prédiction avec le modèle
        prediction = self.model.predict(matrix)
        predicted_digit = np.argmax(prediction)

        # Afficher le résultat
        messagebox.showinfo(
            "Résultat", f"Le chiffre prédit est : {predicted_digit}")

# Initialisation


def main():
    model = tf.keras.models.load_model("mon_modele.h5")
    app = DrawingApp(MainFrame, model)
    app.pack(side=RIGHT)


if __name__ == "__main__":
    main()
    fenetre.mainloop()
