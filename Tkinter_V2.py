from tkinter import * 
import string
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import tensorflow as tf
import numpy as np
import os


fenetre = Tk()
fenetre.title("Reconnaissance de chiffres manuscrits")

MainFrame = Frame(fenetre, bg="white", borderwidth=2, relief="ridge", width=768, height=576)
MainFrame.pack(padx=10, pady=10)

# Charger et redimensionner les images avec `thumbnail()`  "C:\Users\seben\OneDrive\Bureau\Tout\École\ia\projet-d-int-gration-equipe-a\denserflow\image\Phase(-15).png"
def charger_images():
    images = {}
    for i in range(-15, 13):  # Suppose que tu as des images de -10 à 10
        chemin = f"C:/Users/seben/OneDrive/Bureau/Tout/École/ia/projet-d-int-gration-equipe-a/denserflow/image/Phase({i}).png"
        if os.path.exists(chemin):
            # Charger l'image avec Pillow
            img = Image.open(chemin)
            
            # Redimensionner l'image à une taille maximale tout en maintenant l'aspect ratio
            img.thumbnail((200, 200))  # Taille maximale de 400x400 pixels
            images[i] = ImageTk.PhotoImage(img)  # Convertir l'image pour Tkinter
        else:
            print(f"Image manquante : {chemin}")
    return images

images = charger_images()

# Variable globale
chiffre = 0

# Fonction pour gérer le choix Vrai/Faux et mettre à jour l'image
def choix(valeur):
    global chiffre
    chiffre += valeur
    print(f"Chiffre actuel : {chiffre}")
    actualiser_image()

# Fonction pour actualiser l'image
def actualiser_image():
    if chiffre in images:
        label_image.config(image=images[chiffre])
    else:
        print(f"Aucune image pour la valeur : {chiffre}")
        
# Créer un label pour afficher l'image
label_image = tk.Label(MainFrame)
label_image.pack(side='right', expand=True, padx=10, pady=10)

# Zone d'affichage du résultat
result_frame = Frame(MainFrame, bg="white", borderwidth=2, relief=GROOVE, width=200, height=350)
result_frame.pack(side='right',fill=Y, expand=True, padx=10, pady=10)
result_label1 = Label(result_frame, text="Prédiction", font=("Arial", 36), bg="white", fg="black")
result_label1.pack(side='top', expand=True, pady=10, padx=10)
result_label2 = Label(result_frame, text=" ? ", font=("Arial", 116), bg="white", fg="black")
result_label2.pack(expand=True, pady=10, padx=10)

bouton_vrai = Button(result_frame,
            command=lambda: choix(-1),
            text="Vrai", 
            bg="white",             
            fg="gray",           
            font=(12),
            relief="ridge",          
            width=6,
            height=2)
bouton_vrai.pack(side='left', expand=True, padx=10, pady=10)

# Effet de surbrillance (néon)
def on_enter(e):
    bouton_vrai.config(fg="white", bg="green", highlightbackground="black")
def on_leave(e):
    bouton_vrai.config(fg="gray", bg="white", highlightthickness=0)
    
bouton_vrai.bind("<Enter>", on_enter)
bouton_vrai.bind("<Leave>", on_leave)

bouton_faux = Button(result_frame,
            command=lambda: choix(+1),
            text="Faux", 
            bg="white",             
            fg="gray",           
            font=(12),
            relief="ridge",          
            width=6,
            height=2)
bouton_faux.pack(side='right', expand=True, padx=10, pady=10)

# Effet de surbrillance (néon)
def on_enter(e):
    bouton_faux.config(fg="black", bg="red", highlightbackground="black")
def on_leave(e):
    bouton_faux.config(fg="gray", bg="white", highlightthickness=0)

bouton_faux.bind("<Enter>", on_enter)
bouton_faux.bind("<Leave>", on_leave)

# Canvas pour dessiner 
class DrawingApp(tk.Frame):
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
            font=(12),
            relief="ridge",          
            width=12,
            height=2)
        self.btn_Fermer.pack(side='left', expand=True, padx=10, pady=10)
        
        # Effet de surbrillance (néon)
        def on_enter(e):
            self.btn_Fermer.config(fg="black", bg="red", highlightbackground="black")
        def on_leave(e):
            self.btn_Fermer.config(fg="gray", bg="white", highlightthickness=0)

        self.btn_Fermer.bind("<Enter>", on_enter)
        self.btn_Fermer.bind("<Leave>", on_leave)

        # Bouton pour prédire le chiffre
        self.btn_predict = tk.Button(self, 
            text="🔎 Prédire", 
            command=self.predict_digit,
            bg="white",             
            fg="gray",           
            font=(12),
            relief="ridge",          
            width=12,
            height=2)
        self.btn_predict.pack(side='left', expand=True, padx=10, pady=10)
        
        # Effet de surbrillance (néon)
        def on_enter(e):
            self.btn_predict.config(fg="white", bg="green", highlightbackground="white")
        def on_leave(e):
            self.btn_predict.config(fg="gray", bg="white", highlightthickness=0)

        self.btn_predict.bind("<Enter>", on_enter)
        self.btn_predict.bind("<Leave>", on_leave)

        # Bouton pour effacer le dessin
        self.btn_clear = tk.Button(self, 
            text="🧹 Effacer", 
            command=self.clear_canvas,
            bg="white",             
            fg="gray",           
            font=(12),
            relief="ridge",
            width=12,
            height=2)
        self.btn_clear.pack(side='left', expand=True, padx=10, pady=10)
        
        # Effet de surbrillance (néon)
        def on_enter_clear(e):
            self.btn_clear.config(fg="yellow", bg="blue", highlightbackground="white")
        def on_leave_clear(e):
            self.btn_clear.config(fg="gray", bg="white", highlightthickness=0)

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
            self.canvas.create_line(self.last_x, self.last_y, x, y, fill="black", width=8)
            self.draw1.line([self.last_x, self.last_y, x, y], fill=0, width=8)
        self.last_x, self.last_y = x, y

    def reset(self, event):
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw1 = ImageDraw.Draw(self.image1)
        result_label2.config(text="?")

    def predict_digit(self):
        img = self.image1.resize((28, 28))
        img_array = np.array(img)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = self.model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Afficher le résultat dans le Label
        result_label2.config(text=f"{predicted_digit}")

def main():
    model = tf.keras.models.load_model("mon_modele.h5")
    app = DrawingApp(MainFrame, model)
    app.pack(side=RIGHT)

if __name__ == "__main__":
    main()

fenetre.mainloop()
