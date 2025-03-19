import tkinter as tk
from PIL import Image, ImageTk
from Tkinter_V0 import DrawingApp  # Assure-toi que la classe DrawingApp est définie correctement
import tensorflow as tf

def on_button_click():
    # Charger le modèle (assure-toi que le chemin du modèle est correct)
    model = tf.keras.models.load_model("mon_modele.h5")

    # Créer et lancer l'application de dessin dans cette nouvelle fenêtre
    app = DrawingApp(model)  # On passe le modèle à l'instance de DrawingApp
    app.mainloop()  # Lancer la boucle principale de l'application de dessin

# Créer la fenêtre principale
root = tk.Tk()
root.title("Interface Tkinter avec Image")
root.geometry("500x500")

# Charger et afficher une image       
try:
    image = Image.open("C:/Users/seben/OneDrive/Bureau/R.jpg")
    image = image.resize((500, 350))
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.pack(pady=10)
except Exception as e:
    print(f"Erreur lors du chargement de l'image : {e}")

# Créer un label
label = tk.Label(root, text="Bienvenue!")
label.pack(pady=10)

# Créer un bouton
button = tk.Button(root, text="Cliquez-moi", command=on_button_click)
button.pack()

# Lancer la boucle principale
root.mainloop()
