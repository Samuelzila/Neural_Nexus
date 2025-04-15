import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import emnist
from denserflow import models
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# === Configuration de la fenêtre principale === #
fenetre = tk.Tk()
fenetre.title("Reconnaissance de chiffres manuscrits")

main_frame = tk.Frame(fenetre, bg="black", borderwidth=0, relief="ridge", width=700, height=700)
main_frame.grid(padx=10, pady=10)

# === Résultats affichés (1, 1) === #
result_frame = tk.Frame(main_frame, bg="black", borderwidth=0, relief="ridge", width=350, height=350)
result_frame.grid(row=0, column=1)

tk.Label(result_frame, text="Prédiction", font=("Arial", 36), bg="black", fg="cyan").pack(side='top')
result_label = tk.Label(result_frame, text=" ? ", font=("Arial", 116), bg="black", fg="cyan")
result_label.pack(expand=True)

# === Canvas pour le dessin (-1, 1) === #
canvas_frame = tk.Frame(main_frame, bg="gray", borderwidth=0, relief="ridge", width=350, height=350)
canvas_frame.grid(row=0, column=0)

# === Frame bleu (-1, -1) === #
input_frame = tk.Frame(main_frame, bg="black", borderwidth=0, relief="ridge", width=350, height=350)
input_frame.grid(row=1, column=0)

# === Frame vert (1, -1) === #
statistics_frame = tk.Frame(main_frame, bg="black", borderwidth=0, relief="ridge", width=350, height=350)
statistics_frame.grid(row=1, column=1)

# === Classe pour le dessin et prédiction === #
class DrawingApp(tk.Frame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model
        self.canvas_width = 350
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
        radius = 16  # rayon du cercle, donc diamètre = 32 comme la largeur de ta ligne
        x0, y0 = event.x - radius, event.y - radius
        x1, y1 = event.x + radius, event.y + radius

        # Dessine sur le canvas tkinter
        self.canvas.create_oval(x0, y0, x1, y1, fill="black", outline="black")

        # Dessine sur l'image (noir = 0 si image en niveaux de gris)
        self.draw.ellipse([x0, y0, x1, y1], fill=0)

        self.last_x, self.last_y = event.x, event.y

    def on_release(self, event):
        self.last_x, self.last_y = None, None
        try:
            self.predict()
        except:
            self.canvas.after(1000, self.clear_canvas)
            result_label.config(text="?")  # Ignore les erreurs de prédiction si l'image est vide
        self.canvas.after(1500, self.clear_canvas)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        import image_processing  # Import local pour éviter les imports inutiles au lancement

        # Convertir l'image en matrice et la reformater
        matrix = image_processing.format_matrix(np.array(self.image))

        # Faire la prédiction
        prediction = self.model(matrix.reshape(1, 784))
        digit = emnist.label_to_char(np.argmax(prediction))
        result_label.config(text=f"{digit}")

        # Accéder aux probabilités de sortie
        probabilities = prediction.flatten()  # Convertir en tableau 1D si nécessaire
        labels = [emnist.label_to_char(i) for i in range(len(probabilities))]  # Générer des étiquettes pour chaque chiffre (0 à 9)

        # Trier les probabilités et les étiquettes par ordre décroissant
        sorted_indices = np.argsort(probabilities)[::-1]  # Indices triés par ordre décroissant
        top_indices = sorted_indices[:6]  # Sélectionner les 6 plus grandes probabilités
        top_probabilities = probabilities[top_indices]
        top_labels = [labels[i] for i in top_indices]

        # Créer un diagramme circulaire avec des couleurs personnalisées
        dpi = 200
        figsize = (350 / dpi, 350 / dpi)  # Taille de la figure en pouces
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('black')  # Couleur de fond de la figure

        ax = fig.add_subplot(111)
        ax.set_facecolor('black')  # Couleur de fond de l'axe

        # Couleurs personnalisées pour les segments du pie chart
        segment_colors = ['#00ffff', '#e40800', '#f8ff69', '#03008e', '#f639ff', '#ff7c41']

        textprops = {'fontsize': 6, 'color': 'gray'}  # Propriétés du texte
        ax.pie(top_probabilities, labels=top_labels, autopct='%1.1f%%', startangle=90, colors=segment_colors, textprops=textprops) # Afficher les pourcentages sur le pie chart
        ax.axis('equal')  # Assurer que le pie chart est circulaire

        # Afficher le diagramme dans le frame vert (statistics_frame)
        for widget in statistics_frame.winfo_children():
            widget.destroy()  # Supprimer les anciens widgets dans le frame vert

        canvas = FigureCanvasTkAgg(fig, master=statistics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True)

        # Convertir la matrice en image pour l'afficher dans le frame bleu (input_frame)
        img_array = np.array(matrix)  # Convertir l'image PIL en tableau numpy
        img_resized = Image.fromarray(img_array).resize((350, 350))  # Redimensionner pour le frame bleu
        img_tk = ImageTk.PhotoImage(img_resized)

        # Mettre à jour l'image dans le frame bleu (frame input)
        if hasattr(self, 'blue_frame_label'):
            self.blue_frame_label.config(image=img_tk)
            self.blue_frame_label.image = img_tk  # Conserver une référence pour éviter le garbage collection
        else:
            self.blue_frame_label = tk.Label(input_frame, image=img_tk, bg="black")
            self.blue_frame_label.image = img_tk
            self.blue_frame_label.pack(expand=True)

# === Point d'entrée principal === #
def main():
    model = models.load_model("NeuralNexus0,87.json")
    app = DrawingApp(canvas_frame, model)
    app.pack()

if __name__ == "__main__":
    main()
    fenetre.mainloop()
