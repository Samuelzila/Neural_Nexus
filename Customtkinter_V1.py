from annotated_types import Le
from cv2 import imshow
import image_processing
from networkx import les_miserables_graph
import customtkinter as ctk
from polars import col
import processImg as pm
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import emnist
from denserflow import models
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog

# Configuration de l’apparence
ctk.set_appearance_mode("black")  # ou "light", "system"
ctk.set_default_color_theme("blue")  # Thème global

# === Configuration de la fenêtre principale === #
fenetre = ctk.CTk()
fenetre.title("Reconnaissance de chiffres manuscrits")
fenetre.geometry("700x700")
fenetre._fg_color = 'gray'

main_frame = ctk.CTkFrame(fenetre, width=700, height=700, corner_radius=0,fg_color='gray')
main_frame.pack()

# === Résultats affichés (1, 1) === #
result_frame = ctk.CTkFrame(main_frame, width=350, height=350, corner_radius=10,fg_color='gray')
result_frame.grid(row=0, column=1)

#ctk.CTkLabel(result_frame, text="Prédiction", font=("Arial", 36),text_color='cyan',fg_color='black').pack(expand=True)
result_label = ctk.CTkLabel(result_frame, text="?", font=("Arial", 116), width=350, height=350, text_color='cyan',fg_color='gray')
result_label.pack(expand=True)

# === Canvas pour le dessin (-1, 1) === #
canvas_frame = ctk.CTkFrame(main_frame, width=350, height=350, corner_radius=10,fg_color='gray')
canvas_frame.grid(row=0, column=0)

# === Frame bleu (-1, -1) === #
input_frame = ctk.CTkFrame(main_frame, width=350, height=350, corner_radius=10,fg_color='gray')
input_frame.grid(row=1, column=0)

# === Frame vert (1, -1) === #
statistics_frame = ctk.CTkFrame(main_frame, width=350, height=350, corner_radius=10,fg_color='gray')
statistics_frame.grid(row=1, column=1)



# === Classe pour le dessin et prédiction === #
class DrawingApp(ctk.CTkFrame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model
        self.canvas_width = 350
        self.canvas_height = 350

        self.canvas = ctk.CTkCanvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(fill='both', expand=True)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.bind_events()


    def bind_events(self):
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def draw_on_canvas(self, event):
        radius = 12
        x0, y0 = event.x - radius, event.y - radius
        x1, y1 = event.x + radius, event.y + radius

        self.canvas.create_oval(x0, y0, x1, y1, fill="black", outline="black")
        self.draw.ellipse([x0, y0, x1, y1], fill=0)

        self.last_x, self.last_y = event.x, event.y

    def on_release(self, event):
        self.last_x, self.last_y = None, None
        try:
            self.predict()
        except Exception as e:
            print("Erreur de prédiction :", e)
            self.canvas.after(1000, self.clear_canvas)
            result_label.configure(text="?")
        self.canvas.after(1500, self.clear_canvas)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        

        matrix = image_processing.format_matrix(np.array(self.image))
        prediction = self.model(matrix.reshape(1, 784))
        digit = emnist.label_to_char(np.argmax(prediction))
        result_label.configure(text=f"{digit}")

        probabilities = prediction.flatten()
        labels = [emnist.label_to_char(i) for i in range(len(probabilities))]

        sorted_indices = np.argsort(probabilities)[::-1]
        top_indices = sorted_indices[:6]
        top_probabilities = probabilities[top_indices]
        top_labels = [labels[i] for i in top_indices]

        dpi = 200
        figsize = (350 / dpi, 350 / dpi)
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('gray')

        ax = fig.add_subplot(111)
        ax.set_facecolor('gray')

        segment_colors = ['#00ffe0', '#e40800', '#f8ff69', '#03008e', '#f639ff', '#ff7c41']
        textprops = {'fontsize': 6, 'color': 'black'}
        ax.pie(top_probabilities, labels=top_labels, autopct='%1.1f%%',
               startangle=90, colors=segment_colors, textprops=textprops)
        ax.axis('equal')

        for widget in statistics_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=statistics_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True)

        img_array = np.array(matrix)
        img_resized = Image.fromarray(img_array).resize((350, 350))
        img_tk = ImageTk.PhotoImage(img_resized)

        if hasattr(self, 'blue_frame_label'):
            self.blue_frame_label.configure(image=img_tk)
            self.blue_frame_label.image = img_tk
        else:
            self.blue_frame_label = ctk.CTkLabel(input_frame, image=img_tk, text="")
            self.blue_frame_label.image = img_tk
            self.blue_frame_label.pack(expand=True)

def select_file_customtk():
    # Créer une instance de fenêtre CustomTkinter
    root = ctk.CTk()
    # On peut centrer ou configurer la fenêtre si nécessaire
    root.title("Sélection de fichier")
    
    # On masque la fenêtre principale pour afficher uniquement le dialogue de sélection
    root.withdraw()
    # Ouvrir le dialogue de sélection de fichier
    file_path = filedialog.askopenfilename(
        title="Sélectionnez un fichier",
        filetypes=[
            ("Tous les fichiers", "*.*"),
            ("Fichiers texte", "*.txt"),
            ("Fichiers PDF", "*.pdf")
        ]
    )
    
    # Détruire la fenêtre une fois le fichier sélectionné
    root.destroy()
    
    return file_path
def Anlayse_texte():
    model = models.load_model("NeuralNexus0,87.json")
    image_path = select_file_customtk()
    LesMots = pm.ImgToChar(image_path, 0.1)
    LeTexte = []
    for mots in LesMots:
        LeTexte.append(" ")
        for char in mots:
            matrix = image_processing.format_matrix(np.array(char))
            try:
                prediction = model(matrix.reshape(1, 784))
                print(matrix.shape)
                
            except:
                plt.imshow(matrix, "gray")
                plt.show()
            digit = emnist.label_to_char(np.argmax(prediction))
            LeTexte.append(digit)
        
    
    LeTexte = "".join(LeTexte)
    print(LeTexte)
    
        
        
        

button = ctk.CTkButton(main_frame, text="Sélectionner un fichier", command=Anlayse_texte)
button.grid(row= 0, column = 1)

# === Point d’entrée principal === #
def main():
    model = models.load_model("NeuralNexus0,87.json")
    app = DrawingApp(canvas_frame, model)
    app.pack()

if __name__ == "__main__":
    main()
    fenetre.mainloop()
