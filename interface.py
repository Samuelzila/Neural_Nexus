import customtkinter as ctk
from customtkinter import CTkImage  # Assurez-vous que CTkImage est importé
from PIL import Image, ImageDraw
import numpy as np
import emnist
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import image_processing

# ======================================================================
# === Configuration de l’apparence === #

# Fonts
font_Type_default = 'Roboto'  # Police par défaut
# Poids de la police pour le label du modèle et du temps de calcul
font_Weight_info_model_temps = 'bold'
font_Size_info_model_temps = 16
font_Size_result_Prediction_en_attente = 64
font_Size_result2_Ressayer = 112
font_Size_result3_Reponse = 202
font_Size_graph_catégorie = 6

# Couleurs
couleur1 = '#AA0505'  # rouge
couleur3 = '#6A0C0B'  # rouge clair
couleur2 = '#FBCA03'  # jaune

# Configuration de l'apparence de CustomTkinter
Page_wight = 1024
Page_height = 768  # 1024/2 + 1024/4
Page_grid_row = 16
Page_grid_column = 16

# ======================================================================

# Configuration de l’apparence
ctk.set_appearance_mode("black")  # ou "light", "system"
ctk.set_default_color_theme("blue")  # Thème global

# === Configuration de la fenêtre principale === #
fenetre = ctk.CTk()
fenetre.title("Reconnaissance de chiffres manuscrits")
fenetre._fg_color = couleur3

# main frame (big picture of the app) dimension = 1024x768   #checked
main_frame = ctk.CTkFrame(
    fenetre,
    width=1024,
    height=768,
    fg_color=couleur1
)
main_frame.pack(expand=False)  # Remplir l'espace disponible

# Configurer une grille 16x16 pour main_frame   #checked
for i in range(16):
    main_frame.grid_rowconfigure(i, weight=1)  # Chaque ligne a un poids égal
    # Chaque colonne a un poids égal
    main_frame.grid_columnconfigure(i, weight=1)

# === Résultats affichés (1, 1) === #    #checked
# dimension = 512x384
result_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=384,
    fg_color=couleur1
)
result_frame.grid(row=0, column=8, rowspan=8, columnspan=8,
                  padx=10, pady=10)  # Ligne 39

# === Label pour le résultat === #   #checked
# dimension = 512x384
result_label = ctk.CTkLabel(
    result_frame,
    text="Prediction\nen attente",  # Utiliser \n pour le saut de ligne
    font=(font_Type_default, 64, "bold"),
    width=512,
    height=384,
    text_color=couleur2,
    corner_radius=30,
    fg_color=couleur3
)
result_label.pack(expand=False)

# === Canvas pour le dessin (-1, 1) === # #checked
# dimension = 512x3(768/4)
canvas_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=(3*(768/4)),
    fg_color=couleur1,
    corner_radius=30
)
canvas_frame.grid(row=0, column=0, rowspan=12, columnspan=8, padx=10, pady=10)

# === Frame vert (1, -1) === # #checked
# dimension = 512x 3(768/8)
statistics_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=336,
    corner_radius=30,
    fg_color=couleur3
)
statistics_frame.grid(row=8, column=8, rowspan=7,
                      columnspan=8, padx=10, pady=10)

# === Frame bleu (-1, -1) === # #checked   mais je ne comprend pas pourquoi toute les autre frame ne sont pas affecter par la fonction corner_radius mais cette frame la oui
# dimension = 512x(768/4)
input_frame = ctk.CTkFrame(
    main_frame,
    width=256,
    height=(768/4),
    corner_radius=30,
    fg_color=couleur3
)
input_frame.grid(row=12, column=4, rowspan=4, columnspan=4, padx=10, pady=10)

# === Frame Menu === # # #checked
# dimension = 256x(768/4)
menu_frame = ctk.CTkFrame(
    main_frame,
    width=256,
    height=(768/4),
    corner_radius=30,
    fg_color=couleur3
)
menu_frame.grid(row=12, column=0, rowspan=4, columnspan=4, padx=10, pady=10)

# === Label pour le nom du model === #  #checked
# dimension = 256x(768/16)
model_label = ctk.CTkLabel(
    main_frame,
    text="Model : NeuralNexus0.87",
    font=(font_Type_default, font_Size_info_model_temps, "bold"),
    width=256,
    height=(768/16),
    text_color=couleur2,
    corner_radius=20,
    fg_color=couleur3
)
model_label.grid(row=15, column=8, rowspan=1, columnspan=4, padx=10, pady=10)

# === Label pour le temps de calcul de la prediciton === # #checked
# dimension = 256x(768/16)
time_label = ctk.CTkLabel(
    main_frame,
    text="Temps de calcul : 0.0s",
    font=(font_Type_default, font_Size_info_model_temps, "bold"),
    width=256,
    height=(768/16),
    text_color=couleur2,
    corner_radius=20,
    fg_color=couleur3
)
time_label.grid(row=15, column=12, rowspan=1, columnspan=4, padx=10, pady=10)

# === Classe pour le dessin et prédiction === #


class DrawingApp(ctk.CTkFrame):
    def __init__(self, parent, model):  # checked-ish
        super().__init__(parent)
        self.model = model
        self.canvas_width = 572
        self.canvas_height = 606

        self.canvas = ctk.CTkCanvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=couleur1
        )
        self.canvas.pack(fill='both', expand=False, padx=10, pady=10)

        self.image = Image.new(
            "L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.bind_events()

    # Fonction pour lier les événements de la souris au canvas #checked-ish
    def bind_events(self):  # checked-ish
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    # Fonction appelée lorsque le bouton de la souris est enfoncé
    def draw_on_canvas(self, event):  # updated + checked
        radius = 30
        x0, y0 = event.x - radius, event.y - radius
        x1, y1 = event.x + radius, event.y + radius

        self.canvas.create_oval(x0, y0, x1, y1, fill="white", outline="white")
        self.draw.ellipse([x0, y0, x1, y1], fill=0)

        # Si un point précédent existe, dessiner une ligne entre les deux points
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y, fill="white", width=radius * 2)
            self.draw.line([self.last_x, self.last_y, event.x,
                           event.y], fill=0, width=radius * 2)

        # Mettre à jour les coordonnées du dernier point
        self.last_x, self.last_y = event.x, event.y

    # Fonction appelée lorsque le bouton de la souris est relâché
    def on_release(self, event):  # checked
        self.last_x, self.last_y = None, None
        try:
            self.predict()
        except Exception as e:
            print("Erreur de prédiction :", e)
            self.canvas.after(1000, self.clear_canvas)
            result_label.configure(text="Nope.", font=(
                font_Type_default, 90), text_color=couleur2, fg_color=couleur3)
        self.canvas.after(1500, self.clear_canvas)

    def clear_canvas(self):  # checked-ish
        self.canvas.delete("all")
        self.image = Image.new(
            "L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Mesurer le temps de début
        start_time = time.time()

        # Préparer l'image pour la prédiction
        matrix = image_processing.format_matrix(np.array(self.image))

        prediction = self.model(matrix.reshape(1, 784))

        digit = emnist.label_to_char(np.argmax(prediction))

        result_label.configure(text=f"{digit}", font=(
            "Comic sans ms", 202), text_color=couleur2, fg_color=couleur3)

        # Calculer le temps de fin
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Mettre à jour le time_label avec le temps de calcul
        time_label.configure(text=f"Temps de calcul : {elapsed_time:.2f}s")

        # Accéder aux probabilités de sortie
        probabilities = prediction.flatten()
        labels = [emnist.label_to_char(i) for i in range(len(probabilities))]

        # Trier les probabilités et les étiquettes par ordre décroissant
        sorted_indices = np.argsort(probabilities)[::-1]
        top_indices = sorted_indices[:6]
        top_probabilities = probabilities[top_indices]
        top_labels = [labels[i] for i in top_indices]

        # Créer un diagramme circulaire
        dpi = 200
        figsize = (610 / dpi, 380 / dpi)
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(couleur3)

        ax = fig.add_subplot(111)
        ax.set_facecolor(couleur3)

        segment_colors = ['#D4342D', '#F3DDAC',
                          '#AA2822', '#E7BF71', '#8E0F06', '#C09645']
        textprops = {'fontsize': 6, 'color': 'black'}
        ax.pie(top_probabilities, labels=top_labels, autopct='%1.1f%%',
               startangle=90, colors=segment_colors, textprops=textprops)
        ax.axis('equal')

        # Afficher le diagramme dans le frame vert (statistics_frame)
        for widget in statistics_frame.winfo_children():
            widget.destroy()

        graph_frame = ctk.CTkFrame(
            statistics_frame,
            width=512,
            height=336,
            fg_color="transparent",
            corner_radius=0
        )
        graph_frame.pack(fill="both", padx=10, pady=10, expand=False)

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=False)

        # Convertir la matrice en image pour l'afficher dans le frame bleu (input_frame)
        img_array = np.array(matrix)
        # les dimensions sont durs à bien estimer si jamais y'a une a,mélioration à faire je suis preneur
        img_resized = Image.fromarray(img_array).resize((236, 172))

        # Utiliser CTkImage au lieu de ImageTk.PhotoImage
        # les dimensions sont durs à bien estimer si jamais y'a une a,mélioration à faire je suis preneur
        img_tk = CTkImage(light_image=img_resized, size=(236, 172))

        if hasattr(self, 'blue_frame_label'):
            self.blue_frame_label.configure(
                image=img_tk, text="", fg_color=couleur3)
        else:
            self.blue_frame_label = ctk.CTkLabel(
                input_frame,
                image=img_tk,
                text="",
                fg_color=couleur3
            )
            self.blue_frame_label.pack(padx=10, pady=10, expand=False)
