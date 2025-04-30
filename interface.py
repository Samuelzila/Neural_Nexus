import customtkinter as ctk
# Assurez-vous que CTkImage est importé
from customtkinter import CTkImage, CTkButton
from PIL import Image, ImageDraw
import numpy as np
import emnist
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import image_processing

# ======================================================================
# === Style variables === #

# Fonts
font_Type_default = 'Roboto'  # Police par défaut

font_Weight_info_model_temps = 'bold'
font_Size_info_model_temps = 16
font_Size_result_Prediction_en_attente = 64
font_Size_result2_Ressayer = 112
font_Size_result3_Reponse = 202
font_Size_graph_catégorie = 6

# Colours
color1 = '#AA0505'  # red
color2 = '#FBCA03'  # yellow
color3 = '#6A0C0B'  # dark red

# Tkinter settings
page_width = 1024
page_height = 768  # 1024/2 + 1024/4
page_grid_row = 16
page_grid_column = 16

# ======================================================================

# Style configuration
ctk.set_appearance_mode("black")  # Dark theme
ctk.set_default_color_theme("blue")  # Global theme

# === Main window === #
fenetre = ctk.CTk(color3)
fenetre.title("Reconnaissance de chiffres manuscrits")

# Main frame (big picture of the app) dimensions = 1024x768   #checked
main_frame = ctk.CTkFrame(
    fenetre,
    width=1024,
    height=768,
    fg_color=color1,
)
main_frame.pack(expand=True)  # Center with available space

# Configure 16x16 grid inside the main frame
for i in range(16):
    main_frame.grid_rowconfigure(i, weight=1)
    main_frame.grid_columnconfigure(i, weight=1)

# ===  Displayed results (1, 1) === #    #checked
# dimensions = 512x384
result_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=384,
    fg_color=color1
)
result_frame.grid(row=0, column=8, rowspan=8, columnspan=8,
                  padx=10, pady=10)  # 39th row

# === Result label === #   #checked
# dimensions = 512x384
result_label = ctk.CTkLabel(
    result_frame,
    text="Prediction\nen attente",
    font=(font_Type_default, 64, "bold"),
    width=512,
    height=384,
    text_color=color2,
    corner_radius=30,
    fg_color=color3
)
result_label.pack(expand=False)

# === Drawing canvas (-1, 1) === #checked
# dimensions = 512x3(768/4)
canvas_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=(3*(768/4)),
    fg_color=color1,
    corner_radius=30
)
canvas_frame.grid(row=0, column=0, rowspan=12, columnspan=8, padx=10, pady=10)

# === Statistics frame (1, -1) === #checked
# dimension = 512x 3(768/8)
statistics_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=336,
    corner_radius=30,
    fg_color=color3
)
statistics_frame.grid(row=8, column=8, rowspan=7,
                      columnspan=8, padx=10, pady=10)

# ===  AI input frame (preview) (-1, -1) === #checked
# dimension = 512x(768/4)
input_frame = ctk.CTkFrame(
    main_frame,
    width=256,
    height=(768/4),
    corner_radius=30,
    fg_color=color3
)
input_frame.grid(row=12, column=4, rowspan=4, columnspan=4, padx=10, pady=10)

# === Menu frame === # checked
# dimension = 256x(768/4)
menu_frame = ctk.CTkFrame(
    main_frame,
    width=256,
    height=(768/4),
    corner_radius=30,
    fg_color=color3
)
menu_frame.grid(row=12, column=0, rowspan=4, columnspan=4, padx=10, pady=10)

# Submenus
options = {
    "Changer de modele": ["Modèle 1", "Modèle 2", "Modèle 3", "Modèle 4", "Modèle 5", "Modèle 6"],
    "Changer couleur": ["Rouge", "Bleu", "Vert"],
    "Changer le font": ["Arial", "Roboto", "Comic Sans"],
    "Changer la taille": ["Petit", "Moyen", "Grand"],
    "Changer la police": ["Gras", "Italique", "Souligné"]
}
# Liste pour stocker les boutons dynamiques
listes_boutons_dynamiques = []

app_instance = None  # Variable globale pour stocker l'instance de DrawingApp

def set_app_instance(app):

    #Définit l'instance globale de DrawingApp.

    global app_instance
    app_instance = app

def fonction_bouton_modele(sous_option, label):
    match sous_option:
        case "Modèle 1":
            app_instance.set_model(0, label)
        case "Modèle 2":
            app_instance.set_model(1, label)
        case "Modèle 3":
            app_instance.set_model(2, label)
        case "Modèle 4":
            app_instance.set_model(3, label)
        case "Modèle 5":
            app_instance.set_model(4, label)
        case "Modèle 6":
            app_instance.set_model(5, label)
            
    afficher_Menu()





def clear_menu():
    for bouton in listes_boutons_dynamiques:
        bouton.destroy()
    listes_boutons_dynamiques.clear()


def afficher_sous_options(option):
    clear_menu()
# Ajouter les sous-options
    for sous_option in options[option]:
        bouton_sous_option = ctk.CTkButton(
            menu_frame,
            text=sous_option,
            #command=afficher_Menu,  # Afficher la sélection   ##<<<<<<<<<<<<<< à finir
            command=lambda opt=sous_option: fonction_bouton_modele(opt, model_label),

            #command = fonction_sous_option(option,sous_option),  # Fonction à définir pour chaque sous-option
            height=((768/4) / (len(options[option])+1)),
            width=235,
            text_color=color2,
            hover_color=color1,
            fg_color=color3,
            font=(font_Type_default, 16),
            corner_radius=30
        )
        bouton_sous_option.pack(padx=10, pady=1)
        listes_boutons_dynamiques.append(bouton_sous_option)
    bouton_retour = ctk.CTkButton(
        menu_frame,
        text="Retour",
        command=afficher_options,  # Revenir au menu principal
        height=((768/4) / (len(options[option])+1)),
        width=235,
        text_color=color2,
        hover_color=color1,
        fg_color=color3,
        font=(font_Type_default, 16),
        corner_radius=30
    )
    bouton_retour.pack(padx=10, pady=1)
    listes_boutons_dynamiques.append(bouton_retour)


def afficher_options():
    clear_menu()
# Ajouter les boutons principaux
    for option in options.keys():
        bouton_option = ctk.CTkButton(
            menu_frame,
            text=option,
            # Afficher les sous-options   #<<<<<<<<<< potentiellement bon
            command=lambda opt=option: afficher_sous_options(opt),
            height=((768/4) / (len(options)+1)),
            width=235,
            text_color=color2,
            hover_color=color1,
            fg_color=color3,
            font=(font_Type_default, 16),
            corner_radius=30
        )
        bouton_option.pack(padx=10, pady=1)
        listes_boutons_dynamiques.append(bouton_option)
    bouton_retour = ctk.CTkButton(
        menu_frame,
        text="Retour",
        command=afficher_Menu,  # Revenir au menu principal
        height=((768/4) / (len(options)+1)),
        width=235,
        text_color=color2,
        hover_color=color1,
        fg_color=color3,
        font=(font_Type_default, 16),
        corner_radius=30
    )
    bouton_retour.pack(padx=10, pady=1)
    listes_boutons_dynamiques.append(bouton_retour)


def afficher_Menu():
    clear_menu()
    bouton_menu = ctk.CTkButton(
        menu_frame,
        command=afficher_options,  # Afficher le menu principal
        height=(768 / 4),
        width=256,
        text="Menu",
        text_color=color2,
        hover_color=color3,
        bg_color=color1,
        fg_color=color3,
        font=(font_Type_default, 64),
        corner_radius=30
    )
    bouton_menu.pack(expand=False)
    listes_boutons_dynamiques.append(bouton_menu)


afficher_Menu()

# === Model name label === # checked
# dimensions = 256x(768/16)
model_label = ctk.CTkLabel(
    main_frame,
    text="Model : v ej " ,
    font=(font_Type_default, font_Size_info_model_temps, "bold"),
    width=256,
    height=(768/16),
    text_color=color2,
    corner_radius=20,
    fg_color=color3
)
model_label.grid(row=15, column=8, rowspan=1, columnspan=4, padx=10, pady=10)

# === Computing time label === # #checked
# dimensions = 256x(768/16)
time_label = ctk.CTkLabel(
    main_frame,
    text="Temps de calcul : 0.0s",
    font=(font_Type_default, font_Size_info_model_temps, "bold"),
    width=256,
    height=(768/16),
    text_color=color2,
    corner_radius=20,
    fg_color=color3
)
time_label.grid(row=15, column=12, rowspan=1, columnspan=4, padx=10, pady=10)


class DrawingApp(ctk.CTkFrame):
    """
    Main app class
    """

    def __init__(self, parent, modeles):  # checked-ish
        super().__init__(parent)
        self.modeles = modeles
        self.model = modeles[0]
        self.canvas_width = 572
        self.canvas_height = 606

        self.canvas = ctk.CTkCanvas(
            self,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=color1
        )
        self.canvas.pack(fill='both', expand=False, padx=10, pady=10)

        self.image = Image.new(
            "L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.last_x, self.last_y = None, None
        self.bind_events()

    def bind_events(self):  # checked-ish
        """
        Bind mouse events to the drawing canvas
        """
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
    def set_model(self, index, label=None):
        if 0 <= index < len(self.modeles):
            self.model = self.modeles[index]
            print(f"Modèle changé : {self.model.name}")
        if label:
            label.configure(text=f"Model = {self.model.name}")
        else:
            print("Index invalide pour le modèle.")

    def draw_on_canvas(self, event):  # updated + checked
        """
        Called when the mouse is pressed in the canvas. Updates the image.
        """
        radius = 30
        x0, y0 = event.x - radius, event.y - radius
        x1, y1 = event.x + radius, event.y + radius

        self.canvas.create_oval(x0, y0, x1, y1, fill="white", outline="white")
        self.draw.ellipse([x0, y0, x1, y1], fill=0)

        # If a previous position is recorded, draw a line between them.
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y, fill="white", width=radius * 2)
            self.draw.line([self.last_x, self.last_y, event.x,
                           event.y], fill=0, width=radius * 2)

        # Record the coordinates of the mouse
        self.last_x, self.last_y = event.x, event.y

    def on_release(self, event):  # checked
        """
        Called when the mouse is released.
        """
        # Reset mouse position
        self.last_x, self.last_y = None, None
        # Try to identify the character
        try:
            self.predict()
        except Exception as e:
            # Display error message in console and clear the canvas after 1 second.
            print("Erreur de prédiction :", e)
            self.canvas.after(1000, self.clear_canvas)
            result_label.configure(text="Nope.", font=(
                font_Type_default, 90), text_color=color2, fg_color=color3)
        # Clear the canvas after 1.5 seconds
        self.canvas.after(1500, self.clear_canvas)

    def clear_canvas(self):  # checked-ish
        self.canvas.delete("all")
        self.image = Image.new(
            "L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        """
        Send the canvas data to the model and update the interface to fit the predicion
        """
        # Start a timer
        start_time = time.time()

        # Format the image before sending it to the model.
        matrix = image_processing.format_matrix(np.array(self.image))

        prediction = self.model(matrix.reshape(1, 784))

        # Get the label of the character with the highest confidence
        character = emnist.label_to_char(np.argmax(prediction))

        # Display the predicted character
        result_label.configure(text=f"{character}", font=(
            "Comic sans ms", 202), text_color=color2, fg_color=color3)

        # Measure the computation time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Display the computation time in the interface
        time_label.configure(text=f"Temps de calcul : {elapsed_time:.2f}s")

        # Get the confidence in every label
        probabilities = np.array(prediction).flatten()
        labels = [emnist.label_to_char(i) for i in range(len(probabilities))]

        # Sort the labels by confidence
        sorted_indices = np.argsort(probabilities)[::-1]
        top_indices = sorted_indices[:6]
        top_probabilities = probabilities[top_indices]
        top_labels = [labels[i] for i in top_indices]

        # Display the labels in a pie chart
        dpi = 200
        figsize = (610 / dpi, 380 / dpi)
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(color3)

        ax = fig.add_subplot(111)
        ax.set_facecolor(color3)

        segment_colors = ['#D4342D', '#F3DDAC',
                          '#AA2822', '#E7BF71', '#8E0F06', '#C09645']
        textprops = {'fontsize': 6, 'color': 'black'}
        ax.pie(top_probabilities, labels=top_labels, autopct='%1.1f%%',
               startangle=90, colors=segment_colors, textprops=textprops)
        ax.axis('equal')

        # Display the pie chart in the statistics frame
        for widget in statistics_frame.winfo_children():
            widget.destroy()

        graph_frame = ctk.CTkFrame(
            statistics_frame,
            width=512,
            height=336,
            fg_color="transparent",
            corner_radius=0
        )
        # <<<<<<<<<<<<<<< Don't touch pady
        graph_frame.pack(fill="both", padx=10, pady=16, expand=False)

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=False)

        # Convert the formated image matrix in an image to display in the preview frame
        img_array = np.array(matrix)
        img_resized = Image.fromarray(img_array).resize((236, 172))
        img_tk = CTkImage(light_image=img_resized, size=(236, 172))

        if hasattr(self, 'blue_frame_label'):
            self.blue_frame_label.configure(
                image=img_tk, text="", fg_color=color3)
        else:
            self.blue_frame_label = ctk.CTkLabel(
                input_frame,
                image=img_tk,
                text="",
                fg_color=color3
            )
            self.blue_frame_label.pack(padx=10, pady=10, expand=False)
