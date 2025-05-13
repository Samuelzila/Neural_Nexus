import customtkinter as ctk
from customtkinter import CTkImage, CTkButton
from PIL import Image, ImageDraw
import numpy as np
import emnist
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import image_processing

# ============================================================================
# == STYLE CONFIGURATION =====================================================
# ============================================================================

# Fonts
font_default = 'Roboto'
font_weight_default = 'bold'
font_size_info_model = 16
font_size_prediction = 64
font_size_retry = 112
font_size_reponse = 202
font_size_pie = 6
font_style = "normal"

# Colors

color1 = "red"
color2 = "yellow"
color3 = "darkred"
color4 = "orange"
color5 = "red"
color6 = "green"
color7 = "orange"
color8 = "purple"
color9 = "black"

# Dimensions
PAGE_WIDTH = 1024
PAGE_HEIGHT = 768

# Grids
GRID_ROWS = 16
GRID_COLUMNS = 16

# ============================================================================
# == APPLICATION INITIALIZATION =============================================
# ============================================================================

ctk.set_appearance_mode("black")

# Main window
fenetre = ctk.CTk("black")
fenetre.title("Reconnaissance de chiffres manuscrits")

# Main Frame
main_frame = ctk.CTkFrame(
    fenetre,
    width=PAGE_WIDTH,
    height=PAGE_HEIGHT,
    fg_color=color1,
)
main_frame.pack(expand=True)

# Grid configuration
for i in range(GRID_ROWS):
    main_frame.grid_rowconfigure(i, weight=1)
for i in range(GRID_COLUMNS):
    main_frame.grid_columnconfigure(i, weight=1)

# ============================================================================
# == FRAMES ET COMPOSANTS ====================================================
# ============================================================================

# Résultats prédits
result_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=384,
    fg_color=color1
)
result_frame.grid(row=0, column=8, rowspan=8, columnspan=8, padx=10, pady=10, sticky ="nsew")

result_label = ctk.CTkLabel(
    result_frame,
    text="Prediction\nen attente",
    font=(font_default, font_size_prediction, font_weight_default),
    width=512,
    height=384,
    text_color=color2,
    corner_radius=30,
    fg_color=color3
)
result_label.pack(expand=True)

# Zone de dessin
canvas_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=(3 * (PAGE_HEIGHT / 4)),
    fg_color=color1,
    corner_radius=30
)
canvas_frame.grid(row=0, column=0, rowspan=12, columnspan=8, padx=10, pady=10)

# Statistiques
statistics_frame = ctk.CTkFrame(
    main_frame,
    width=512,
    height=512,
    corner_radius=30,
    fg_color=color3
)
statistics_frame.grid(row=8, column=8, rowspan=7, columnspan=8, padx=10, pady=10, sticky ="nsew")

# Aperçu AI input
input_frame = ctk.CTkFrame(
    main_frame,
    width=256,
    height=(PAGE_HEIGHT / 4),
    corner_radius=30,
    fg_color=color3
)
input_frame.grid(row=12, column=4, rowspan=4, columnspan=4, padx=10, pady=10)

# Menu
menu_frame = ctk.CTkFrame(
    main_frame,
    width=256,
    height=(PAGE_HEIGHT / 4),
    corner_radius=30,
    fg_color=color3
)
menu_frame.grid(row=12, column=0, rowspan=4, columnspan=4, padx=10, pady=10)

# Sous-menus
options = {
    "Changer de modele": ["NN.0,8850528270419435", "NN.0,884915279", "NN.0,884889", "NN.0,88", "NN.0,87", "always_right"],
    "Changer couleur": ["Barbie", "Captain America", "Sea", "Clouds", "Hot wheels"],
    "Changer le font": ["Arial", "Roboto", "Comic Sans Ms", "Times New Roman", "Courier New", "Verdana"],
    "Changer la taille": ["Normal", "Moyen", "Grand", "Très grand", "Géant"],
    "Changer la police": ["Gras", "Italique", "Souligné", "Barré", "Normal"]
}

# Dictionnaires de mapping
Models = {
    "NN.0,8850528270419435": 0,
    "NN.0,884915279": 1,
    "NN.0,884889": 2,
    "NN.0,88": 3,
    "NN.0,87": 4,
    "always_right": 5
}

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv     please dont touch colors 
Themes = {
    "Clouds": ["#ADC5E0", "#8EACD3", "#DDE5F7", "#FCFEFF", "#E8F1FF", "#DEDEDE", "#C4D9F0"],
    "Sea": ["#B8DBB0", "#2B3686", "#5DC1C0", "#1553AA", "#1198BE", "#111636", "#0A223F"],
    "Captain America": ["#0E305D", "#14406F", "#F6F6F7", "#BD142B", "#7E1918", "#4D0F0F", "#091E3A"],
    "Barbie": ["#E0218A", "#ED5C9B", "#F18DBC", "#F7B9D7", "#FACDE5", "#5C0E39", "#FEE6F2"],
    "Hot wheels": [ "#4251AE", "#F1D74D", "#D42F41", "#D82804", "#FF6600", "#2C84C7"]
}

FontTypes = {
    "Arial": "Arial",
    "Roboto": "Roboto",
    "Comic Sans Ms": "Comic Sans MS",
    "Times New Roman": "Times New Roman",
    "Courier New": "Courier New",
    "Verdana": "Verdana"
}

FontSizes = {
    "Normal": 16,
    "Moyen": 24,
    "Grand": 36,
    "Très grand": 48,
    "Géant": 60
}

FontStyles = {
    "Gras": "bold",
    "Italique": "italic",
    "Souligné": "underline",
    "Barré": "strikethrough",
    "Normal": "normal"
}

# Variable globale pour stocker l'instance de l'application
app_instance = None

# Fonction pour assigner l'instance
def set_app_instance(app):
    global app_instance
    app_instance = app

def fonction_sous_option(option, sous_option):
    print(f"fonction_sous_option : {option} - {sous_option}")
    if option in fonction_map:
        fonction_map[option](sous_option)
    else:
        print("Option non reconnue")

def fonction_bouton_modele(sous_option):
    print(f"fonction_bouton_modele : {sous_option}")
    if sous_option in Models:
        app_instance.set_model(Models[sous_option])
    else:
        print("Modèle non reconnu")
    update_theme()
    afficher_Menu()

def fonction_bouton_couleur(sous_option):
    print(f"fonction_bouton_couleur : {sous_option}")
    global color1, color2, color3, color4, color5, color6, color7
    if sous_option in Themes:
        couleurs = Themes[sous_option]
        couleurs_complètes = [couleurs[i % len(couleurs)] for i in range(7)]
        color1, color2, color3, color4, color5, color6, color7 = couleurs_complètes
        print(f"Couleurs appliquées : {color1}, {color2}, {color3}, {color4}, {color5}, {color6}, {color7}")
    else:
        print("Thème non reconnu")
    update_theme()
    afficher_Menu()

def fonction_bouton_font(sous_option):
    print(f"fonction_bouton_font : {sous_option}")
    global font_default
    if sous_option in FontTypes:
        font_default = FontTypes[sous_option]
    else:
        print("Font non reconnue")
    update_theme()
    afficher_Menu()

def fonction_bouton_taille(sous_option):
    print(f"fonction_bouton_taille : {sous_option}")
    global font_Size, font_size_info_model
    if sous_option in FontSizes:
        font_Size = FontSizes[sous_option]
        font_size_info_model = font_Size
        print(font_size_info_model)
    else:
        print("Taille non reconnue")
    update_theme()
    afficher_Menu()

def fonction_bouton_police(sous_option):
    print(f"fonction_bouton_taille : {sous_option}")
    global font_style
    if sous_option in FontStyles:
        font_style = FontStyles[sous_option]
    else:
        print("Taille non reconnue")
    update_theme()
    afficher_Menu()

# Mapping option vers fonction
fonction_map = {
    "Changer de modele": fonction_bouton_modele,
    "Changer couleur": fonction_bouton_couleur,
    "Changer le font": fonction_bouton_font,
    "Changer la taille": fonction_bouton_taille,
    "Changer la police": fonction_bouton_police
}
listes_boutons_dynamiques = []

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
            command=lambda opt=option, sous_opt=sous_option: fonction_sous_option(opt, sous_opt),
            height=((768/4) / (len(options[option])+1)),
            width=235,
            text_color=color2,
            hover_color=color5,
            fg_color=color3,
            font=(font_default, 16, "bold"),
            corner_radius=30
        )
        print("afficher_sous_option : "+ option + " - " +sous_option)
        bouton_sous_option.pack(padx=10, pady=1)
        listes_boutons_dynamiques.append(bouton_sous_option)
    bouton_retour = ctk.CTkButton(
        menu_frame,
        text="Retour",
        command=afficher_options,  # Revenir au menu principal
        height=((768/4) / (len(options[option])+1)),
        width=235,
        text_color=color2,
        hover_color=color5,
        fg_color=color3,
        font=(font_default, 16),
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
            command=lambda opt=option: afficher_sous_options(opt),
            height=((768/4) / (len(options)+1)),
            width=235,
            text_color=color2,
            hover_color=color5,
            fg_color=color3,
            font=(font_default, 16),
            corner_radius=30
        )
        print("afficher_option : "+ option)
        bouton_option.pack(padx=10, pady=1)
        listes_boutons_dynamiques.append(bouton_option)
    bouton_retour = ctk.CTkButton(
        menu_frame,
        text="Retour",
        command=afficher_Menu,  # Revenir au menu principal
        height=((768/4) / (len(options)+1)),
        width=235,
        text_color=color2,
        hover_color=color5,
        fg_color=color3,
        font=(font_default, 16),
        corner_radius=30
    )
    bouton_retour.pack(padx=10, pady=1)
    listes_boutons_dynamiques.append(bouton_retour)

listes_boutons_dynamiques = []

def clear_menu():
    for bouton in listes_boutons_dynamiques:
        bouton.destroy()
    listes_boutons_dynamiques.clear()

def afficher_sous_options(option):
    clear_menu()
    for sous_option in options[option]:
        bouton = ctk.CTkButton(
            menu_frame,
            text=sous_option,
            command=lambda opt=option, sous_opt=sous_option: fonction_sous_option(opt, sous_opt),
            height=(768/4) / (len(options[option])+1),
            width=235,
            text_color=color2,
            hover_color=color5,
            fg_color=color3,
            font=(font_default, 16, "bold"),
            corner_radius=30
        )
        print(f"afficher_sous_option : {option} - {sous_option}")
        bouton.pack(padx=10, pady=1)
        listes_boutons_dynamiques.append(bouton)

    bouton_retour = ctk.CTkButton(
        menu_frame,
        text="Retour",
        command=afficher_options,
        height=(768/4) / (len(options[option])+1),
        width=235,
        text_color=color2,
        hover_color=color5,
        fg_color=color3,
        font=(font_default, 16),
        corner_radius=30
    )
    bouton_retour.pack(padx=10, pady=1)
    listes_boutons_dynamiques.append(bouton_retour)

def afficher_options():
    clear_menu()
    for option in options.keys():
        bouton = ctk.CTkButton(
            menu_frame,
            text=option,
            command=lambda opt=option: afficher_sous_options(opt),
            height=(768/4) / (len(options)+1),
            width=235,
            text_color=color2,
            hover_color=color5,
            fg_color=color3,
            font=(font_default, 16),
            corner_radius=30
        )
        print(f"afficher_option : {option}")
        bouton.pack(padx=10, pady=1)
        listes_boutons_dynamiques.append(bouton)

    bouton_retour = ctk.CTkButton(
        menu_frame,
        text="Retour",
        command=afficher_Menu,
        height=(768/4) / (len(options)+1),
        width=235,
        text_color=color2,
        hover_color=color5,
        fg_color=color3,
        font=(font_default, 16),
        corner_radius=30
    )
    bouton_retour.pack(padx=10, pady=1)
    listes_boutons_dynamiques.append(bouton_retour)

def afficher_Menu():
    clear_menu()
    bouton_menu = ctk.CTkButton(
        menu_frame,
        command=afficher_options,
        height=(768/4),
        width=256,
        text="Menu",
        text_color=color2,
        hover_color=color5,
        bg_color=color1,
        fg_color=color3,
        font=(font_default, 64, "normal"),
        corner_radius=30
    )
    bouton_menu.pack(expand=False)
    listes_boutons_dynamiques.append(bouton_menu)

# Lancer le menu principal
afficher_Menu()

# === Label pour le nom du modèle ===
# Dimensions : 256x(768/16)
model_label = ctk.CTkLabel(
    main_frame,
    text="Model : NN.0,8850528270419435", 
    font=(font_default, font_size_info_model, "bold"),
    width=256,
    height=(768/16),
    text_color=color2,
    corner_radius=20,
    fg_color=color3
)
# Placement avec la méthode grid
model_label.grid(row=15, column=8, rowspan=1, columnspan=4, padx=10, pady=10)

# === Label pour le temps de calcul ===
# Dimensions : 256x(768/16)
time_label = ctk.CTkLabel(
    main_frame,
    text="Temps de calcul : 0.0s",
    font=(font_default, font_size_info_model, "bold"),
    width=256,
    height=(768/16),
    text_color=color2,
    corner_radius=20,
    fg_color=color3
)
# Placement avec la méthode grid
time_label.grid(row=15, column=12, rowspan=1, columnspan=4, padx=10, pady=10)

def update_theme():
    print("Mise à jour du thème...")
    
    # --- Mise à jour des couleurs globales ---
    global color1, color2, color3, color4, color5, color6, color7
    global font_default, font_size_info_model, font_style
    
    # --- Mise à jour des frames ---
    main_frame.configure(fg_color=color1)
    result_frame.configure(fg_color=color1)
    canvas_frame.configure(fg_color=color1)
    statistics_frame.configure(fg_color=color3)
    input_frame.configure(fg_color=color3)
    menu_frame.configure(fg_color=color3)
    
    # --- Mise à jour des labels ---
    result_label.configure( text_color=color2, fg_color=color3, font=(font_default, font_size_prediction, font_style))
    
    model_label.configure( text_color=color2, fg_color=color3, font=(font_default, font_size_info_model, font_style))
    
    time_label.configure( text_color=color2, fg_color=color3, font=(font_default, font_size_info_model, font_style))    
    
    # --- Mise à jour des boutons dynamiques ---
    listes_Temporaires = listes_boutons_dynamiques.copy()
    listes_boutons_dynamiques.clear()
    for bouton in listes_Temporaires:
        bouton.configure(
            text_color=color2,
            hover_color=color5,
            fg_color=color3,
            bg_color=color1,
            font=(font_default, 16, font_style)
        )
        listes_boutons_dynamiques.append(bouton)
    
    # --- Mise à jour du canvas ---
    # canvas_frame.configure(fg_color=color2)  # Optionnel si nécessaire
    app_instance.canvas.configure(bg=color3)
    app_instance.set_pen_color(color5)
    
    # --- Mise à jour du contour de l'application ---
    fenetre.configure(border_color=color1, fg_color=color6)  # Met à jour l'interface via la méthode update()
    
    print("Thème mis à jour avec succès.")

class DrawingApp(ctk.CTkFrame):
    """Application de dessin et prédiction de caractères manuscrits."""

    def __init__(self, parent, modeles):
        super().__init__(parent)

        # ─── Paramètres et variables ────────────────────────
        self.modeles = modeles
        self.model = modeles[0]
        self.model_name = self.model.name
        self.canvas_width, self.canvas_height = 572, 606
        self.pen_color = color6
        self.last_x, self.last_y = None, None

        # ─── Initialisation de l'image PIL associée ──────────
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # ─── Canvas de dessin ───────────────────────────────
        self.canvas = ctk.CTkCanvas(
            self, width=self.canvas_width, height=self.canvas_height, bg=color5
        )
        self.canvas.pack(fill='both', expand=True, padx=10, pady=10)

        # ─── Liaison des événements souris ──────────────────
        self.bind_events()

        # ─── Nettoyage initial du canvas ────────────────────
        self.clear_canvas()

    # ──────────────────────────────────────────────────────────────
    # BINDINGS
    # ──────────────────────────────────────────────────────────────

    def bind_events(self):
        """Associe les événements souris au canvas."""
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    # ──────────────────────────────────────────────────────────────
    # PARAMÈTRES
    # ──────────────────────────────────────────────────────────────

    def set_model(self, index):
        """Change le modèle utilisé pour la prédiction."""
        self.model = self.modeles[index]
        self.model_name = self.model.name
        model_label.configure(text=f"Model : {self.model_name}")
        afficher_Menu()
        print(f"set_model : {index}")

    def set_pen_color(self, color):
        """Change la couleur du crayon."""
        self.pen_color = color
        print(f"Couleur du crayon changée en : {self.pen_color}")

    # ──────────────────────────────────────────────────────────────
    # DESSIN
    # ──────────────────────────────────────────────────────────────

    def draw_on_canvas(self, event):
        """Trace un trait entre l'ancienne et la nouvelle position de la souris."""
        radius = 30
        x0, y0 = event.x - radius, event.y - radius
        x1, y1 = event.x + radius, event.y + radius

        # Sur le canvas Tkinter
        self.canvas.create_oval(x0, y0, x1, y1, fill=self.pen_color, outline=self.pen_color)

        # Sur l'image PIL
        self.draw.ellipse([x0, y0, x1, y1], fill=0)

        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    fill=self.pen_color, width=radius * 2)
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                           fill=0, width=radius * 2)

        # Mise à jour des positions
        self.last_x, self.last_y = event.x, event.y

    def on_release(self, event):
        """Appelé quand la souris est relâchée."""
        self.last_x, self.last_y = None, None
        try:
            self.predict()
            self.canvas.after(1000, self.clear_canvas)
        except Exception as e:
            print("Erreur de prédiction :", e)
            self.canvas.after(1000, self.clear_canvas)
            result_label.configure(text="Nope.",
                                   font=(font_default, font_size_retry, font_weight_default),
                                   text_color=color2, fg_color=color3)
            self.canvas.after(1500, self.clear_canvas)

    def clear_canvas(self):
        """Efface le canvas et réinitialise l'image."""
        print("Nettoyage du canvas")
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

    # ──────────────────────────────────────────────────────────────
    # PREDICTION ET AFFICHAGE
    # ──────────────────────────────────────────────────────────────

    def predict(self):
        """Envoie l'image au modèle et met à jour l'interface avec le résultat."""

        # Chronométrage
        start_time = time.time()

        # Prétraitement de l'image
        matrix = image_processing.format_matrix(np.array(self.image))
        prediction = self.model(matrix.reshape(1, 784))
        character = emnist.label_to_char(np.argmax(prediction))

        # Affichage résultat
        result_label.configure(
            text=f"{character}",
            font=(font_default, font_size_reponse, font_style),
            text_color=color2,
            fg_color=color3
        )

        # Temps de calcul
        elapsed_time = time.time() - start_time
        time_label.configure(text=f"Temps de calcul : {elapsed_time:.2f}s")

        # Affichage statistiques
        self.display_statistics(prediction)

        # Aperçu image
        self.update_preview(matrix)

    def display_statistics(self, prediction):
        """Affiche un graphique circulaire des 6 meilleures prédictions."""
        probabilities = np.array(prediction).flatten()
        labels = [emnist.label_to_char(i) for i in range(len(probabilities))]
        sorted_indices = np.argsort(probabilities)[::-1]
        top_indices = sorted_indices[:6]

        top_probabilities = probabilities[top_indices]
        top_labels = [labels[i] for i in top_indices]

        dpi, figsize = 200, (610 / 200, 380 / 200)
        fig = Figure(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(color3)
        ax = fig.add_subplot(111)
        ax.set_facecolor(color3)

        colors = [color1, color2, color4, color5, color6, color9]
        textprops = {'fontsize': font_size_pie, 'color': color9}
        ax.pie(top_probabilities, labels=top_labels, autopct='%1.1f%%',
               startangle=90, colors=colors, textprops=textprops)
        ax.axis('equal')

        # Affichage graphique
        for widget in statistics_frame.winfo_children():
            widget.destroy()

        graph_frame = ctk.CTkFrame(
            statistics_frame, width=512, height=336, fg_color="transparent", corner_radius=0
        )
        graph_frame.pack(fill="both", padx=10, pady=16, expand=True)

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True)

    def update_preview(self, matrix):
        """Affiche un aperçu de l'image prétraitée."""
        img_array = np.array(matrix)
        img_resized = Image.fromarray(img_array).resize((236, 172))
        img_tk = CTkImage(light_image=img_resized, size=(236, 172))

        if hasattr(self, 'input_frame_label'):
            self.input_frame_label.configure(image=img_tk, text="", fg_color="white")
        else:
            self.input_frame_label = ctk.CTkLabel(
                input_frame, image=img_tk, text="", fg_color=color3
            )
            self.input_frame_label.pack(padx=10, pady=10, expand=True)
