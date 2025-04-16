import array
import cv2
import numpy as np
from polars import first
import image_processing as im


def deskew(image):
    """
    Corrige l'inclinaison de l'image binaire en détectant l'angle moyen des pixels non nuls.
    """
    # Récupération des coordonnées des pixels non nuls (texte)
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # Ajustement de l'angle selon la convention de cv2.minAreaRect
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Calcul du centre de l'image et rotation
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(image_path):
    """
    Effectue le prétraitement d'une image pour la préparation à l'OCR :
      - Conversion en niveaux de gris
      - Réduction du bruit par filtre médian
      - Binarisation avec seuil d'Otsu
      - Correction de l'inclinaison (deskew)
    """
    # Charger l'image en couleur
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin.")

    # 1. Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Réduction du bruit avec un filtre médian
    denoised = cv2.medianBlur(gray, 1)  # Taille du noyau = 3 (à ajuster selon l'image)
    
    # 3. Binarisation avec l'algorithme d'Otsu
    # cv2.threshold retourne le seuil utilisé et l'image binaire
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 4. Correction de l'inclinaison (deskew)
    deskewed =  binary
    # deskewed = deskew(binary)
    return deskewed



def segment_lines(binary_img, min_line_height=10):
    """
    Segmente l'image binaire en lignes de texte à l'aide de l'histogramme horizontal.
    
    :param binary_img: Image binaire (0 et 255) avec le texte en noir.
    :param min_line_height: Hauteur minimale pour considérer une région comme une ligne.
    :return: Liste de tuples (start_row, end_row) pour chaque ligne détectée.
    """
    # Inverser l'image pour que le texte soit blanc (255) sur fond noir (0)
    inverted = cv2.bitwise_not(binary_img)
    # Calcul de l'histogramme horizontal (nombre de pixels blancs par ligne)
    hist = np.sum(inverted, axis=1) // 255
    lines = []
    in_line = False
    start = 0
    for i, value in enumerate(hist):
        if value > 0 and not in_line:
            in_line = True
            start = i
        elif value == 0 and in_line:
            in_line = False
            if (i - start) >= min_line_height:
                lines.append((start, i))
    # Gestion de la dernière ligne (si l'image se termine sur du texte)
    if in_line and (len(hist) - start >= min_line_height):
        lines.append((start, len(hist)))
    return lines


def segment_words(line_img, min_word_width=5):
    """
    Segmente une ligne d'image en mots à l'aide de l'histogramme vertical.
    
    :param line_img: Image binaire (d'une seule ligne) avec le texte en noir.
    :param min_word_width: Largeur minimale pour considérer une région comme un mot.
    :return: Liste de tuples (start_col, end_col) pour chaque mot détecté.
    """
    # Inverser la ligne (texte en blanc sur fond noir)
    inverted = cv2.bitwise_not(line_img)
    # Calcul de l'histogramme vertical (nombre de pixels blancs par colonne)
    hist = np.sum(inverted, axis=0) // 255
    words = []
    in_word = False
    start = 0
    for j, value in enumerate(hist):
        if value > 0 and not in_word:
            in_word = True
            start = j
        elif value == 0 and in_word:
            in_word = False
            if (j - start) >= min_word_width:
                words.append((start, j))
    if in_word and (len(hist) - start >= min_word_width):
        words.append((start, len(hist)))
    return words

def avgLenght(charr):
    avg_lenght = 0
    for c in charr:
        avg_lenght += (c[1]-c[0])
    avg_lenght /= len(charr)
    return avg_lenght


def ImgToChar(input_image_path, biais=0):
    # Chemin vers l'image
    # Biais pour ajuster la longeur d'un charactère
    # Retourne un Array 2d de mots 

    binary_img = preprocess_image(input_image_path)
    if binary_img is None:
        print("Erreur lors du chargement de l'image.")
        return

    # Segmenter l'image en lignes
    lines = segment_lines(binary_img)
    Mots = []

    # Pour chaque ligne, segmenter en mots et sauvegarder les images correspondantes
    for idx, (row_start, row_end) in enumerate(lines):

        line_img = binary_img[row_start:row_end, :]
        charr = segment_words(line_img)
        #
        avg_lenght = avgLenght(charr)
        #
        CharacterImg = []
        last_place = 0
        #print(f"Ligne {idx+1} - charactère détectés :", charr)
        for w_idx, (col_start, col_end) in enumerate(charr):
            word_img = line_img[:, col_start:col_end]
            CharacterImg.append(word_img)


            #ajoute les mots
            last_place = col_end
            if(last_place-col_start)> avg_lenght + biais:
                CharacterImg = []
                Mots.append(CharacterImg)
                
            if w_idx + 1 == len(charr):
                CharacterImg = []
                Mots.append(CharacterImg)
                
            

            # Sauvegarde de l'image de chaque mot
            #word_filename = f"line{idx+1}_char{w_idx+1}.jpg"
            #cv2.imwrite(word_filename, word_img)
            #print("Mot sauvegardé :", word_filename)

    return Mots
    