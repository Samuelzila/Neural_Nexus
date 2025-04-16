"""
Basic usage:

import image_processing as ip

matrix = ip.matrix_from_path("./image.png")
matrix = ip.format_matrix(matrix)
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import time  # Importer le module time pour mesurer les temps d'exécution

def matrix_from_path(path):
    """
    With a given path, converts an image into a numpy matrix adn return it.
    """
    with Image.open(path) as img:
        img = img.convert("L")
        return np.array(img)




def format_matrix(matrix, flatten=False):
    """
    From an input image matrix, create a new one with the EMNIST format and return it.
    flatten specifies if the final array should be flattened.
    """
    start_time = time.time()
    
    # Étape 0 : Bicubic resize 1
    t0 = time.time()
    matrix = bicubic_resize(matrix, 128)
    print(f"Temps pour bicubic_resize : {time.time() - t0:.4f} secondes")

    # Étape 1 : White balance
    t1 = time.time()
    matrix = white_balance(matrix, flatten=True)
    print(f"Temps pour white_balance : {time.time() - t1:.4f} secondes")

    # Étape 2 : Black background
    t2 = time.time()
    matrix = black_background(matrix)
    print(f"Temps pour black_background : {time.time() - t2:.4f} secondes")

    # Étape 3 : Center image
    t3 = time.time()
    matrix = center_image(matrix)
    print(f"Temps pour center_image : {time.time() - t3:.4f} secondes")

    # Étape 4 : Crop
    t4 = time.time()
    matrix = crop(matrix, padding=2, keep_centered=True)
    print(f"Temps pour crop : {time.time() - t4:.4f} secondes")

    # Étape 5 : Make square
    t5 = time.time()
    matrix = make_square(matrix)
    print(f"Temps pour make_square : {time.time() - t5:.4f} secondes")

    # Étape 6 : Bicubic resize
    t6 = time.time()
    matrix = bicubic_resize(matrix, 28)
    print(f"Temps pour bicubic_resize : {time.time() - t6:.4f} secondes")

    # Temps total pour format_matrix
    print(f"Temps total pour format_matrix : {time.time() - start_time:.4f} secondes")

    return matrix.flatten() if flatten else matrix


def crop(matrix, padding=0, keep_centered=True):
    """
    Removes empty rows and columns from matrix.
    Padding tells how many empty rows and columns to leave on each side.
    If keep cenetered is true, it won't crop in a way that would uncenter the image.
    """
    start_time = time.time()

    # Dimensions de la matrice
    m, n = matrix.shape

    # Étape 1 : Trouver la ligne supérieure
    t1 = time.time()
    for i in reversed(range(m)):
        if np.all(matrix[:i+1] == 0):
            argtop = i-1
            break
    print(f"Temps pour trouver la ligne supérieure : {time.time() - t1:.4f} secondes")

    # Étape 2 : Trouver la ligne inférieure
    t2 = time.time()
    for i in range(m):
        if np.all(matrix[i:] == 0):
            argbottom = i+1
            break
    print(f"Temps pour trouver la ligne inférieure : {time.time() - t2:.4f} secondes")

    # Étape 3 : Trouver la colonne droite
    t3 = time.time()
    for i in reversed(range(n)):
        if np.all(matrix[:, :i+1] == 0):
            argright = i-1
            break
    print(f"Temps pour trouver la colonne droite : {time.time() - t3:.4f} secondes")

    # Étape 4 : Trouver la colonne gauche
    t4 = time.time()
    for i in range(n):
        if np.all(matrix[:, i:] == 0):
            argleft = i+1
            break
    print(f"Temps pour trouver la colonne gauche : {time.time() - t4:.4f} secondes")

    # Étape 5 : Recadrer la matrice
    t5 = time.time()
    if not keep_centered:
        matrix = matrix[argtop:argbottom+1, argright:argleft+1]
    else:
        argtb = min(argtop, m-argbottom)
        argrl = min(argright, n-argleft)
        matrix = matrix[argtb:m-argtb+1, argrl:n-argrl+1]
    print(f"Temps pour recadrer la matrice : {time.time() - t5:.4f} secondes")

    # Étape 6 : Ajouter du padding
    t6 = time.time()
    matrix = np.append(matrix, np.zeros((padding, matrix.shape[1])), axis=0)
    matrix = np.append(np.zeros((padding, matrix.shape[1])), matrix, axis=0)
    matrix = np.append(matrix, np.zeros((matrix.shape[0], padding)), axis=1)
    matrix = np.append(np.zeros((matrix.shape[0], padding)), matrix, axis=1)
    print(f"Temps pour ajouter du padding : {time.time() - t6:.4f} secondes")

    # Temps total pour crop
    print(f"Temps total pour crop : {time.time() - start_time:.4f} secondes")

    return matrix


def make_square(matrix):
    """
    From an input matrix, make it square by adding empty rows or columns
    """
    m, n = matrix.shape
    η = max(m, n)  # Final dimensions.
    # Add rows if necessary
    if m < η:
        rows = np.zeros((math.floor((η-m)/2), n))
        matrix = np.append(matrix, rows, axis=0)
        rows = np.zeros((math.ceil((η-m)/2), n))
        matrix = np.append(rows, matrix, axis=0)
    # Add columns if necessary
    if n < η:
        cols = np.zeros((m, math.floor((η-n)/2)))
        matrix = np.append(matrix, cols, axis=1)
        cols = np.zeros((m, math.ceil((η-n)/2)))
        matrix = np.append(cols, matrix, axis=1)

    return matrix


def black_background(matrix):
    """
    Makes the background of the image black, if not already.
    """
    m, n = matrix.shape
    ratio = np.sum(matrix)/(m*n*255)  # Percentage of white in the image.

    # We assume that the background should represent a greater percentage than the foreground.
    # If there is more white than black in the image, invert the colours.
    if ratio > 0.5:
        matrix = 255-matrix

    return matrix


def white_balance(matrix, flatten=False):
    """
    Given an image matrix, make its lightest colour white and darkest one black. Adjust all other values in consequence.
    If flatten is true, the colors will only be black and white.
    """
    lightest = np.max(matrix)
    darkest = np.min(matrix)
    if not flatten:
        return ((matrix-darkest)/(lightest-darkest)) * 255

    middle = (darkest+lightest)/2
    # Change light values to white
    matrix[(matrix >= middle)] = 255
    matrix[(matrix < middle)] = 0

    return matrix


def center_image(matrix):
    """
    Centers the image of a matrix
    """
    m, n = matrix.shape

    # Compute the center of mass (C_x, C_y)
    C_x = 0
    C_y = 0
    for y in range(m):
        for x in range(n):
            C_x += (x-n/2)*matrix[y, x]
            C_y += (y-m/2)*matrix[y, x]

    total_mass = np.sum(matrix)

    C_x /= total_mass
    C_y /= total_mass

    # Translate the elements of the matrix along the negative center of mass.
    matrix = np.roll(matrix, -round(C_y), axis=0)
    matrix = np.roll(matrix, -round(C_x), axis=1)

    return matrix


def bicubic_resize(matrix, size_x):
    """
    Resizes a matrix using the bicubic algorithm.
    size_x specify the new horizontal dimension of the matrix.
    The vertical dimension is deduced to keep the aspect ratio.
    """
    m, n = matrix.shape  # Original dimensions of the matrix
    ratio = size_x/n  # scaling factor
    size_y = math.floor(m*ratio)

    image = Image.fromarray(matrix)

    image = image.resize((size_x, size_y), Image.Resampling.BICUBIC)

    return np.array(image)


if __name__ == "__main__":
    image = matrix_from_path("./Screenshot_20250414_132400.png")
    matrix = format_matrix(image, flatten=False)

    plt.imshow(matrix, cmap="grey")
    plt.show()

