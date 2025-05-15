"""#Basic usage:

import image_processing as ip

matrix = ip.matrix_from_path("./image.png")
matrix = ip.format_matrix(matrix)"""


from PIL import Image,ImageOps
import numpy as np
import math


def matrix_from_path(path):
    """
    With a given path, converts an image into a numpy matrix adn return it.
    """
    with Image.open(path) as img:
        img = img.convert("L")  # <-- convertit en niveaux de gris (0-255)
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img = np.array(img)/255
        return img


def format_matrix(matrix, flatten=False):
    """
    From an input image matrix, create a new one with the EMNIST format and return it.
    flatten specifies if the final array should be flattened.
    """

    # Resize image to a fixed resolution to have a predictable execution time
    matrix = bicubic_resize(matrix, 128)

    # White balance
    matrix = white_balance(matrix, flatten_colours=True)

    # Ensure that the background is black.
    matrix = black_background(matrix)

    # Center letter in image
    matrix = center_image(matrix)

    # Crop the image
    matrix = crop(matrix, padding=2, keep_centered=True)

    # Make the image square
    matrix = make_square(matrix)

    # Resize the image to 28x28
    matrix = bicubic_resize(matrix, 28)

    return matrix.flatten() if flatten else matrix


def crop(matrix, padding=0, keep_centered=True):
    """
    Removes empty rows and columns from the outside of the image matrix.
    Padding tells how many empty rows and columns to leave on each side.
    If keep cenetered is true, it won't crop in a way that would uncenter the image.
    """
    
    if np.all(matrix == 0):
        return matrix

    # Dimensions of the matrix
    m, n = matrix.shape

    # Default values, if there is nothing to crop
    argtop = 0
    argbottom = m-1
    argleft = 0
    argright = n-1

    # Find where to crop on the top
    for i in reversed(range(m)):
        if np.all(matrix[:i+1] == 0):
            argtop = i-1
            break

    # Find where to crop on the bottom
    for i in range(m):
        if np.all(matrix[i:] == 0):
            argbottom = i+1
            break

    # Find where to crop on the left
    for i in reversed(range(n)):
        if np.all(matrix[:, :i+1] == 0):
            argleft = i-1
            break

    # Find where to crop on the right
    for i in range(n):
        if np.all(matrix[:, i:] == 0):
            argright = i+1
            break

    # Crop
    if not keep_centered:
        matrix = matrix[argtop:argbottom+1, argleft:argright+1]
    else:
        argtb = min(argtop, m-argbottom)
        argrl = min(argleft, n-argright)
        matrix = matrix[argtb:m-argtb+1, argrl:n-argrl+1]

    # Add padding
    matrix = np.append(matrix, np.zeros((padding, matrix.shape[1])), axis=0)
    matrix = np.append(np.zeros((padding, matrix.shape[1])), matrix, axis=0)
    matrix = np.append(matrix, np.zeros((matrix.shape[0], padding)), axis=1)
    matrix = np.append(np.zeros((matrix.shape[0], padding)), matrix, axis=1)

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


def white_balance(matrix, flatten_colours=False):
    """
    Given an image matrix, make its lightest colour white and darkest one black. Adjust all other values in consequence.
    If flatten is true, the colors will only be black and white.
    """
    lightest = np.max(matrix)
    darkest = np.min(matrix)
    if not flatten_colours:
        return ((matrix-darkest)/(lightest-darkest)) * 255

    middle = (darkest+lightest)/2
    # Change light values to white
    matrix[(matrix >= middle)] = 255
    matrix[(matrix < middle)] = 0

    return matrix


import numpy as np
from scipy.ndimage import shift

def center_image(matrix):
    """
    Centers the image of a matrix using center of mass and sub-pixel shifting.
    """
    m, n = matrix.shape

    # Compute total mass
    total_mass = np.sum(matrix)
    if total_mass == 0:
        return matrix

    # Compute X and Y indices
    Y, X = np.indices((m, n))

    # Compute center of mass
    C_x = np.sum((X - n / 2) * matrix) / total_mass
    C_y = np.sum((Y - m / 2) * matrix) / total_mass

    # Apply sub-pixel shift to center the image
    centered_matrix = shift(matrix, shift=(-C_y, -C_x), mode='nearest')

    return centered_matrix



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

    #image = image.resize((size_x, size_y), Image.Resampling.BICUBIC)
    image = image.resize((size_x, size_x), Image.Resampling.BICUBIC)


    return np.array(image)
