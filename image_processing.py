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

    matrix = white_balance(matrix)

    # Ensure that the background is black
    matrix = black_background(matrix)

    # Trim empty rows and columns
    matrix = crop(matrix, padding=2)

    # Make the matrix square, relative to its largest dimension.
    matrix = make_square(matrix)

    matrix = center_image(matrix)

    matrix = bicubic_resize(matrix, 28)

    return matrix.flatten() if flatten else matrix


def crop(matrix, padding=0):
    """
    Removes empty rows and columns from matrix.
    Padding tells how many empty rows and columns to leave on each side.
    """
    # Trim empty rows and columns
    matrix = matrix[~np.all(matrix == 0, axis=1)]
    matrix = matrix[:, ~np.all(matrix == 0, axis=0)]

    # Add <padding> rows and columns on every side.
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
        rows = np.zeros((η-m, n))
        matrix = np.append(matrix, rows, axis=0)
    # Add columns if necessary
    if n < η:
        cols = np.zeros((m, η-n))
        matrix = np.append(matrix, cols, axis=1)

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


def white_balance(matrix):
    """
    Given an image matrix, make its lightest colour white and darkest one black. Adjust all other values in consequence.
    """
    lightest = np.max(matrix)
    darkest = np.min(matrix)

    return ((matrix-darkest)/(lightest-darkest)) * 255


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
