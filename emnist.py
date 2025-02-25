import polars as pl
import numpy as np
import itertools


def training():
    """
    Returns a tuple (X,y)
    X is a NumPy array with rows being a flattened image (row major), and columns a pixel
    y is the associated label. The label is an id that is associated with a specific character.
    """
    return load_from_csv("training")


def test():
    """
    Returns a tuple (X,y)
    X is a NumPy array with rows being a flattened image (row major), and columns a pixel
    y is the associated label. The label is an id that is associated with a specific character.
    """
    return load_from_csv("test")


def training_batched(batch_size=50000):
    """
    The batch size is an int representing the size of each batch.

    Returns an iterator tuple (X,y)
    X is a NumPy array with rows being a flattened image (row major), and columns a pixel
    y is the associated label. The label is an id that is associated with a specific character.
    """
    return load_from_csv_batched("training", batch_size)


def test_batched(batch_size=50000):
    """
    The batch size is an int representing the size of each batch.

    Returns an iterator tuple (X,y)
    X is a NumPy array with rows being a flattened image (row major), and columns a pixel
    y is the associated label. The label is an id that is associated with a specific character.
    """
    return load_from_csv_batched("test", batch_size)


def load_from_csv(datasetName):
    """
    The dataset name is either "training" or "test".

    Returns a tuple (X,y)
    X is a NumPy array with rows being a flattened image (row major), and columns a pixel
    y is the associated label. The label is an id that is associated with a specific character.
    """
    data = pl.scan_csv(f"./{datasetName}_data.csv", has_header=False)
    labels = pl.scan_csv(f"./{datasetName}_labels.csv", has_header=False)

    return data.collect().to_numpy(order="c"), labels.collect().to_numpy(order="c")


def load_from_csv_batched(datasetName, batch_size=50000):
    """
    The dataset name is either "training" or "test".
    The batch size is an int representing the size of each batch.

    Returns an iterator tuple (X,y)
    X is a NumPy array with rows being a flattened image (row major), and columns a pixel
    y is the associated label. The label is an id that is associated with a specific character.
    """

    start = 0
    while True:
        # TODO: Find better way to handle breaking.
        try:
            data = pl.scan_csv(
                f"./{datasetName}_data.csv", has_header=False, skip_rows_after_header=start, n_rows=batch_size)
            labels = pl.scan_csv(
                f"./{datasetName}_labels.csv", has_header=False, skip_rows_after_header=start, n_rows=batch_size)
            start += batch_size
            yield data.collect().to_numpy(order="c"), labels.collect().to_numpy(order="c")
        except:
            break


# Keys from 0 to 62, values from 0 to 9, A to Z and a to z
_label_to_char_dict = {label: chr(ascii) for label, ascii in enumerate(itertools.chain(
    range(48, 58), range(65, 91), range(97, 123)))}


def label_to_char(label):
    """
    Converts the given numerical label into an ascii character.
    """
    return _label_to_char_dict[label]
