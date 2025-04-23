import polars as pl
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

    return data.collect().to_numpy(order="c"), labels.collect().to_numpy(order="c").flatten()


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
            yield data.collect().to_numpy(order="c"), labels.collect().to_numpy(order="c").flatten()
        except:
            break


# Keys from 0 to 62, values from 0 to 9, A to Z and a to z (deprecated)
_label_to_char_dict62 = {label: chr(ascii) for label, ascii in enumerate(itertools.chain(
    range(48, 58), range(65, 91), range(97, 123)))}

# Keys from 0 to 47, representing the labels in emnist's bymerge dataset
_label_to_char_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M',
                       23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}


def label_to_char(label):
    """
    Converts the given numerical label into an ascii character.
    """
    return _label_to_char_dict[label]
