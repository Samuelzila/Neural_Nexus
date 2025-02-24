import polars as pl


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


def training_batched(batch_size=100):
    """
    The batch size is an int representing the size of each batch.

    Returns an iterator tuple (X,y)
    X is a NumPy array with rows being a flattened image (row major), and columns a pixel
    y is the associated label. The label is an id that is associated with a specific character.
    """
    return load_from_csv_batched("training", batch_size)


def test_batched(batch_size=100):
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


def load_from_csv_batched(datasetName, batch_size=100):
    """
    The dataset name is either "training" or "test".
    The batch size is an int representing the size of each batch.

    Returns an iterator tuple (X,y)
    X is a NumPy array with rows being a flattened image (row major), and columns a pixel
    y is the associated label. The label is an id that is associated with a specific character.
    """
    data = pl.read_csv_batched(
        f"./{datasetName}_data.csv", has_header=False, batch_size=batch_size)
    labels = pl.read_csv_batched(
        f"./{datasetName}_labels.csv", has_header=False, batch_size=batch_size)

    while (data_batch := data.next_batches(1)) and (label_batch := labels.next_batches(1)):
        for X, y in zip(data_batch, label_batch):
            yield X.to_numpy(order="c"), y.to_numpy(order="c")
