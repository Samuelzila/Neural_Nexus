import pandas as pd
import scipy as sp

data = sp.io.loadmat("./emnist-byclass.mat")["dataset"][0][0]

training = data[0][0][0]
test = data[1][0][0]
conversion_table = data[2]

# Association dictionary between EMNIST labels and ascii.
conversion_dict = {}
for i in conversion_table:
    conversion_dict[i[0]] = i[1]


def to_csv(data):
    df = {i: [] for i in range(28*28)}
    df["label"] = []

    for image, label, writer in zip(*data):
        # Because the label is an array with one element.
        label = label[0]

        ascii = conversion_dict[label]
        # Transpose image to be easier to handle with numpy (now row major instead of column major)
        image = image.reshape(28, 28).T.flatten()

        for i, e in enumerate(image):
            df[i].append(e)
        df["label"].append(chr(ascii))

    return pd.DataFrame(df)


training = to_csv(training)
test = to_csv(test)


training.to_csv("training.csv", index=False)
test.to_csv("test.csv", index=False)
