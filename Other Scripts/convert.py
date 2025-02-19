import polars as pl
import scipy as sp

data = sp.io.loadmat("./emnist-byclass.mat")["dataset"][0][0]

training = data[0][0][0]
test = data[1][0][0]
conversion_table = data[2]


# Association dictionary between EMNIST labels and ascii.
conversion_dict = {}
for i in conversion_table:
    conversion_dict[i[0]] = i[1]


def to_dataframe(data):
    df = {str(i): [] for i in range(28*28)}
    df["label"] = []

    count = 0
    for image, label, writer in zip(*data):
        # Because the label is an array with one element.
        label = label[0]

        ascii = conversion_dict[label]
        # Transpose image to be easier to handle with numpy (now row major instead of column major)
        image = image.reshape(28, 28).T.flatten()

        for i, e in enumerate(image):
            df[str(i)].append(e)
        df["label"].append(chr(ascii))

        count += 1
        print(count)
        # Limit number of elements in dataframe
        # if count > 10000:
        #     break

    return pl.DataFrame(df)


training = to_dataframe(training)
training.write_csv("training.csv")

# test = to_dataframe(test)
# test.write_csv("test.csv")
