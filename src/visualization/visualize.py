from src.data.make_dataset import readucr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = Path("data/raw")


def main():
    x_train, y_train = readucr(DATA_ROOT / "FordA_TRAIN.tsv")
    x_test, y_test = readucr(DATA_ROOT / "FordA_TEST.tsv")

    classes = np.unique(np.concatenate((y_train, y_test),
                                       axis=0))

    plt.figure()
    for class_ in classes:
        c_x_train = x_train[y_train == class_]
        plt.plot(c_x_train[0], label=f"class \"{class_}\"")
    plt.title("One timeseries example for each class")
    plt.legend(loc="best")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
