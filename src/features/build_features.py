from src.data.make_dataset import readucr
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle

DATA_RAW = Path("data/raw")
DATA_PROC = Path("data/processed")


def main():
    x_train, y_train = readucr(DATA_RAW / "FordA_TRAIN.tsv")
    x_test, y_test = readucr(DATA_RAW / "FordA_TEST.tsv")

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Now we shuffle the training set because we will be using the validation_split option later when training.
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]
    y_train = y_train[idx]

    # Standardize the labels to positive integers. The expected labels will then be 0 and 1.
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    def save_to_file(path: Path, fname: str, arr: np.ndarray):
        path.mkdir(exist_ok=True, parents=True)
        np.save(str(path / fname), arr)

    save_to_file(path=DATA_PROC / "train", fname="features", arr=x_train)
    save_to_file(path=DATA_PROC / "train", fname="labels", arr=y_train)
    save_to_file(path=DATA_PROC / "test", fname="features", arr=x_test)
    save_to_file(path=DATA_PROC / "test", fname="labels", arr=y_test)


if __name__ == "__main__":
    main()
