import numpy as np
from tensorflow import keras
from pathlib import Path


def predict():
    root_proc = Path("data/processed")

    x_test = np.load(str(root_proc / "test" / "features"))
    y_test = np.load(str(root_proc / "test" / "labels"))

    model = keras.models.load_model("models/best_model.h5")

    test_loss, test_acc = model.evaluate(x_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)


if __name__ == "__main__":
    predict()
