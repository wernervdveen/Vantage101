import numpy as np
import matplotlib.pyplot as plt


def readucr(filename):
    data = np.loadtxt(str(filename), delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)