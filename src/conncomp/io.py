"""Reading and writing coefficient vectors (one whitespace-separated vector per line)."""

import numpy as np


def save_coefs(fname, rows):
    with open(fname, "w") as f:
        for x in rows:
            f.write(" ".join(str(y) for y in list(x)) + "\n")


def load_coefs(fname):
    with open(fname) as f:
        return [np.array([np.float64(x) for x in line.split()]) for line in f if line.strip()]
