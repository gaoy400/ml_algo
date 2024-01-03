import numpy as np


def get_mismatch_rate(label, true_label):
    # Only works for binary class
    return min(np.mean(label == true_label), np.mean(label != true_label))