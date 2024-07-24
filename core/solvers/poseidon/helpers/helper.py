import numpy as np

def linear_minmax_scale(x: int | float, min: int | float, max: int | float) -> float:
    if min == max:
        return 1.0
    if x == np.inf:
        return 1.0
    if x == -np.inf:
        return -1.0
    if x>max:
        return 1.0
    if x<min:
        return -1.0
    return 2*((x - min) / (max - min)) - 1

def sigmoid_scale(x: int | float) -> float:
    return 1 / (1 + np.exp(-x))

def tanh_scale(x: int | float) -> float:
    return np.tanh(x)

