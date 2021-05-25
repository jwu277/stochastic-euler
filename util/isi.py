### Utility for ISI calculations ###


import numpy as np


## Computes ISIs from spike times
def isi(st):
    return np.diff(st)

