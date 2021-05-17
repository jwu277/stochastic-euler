## Simulates Ito processes


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def sim(tmax, dt, f0=0):

    # Initialize dB
    dB = np.sqrt(dt) * norm.rvs(size=int(tmax / dt))
    dB[0] = f0
    
    return np.cumsum(dB)


def main():

    tmax = 10
    dt = 0.001

    t = np.arange(0, tmax, dt)
    B = sim(tmax, dt)

    plt.plot(t, B)
    plt.show()


if __name__ == "__main__":
    main()

