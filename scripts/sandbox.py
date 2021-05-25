from neurons.bm import BinaryMarkov
from util import isi
import numpy as np
import matplotlib.pyplot as plt


def main():

    pqs = 0.1
    psq = 0.3
    dt = 1.0

    neuron = BinaryMarkov(pqs, psq, dt)

    _isi = isi.isi(neuron.spike_times(1000000))

    # Expected
    t = np.arange(0, 160, 0.1)
    p = (np.exp(-t * pqs / dt) - np.exp(-t * psq / dt)) / (dt * (1 / pqs - 1 / psq))
    plt.plot(t, p * len(_isi), 'k--')
    # Actual
    plt.hist(_isi, bins=200)

    plt.show()


if __name__ == "__main__":
    main()

