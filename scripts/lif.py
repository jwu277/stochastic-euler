from neurons.lif import LIF
from util import isi
import numpy as np
import matplotlib.pyplot as plt


def main():

    vt = 2.8
    tau = 1.0
    lam = 0.8
    ae = 3.0
    ai = 2.4
    te_avg = 5.0
    ti_avg = 7.0
    dt = 0.1

    tmax = 100000

    neuron = LIF(vt, tau, lam, ae, ai, te_avg, ti_avg, dt)

    # plt.plot(np.arange(0, tmax + dt, dt), neuron.signal(tmax))
    # plt.show()

    _isi = isi.isi(isi.spike_times_th(neuron.signal(tmax), dt, vt))

    # Actual
    plt.hist(_isi, bins=200)

    plt.show()


if __name__ == "__main__":
    main()

