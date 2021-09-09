from neurons.izhikevich import Izhikevich
import numpy as np
from util import isi
from util import current
from scipy.stats import uniform
import matplotlib.pyplot as plt


def main():

    a = 0.02
    b = 0.20
    c = -65.0
    d = 8.0
    dt = 5.0e-6

    tmax = 10000.0e-3

    t_avg = 60.0e-6
    I_ampl = 120.0
    I = I_ampl * current.poisson(tmax, t_avg, dt)

    neuron = Izhikevich(a, b, c, d, I, dt)

    sig = neuron.signal(tmax)[:,0]

    plt.plot(np.arange(0, tmax, dt), sig)
    plt.show()

    _isi = isi.isi(isi.spike_times_th(sig, dt, neuron.V_TH))

    # Actual
    plt.hist(_isi, bins=160, range=(0, 6.0))

    plt.show()


if __name__ == "__main__":
    main()

