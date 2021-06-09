from neurons.izhikevich import Izhikevich
from util import isi
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt


## Generates Poisson point process delta function signals
def _poisson(tmax, t_avg, dt):
    rvs = uniform.rvs(size=int(tmax / dt) + 1)
    return (1 / dt) * (rvs < dt / t_avg)


def main():

    a = 0.02
    b = 0.20
    c = -65.0
    d = 8.0
    dt = 5.0e-6

    tmax = 10000.0e-3

    t_avg = 60.0e-6
    I_ampl = 120.0
    I = I_ampl * _poisson(tmax, t_avg, dt)

    neuron = Izhikevich(a, b, c, d, I, dt)

    # plt.plot(np.arange(0, tmax + dt, dt), neuron.signal(tmax)[:,0])
    # plt.show()

    _isi = isi.isi(isi.spike_times_th(neuron.signal(tmax)[:,0], dt, neuron.V_TH))

    # Actual
    plt.hist(_isi, bins=160, range=(0, 6.0e-4))

    plt.show()


if __name__ == "__main__":
    main()

