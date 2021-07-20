from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
import matplotlib.pyplot as plt


def trial(dt, method):

    # C ~ uF
    # g ~ mS
    # V ~ mV
    # t ~ ms
    
    C = 20
    gL = 2.0
    gCa = 4.4
    gK = 8
    VL = -60
    VCa = 120
    VK = -84
    V1 = -1.2
    V2 = 18.0
    V3 = 2.0
    V4 = 30.0

    tmax = 200.0

    Nk = 1000

    phi = 0.04

    I_ampl = 90
    I = current.constant(tmax, dt, I_ampl)
    # I = current.poisson(tmax, t_avg, dt, I_ampl)

    x0 = np.array([-30.0, 0.08])

    neuron = MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='', Nk=Nk, method=method)
    return neuron.signal(tmax, x0)


def main():

    plot.pp(trial(0.2, 'SE'), '0.2, SE')
    plot.pp(trial(0.2, 'RK45'), '0.2, RK45')
    plot.pp(trial(0.1, 'SE'), '0.1, SE')
    plot.pp(trial(0.1, 'RK45'), '0.1, RK45')
    plot.pp(trial(0.05, 'SE'), '0.05, SE')
    plot.pp(trial(0.05, 'RK45'), '0.05, RK45')

    plt.xlabel('v')
    plt.ylabel('w')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

