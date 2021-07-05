from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
from util.ml import *
from util.dyn import *
import matplotlib.pyplot as plt

import time


def trial(phi, Nk, v0, w0, dt, tmax):

    # C ~ uF
    # g ~ mS
    # V ~ mV
    # t ~ ms
    
    I_ampl = 80
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

    x0 = np.array([v0, w0])

    # t_avg = 0.1
    I = current.constant(tmax, dt, I_ampl)
    # I = current.poisson(tmax, t_avg, dt, I_ampl)

    t = time.time()

    neuron = MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='', Nk=Nk)

    print(get_fixed_pt(neuron, x0, 200, tmax, np.array([1.0, 0.01]), dt))

    return neuron.signal(tmax, x0)


def main():

    dt = 0.1
    tmax = 350.0

    Nk = 500
    phi = 0.04

    sig = trial(phi, Nk, -30, 0.13, dt, tmax)

    plot.pp_scatter(sig)
    plt.xlabel('v')
    plt.ylabel('w')
    plt.show()


if __name__ == "__main__":
    main()

