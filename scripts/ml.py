from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
import matplotlib.pyplot as plt

import time


def main():

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

    dt = 0.1
    tmax = 1000.0

    Nk = 1000

    phi = 0.04

    t_avg = 0.1
    I_ampl = 90
    I = current.constant(tmax, dt, I_ampl)
    # I = current.poisson(tmax, t_avg, dt, I_ampl)

    x0 = np.array([-30.0, 0.15])

    t = time.time()

    neuron = MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='', Nk=Nk)
    x = neuron.signal(tmax, x0)

    print(time.time() - t)

    plot.tr(x[:,0], dt)
    plt.xlabel('t')
    plt.ylabel('v')

    plt.figure()

    plot.pp(x)
    # plot.pp_scatter(x, c='heatmap', s=10)
    # plot.pp_scatter(x, step=20, s=40, c='r')
    plt.xlabel('v')
    plt.ylabel('w')
    plt.show()


if __name__ == "__main__":
    main()

