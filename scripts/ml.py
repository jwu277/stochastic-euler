import numpy as np
import matplotlib.pyplot as plt

from neurons.ml import MorrisLecar

from util import current
from util import plot

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
    tmax = 400.0

    Nk = 2000

    phi = 0.04

    I_ampl = 90 # current amplitude

    I = current.constant(tmax, dt, I_ampl)

    x0 = np.array([-30.0, 0.15]) # initial point

    stochastic_method = 'euler'

    t = time.time()

    neuron = MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic=stochastic_method, Nk=Nk)
    x = neuron.signal(tmax, x0)

    print(f'Computation time: {time.time() - t}')

    # 1. v time response
    plot.tr(x[:,0], dt)
    plt.title('$v(t)$ signal')
    plt.xlabel('t')
    plt.ylabel('v')
    plt.figure()

    # 2. Phase portrait
    plot.pp(x)
    # plot.pp_scatter(x, c='heatmap', s=10)
    # plot.pp_scatter(x, step=20, s=40, c='r')
    plt.title('Phase Portrait')
    plt.xlabel('v')
    plt.ylabel('w')
    plt.show()


if __name__ == "__main__":
    main()

