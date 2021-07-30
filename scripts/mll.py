import numpy as np
import matplotlib.pyplot as plt

from neurons.ml import MorrisLecar

from util import current
from util import dyn
from util import plot

import time


def _gen_neuron(dt):

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
    
    tmax_I = 1500.0

    Nk = 10000

    phi = 0.04

    I_ampl = 90
    I = current.constant(tmax_I, dt, I_ampl)

    return MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='euler', Nk=Nk)


def _get_eq(neuron, dt):

    eq0 = np.array([-30, 0.13])
    dist = np.array([0.1, 0.001])

    tmax1 = 1000.0
    tmax2 = 1100.0

    return dyn.get_fixed_pt(neuron, eq0, tmax1, tmax2, dist, dt)


def main():

    ## 1. Generate neurons ##
    dt = 0.1
    neuron = _gen_neuron(dt)
    mll = neuron.gen_mll()

    ## 2. Get equilibrium point ##
    eq = _get_eq(neuron, dt)
    dv = 0.2
    dw = 0.002

    ## 3. Initialize MLQ ##
    mll.init(eq, dv, dw)

    ## 4. Generate signal ##
    tmax = 2000.0
    x0 = np.array([-25, 0.13])

    t = time.time()
    x = mll.signal(tmax, x0)
    print(f'Computation time: {time.time() - t}')

    ## 5. Plot results ##

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

