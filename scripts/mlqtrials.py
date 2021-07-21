from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
import matplotlib.pyplot as plt
from util import dyn

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

    Nk = 1000

    phi = 0.04

    I_ampl = 90
    I = current.constant(tmax_I, dt, I_ampl)

    return MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='euler', Nk=Nk)


def _get_eq(neuron, dt):

    eq0 = np.array([-30, 0.1])
    dist = np.array([0.1, 0.001])

    tmax1 = 1000.0
    tmax2 = 1100.0

    return dyn.get_fixed_pt(neuron, eq0, tmax1, tmax2, dist, dt)


def _trials(mlq, psi0, psic, ntrials):

    data = []

    for i in range(ntrials):
        data.append(mlq.trial(psi0, psic))

    return np.array(data)


def main():

    ## 1. Generate neurons ##
    dt = 0.1
    neuron = _gen_neuron(dt)
    mlq = neuron.prod_mlq()

    ## 2. Get equilibrium point ##
    eq = _get_eq(neuron, dt)
    dv = 0.2
    dw = 0.002

    ## 3. Perform trials ##
    psi0 = 0.001
    psic = 0.02
    ntrials = 100
    data = _trials(mlq, psi0, psic, ntrials)

    ## 4. Plot T distribution ##
    plt.hist(data, bins=10, density=True)

    plt.xlabel('T')
    plt.ylabel('Rel. Freq')

    plt.show()


if __name__ == "__main__":
    main()

