import matplotlib.pyplot as plt

from neurons.ml import MorrisLecar
from neurons.modelb import ModelB

from util import current

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


def main():

    ## 1. Generate model B neuron ##

    dt = 0.1
    neuron = _gen_neuron(dt)
    mb = neuron.gen_model(ModelB)

    ## 2. Simulate model B ##

    tmax = 20000.0
    psi0 = 0.01

    t = time.time()
    st, tv, psiv = mb.sim_st(tmax, psi0)
    print(f'Simulation time: {time.time() - t}')

    ## 3. Plot results ##

    plt.plot(tv, psiv)
    plt.xlabel('t')
    plt.ylabel('$\psi$')
    plt.show()


if __name__ == "__main__":
    main()

