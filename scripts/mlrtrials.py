from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt

from neurons.ml import MorrisLecar
from neurons.mlr import MorrisLecarR

from util import current
from util import dyn

import time


THREADS = 12


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

    Nk = 2000000

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


def _trial_wrapper(args):
    return args[0].trial(args[1], args[2])


def _trials(mlr, psi0, psic, ntrials):
    with Pool(THREADS) as p:
        return np.array(list(p.map(_trial_wrapper, [(mlr, psi0, psic)] * ntrials)))


def main():

    t = time.time()

    ## 1. Generate neurons ##
    dt = 0.1
    neuron = _gen_neuron(dt)
    mlr = neuron.gen_model(MorrisLecarR)

    ## 2. Get equilibrium point ##
    eq = _get_eq(neuron, dt)
    dv = 0.2
    dw = 0.002

    ## 3. Initialize MLQ ##
    mlr.init(eq, dv, dw)
    print(f'Setup time: {time.time() - t}')

    ## 4. Perform trials ##
    
    # Let L(psi) be the point on the Poincare section L parametrized by psi
    psi0 = 0.001 # starting point = L(psi0)
    psic = 0.01 # cutoff distance = distance of L(psic) in the transformed MLR space (not v,w space)
    ntrials = 50000
    
    # Time taken for system to exit (i.e. exceed cutoff distance) is measured
    
    t = time.time()
    data = _trials(mlr, psi0, psic, ntrials)
    print(f'Trial time: {time.time() - t}')

    ## 5. Plot T distribution ##

    plt.hist(data, bins=150, density=True, rwidth=1)

    plt.title(f'$\\psi_0 = {psi0}$ | $\\psi_c$ = {psic}')
    plt.xlabel('T')
    plt.ylabel('Rel. Freq')

    plt.show()


if __name__ == "__main__":
    main()

