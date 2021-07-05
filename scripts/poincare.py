from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
from util.ml import *
from util.dyn import *
import matplotlib.pyplot as plt

import time


def gen_neuron(dt, tmax):

    # C ~ uF
    # g ~ mS
    # V ~ mV
    # t ~ ms
    
    I_ampl = 90
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

    Nk = 500
    phi = 0.04

    I = current.constant(tmax, dt, I_ampl)

    return MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='', Nk=Nk)


def main():

    dt = 0.1
    tmax = 600.0

    # 1. Generate neuron
    neuron = gen_neuron(dt, tmax)

    # 2. Find fixed point
    tmax1 = 500.0
    tmax2 = 550.0
    eq0 = np.array([-30, 0.13])
    dist = np.array([0.4, 0.005])
    eq = get_fixed_pt(neuron, eq0, tmax1, tmax2, dist, dt)

    if eq is None:
        print("Failed to find fixed point!")
        return
    
    v0, w0 = eq
    print(eq)

    # 3. Generate signal
    # TODO sweep through psi
    x0 = np.array([v0, 0.08])
    signal = neuron.signal(tmax, x0)

    # 4. Truncate signal to one cycle
    tmin = 10 # minimum cycle time, prevents noisy backwash    
    cycle = poincare_cycle(signal, v0, w0, tmin, dt)

    if cycle is None:
        print("Failed to obtain Poincare cycle!")
        return
    
    # 5. Analyze signal

    # 5a. T(psi)
    t_cyc = cycle.shape[0] * dt
    print(f'T(psi): {t_cyc}')

    # 5b. P(psi)
    p_map = cycle[-1,1] # TODO: linear interpolation
    print(f'P(psi): {p_map}')

    # 5c. Determine if spiking or subthreshold
    vthresh = 20
    spikeb = np.amax(cycle[:,0]) > vthresh
    print(f'Spiking: {spikeb}')

    # 6. Plotting
    plot.pp(cycle)
    plt.xlabel('v')
    plt.ylabel('w')

    plt.figure()
    plot.tr(cycle[:,0], dt)
    plt.xlabel('t')
    plt.ylabel('v')

    plt.show()


if __name__ == "__main__":
    main()

