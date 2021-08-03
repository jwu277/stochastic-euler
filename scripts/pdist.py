from multiprocessing import Pool
from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
from util.ml import *
from util.dyn import *
import matplotlib.pyplot as plt

import time


THREADS = 12


def gen_neuron(dt, tmax, I_ampl, phi, Nk):

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

    I = current.constant(tmax, dt, I_ampl)

    return MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='euler', Nk=Nk)


def cycle(neuron, v0, w0, tmax, dt, tmin, vthresh, psi):

    # 1. Generate signal
    x0 = np.array([v0, w0 - psi])
    signal = neuron.signal(tmax, x0)

    # 2. Truncate signal to one cycle
    cycle = poincare_cycle(signal, v0, w0, tmin, dt)

    if cycle is None:
        # print("Failed to obtain Poincare cycle!")
        return
    
    # 3. Analyze signal

    # 3a. T(psi)
    t_cyc = cycle.shape[0] * dt

    # 3b. P(psi)
    p_map = w0 - cycle[-1,1]

    # 3c. Determine if spiking or subthreshold
    spikeb = np.amax(cycle[:,0]) > vthresh

    return [t_cyc, p_map, spikeb]


def _cycle_wrapper(args):
    return cycle(*args)


def cycles(neuron, v0, w0, tmax, dt, tmin, vthresh, psi, trials=1):
    with Pool(THREADS) as p:
        return np.array(list(filter(lambda v: v is not None, p.map(_cycle_wrapper, [(neuron, v0, w0, tmax, dt, tmin, vthresh, psi)] * trials))))


def main():

    dt = 0.1
    tmax = 250.0

    I_ampl = 90
    phi = 0.04
    
    Nk = 2000000

    tmax1 = 500.0
    tmax2 = 550.0

    # 1. Generate neuron
    neuron = gen_neuron(dt, max(tmax, tmax2), I_ampl, phi, Nk)

    # 2. Find fixed point
    eq0 = np.array([-30, 0.13])
    dist = np.array([0.4, 0.005])
    eq = get_fixed_pt(neuron, eq0, tmax1, tmax2, dist, dt)

    if eq is None:
        print("Failed to find fixed point!")
        return
    
    v0, w0 = eq
    print(f'Fixed point: {eq}')

    # 3. Poincare dsitribution analysis

    tmin = 10 # minimum cycle time, prevents noisy backwash
    vthresh = 20

    psi = 0.018
    trials = 2000

    t = time.time()
    tdata = cycles(neuron, v0, w0, tmax, dt, tmin, vthresh, psi, trials)
    print(f'Cycles time: {time.time() - t}')
    # print(f'P({psi}) = {tdata[0][1]}')

    # 4. Plotting

    bins = 30

    plt.hist(tdata[:,0], density=True, bins=bins)
    plt.title(f'I = {I_ampl} | $\\phi$ = {phi} | $N_k$ = {Nk} | Trials = {trials} | $\\psi$ = {psi}')
    plt.xlabel('Cycle Time')
    plt.ylabel('Relative Frequency')
    plt.figure()

    plt.hist(tdata[:,1], density=True, bins=bins)
    plt.title(f'I = {I_ampl} | $\\phi$ = {phi} | $N_k$ = {Nk} | Trials = {trials} | $\\psi$ = {psi}')
    plt.xlabel('$P(\\psi)$')
    plt.ylabel('Relative Frequency')
    plt.figure()

    plt.scatter(tdata[:,0], tdata[:,1])
    plt.title(f'I = {I_ampl} | $\\phi$ = {phi} | $N_k$ = {Nk} | Trials = {trials} | $\\psi$ = {psi}')
    plt.xlabel('Cycle Time')
    plt.ylabel('$P(\\psi)$')

    plt.figure()

    plt.scatter(tdata[:,1], tdata[:,2])
    plt.title(f'I = {I_ampl} | $\\phi$ = {phi} | $N_k$ = {Nk} | Trials = {trials} | $\\psi$ = {psi}')
    plt.xlabel('$P(\\psi)$')
    plt.ylabel('Spiked?')

    plt.show()


if __name__ == "__main__":
    main()

