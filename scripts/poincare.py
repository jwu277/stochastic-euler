from multiprocessing import Pool
from neurons.ml import MorrisLecar
from util import current
import numpy as np
from util import plot
from util.ml import *
from util.dyn import *
import matplotlib.pyplot as plt

import time


THREADS = 6


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

    return [psi, t_cyc, p_map, spikeb]


def _cycles_psi(neuron, v0, w0, tmax, dt, tmin, vthresh, psi, trials):

    trial = []

    for i in range(trials):
        res = cycle(neuron, v0, w0, tmax, dt, tmin, vthresh, psi)
        if res is not None:
            trial.append(res)
    
    if len(trial) > 0:
        return np.average(trial, axis=0)


def _cycles_psi_wrapper(args):
    return _cycles_psi(*args)


def cycles(neuron, v0, w0, tmax, dt, tmin, vthresh, psi_arr, trials=1):

    n = len(psi_arr)

    with Pool(THREADS) as p:
        return np.array(list(filter(lambda v: v is not None, p.map(_cycles_psi_wrapper, zip([neuron]*n, [v0]*n, [w0]*n, [tmax]*n, [dt]*n, [tmin]*n, [vthresh]*n, psi_arr, [trials]*n)))))


# Gets crossing point of the P(psi) map
# Deterministic sim
def get_ulc(neuron, v0, w0, tmax, dt, tmin, vthresh, psi_arr):

    neuron.set_noise(False)
    pdata = cycles(neuron, v0, w0, tmax, dt, tmin, vthresh, psi_arr, trials=1)
    neuron.set_noise(True)

    return pdata[:,0][np.where(np.diff(np.sign(pdata[:,2] - pdata[:,0])))[0][0]]


# Histogram of spiking proportion vs P(psi)
# Data = list of [p_map, spikeb]
# binsize = histogram bin size
# Returns bins and bin heights
def spike_hist(data, binsize):

    minp = np.amin(data[:,0])
    maxp = np.amax(data[:,0])

    # Init arrays
    hist_spike = np.zeros(int((maxp - minp) / binsize) + 1)
    hist_tot = np.zeros(len(hist_spike))

    for entry in data:

        idx = int((entry[0] - minp) / binsize)

        hist_spike[idx] += entry[1]
        hist_tot[idx] += 1
    
    return np.arange(len(hist_spike)) * binsize, hist_spike / hist_tot


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

    # 3. Get unstable limit cycle psi
    tmin = 10 # minimum cycle time, prevents noisy backwash
    vthresh = 20
    psi_check = np.arange(0, 0.03, 0.001)

    psi_ulc = get_ulc(neuron, v0, w0, tmax, dt, tmin, vthresh, psi_check)
    print(f'psi_ulc: {psi_ulc}')

    # 4. Poincare analysis
    # psi_range = psi_ulc +- delta

    delta = 0.02
    psi_range = np.linspace(max(psi_ulc - delta, 0), psi_ulc + delta, 40)
    trials = 50

    t = time.time()
    pdata = cycles(neuron, v0, w0, tmax, dt, tmin, vthresh, psi_range, trials)
    print(f'Cycles time: {time.time() - t}')

    # 5. Plotting

    plt.scatter(pdata[:,0], pdata[:,1])
    plt.title(f'I = {I_ampl} | $\\phi$ = {phi} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('$\\psi$')
    plt.ylabel('Average Cycle Time')
    plt.figure()

    plt.scatter(pdata[:,0], pdata[:,2], c='k')
    plt.plot(pdata[:,0], pdata[:,0], ls='--', c='r')
    plt.title(f'I = {I_ampl} | $\\phi$ = {phi} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('$\\psi$')
    plt.ylabel('Average $P(\\psi)$')
    plt.figure()
    
    plt.scatter(pdata[:,0], pdata[:,3])
    plt.title(f'I = {I_ampl} | $\\phi$ = {phi} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('$\\psi$')
    plt.ylabel('Spiking Frequency')
    plt.figure()

    # Spiking probability vs P(psi)

    binsize = 0.001

    plt.bar(*spike_hist(pdata[:, [2, 3]], binsize), width=binsize, align='edge')
    plt.title(f'I = {I_ampl} | $\\phi$ = {phi} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('$P(\\psi)$')
    plt.ylabel('Spiking Frequency')

    plt.show()


if __name__ == "__main__":
    main()

