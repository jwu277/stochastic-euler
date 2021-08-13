## Ellipse statistics for fixed starting point ##

from multiprocessing import Pool
from neurons.ml import MorrisLecar
from neurons.mll import MorrisLecarLin
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


def _get_dist(x, mll):
    return mll.og2dist(x)


# Detects exit point
# Returns: psi, theta, t
#          None if value is not applicable
def _get_exit(signal, v0, w0, r, dt, mll):

    psi, theta, t = np.NAN, np.NAN, np.NAN

    wc = w0 - r / np.linalg.norm(mll.get_Qinv()[:,1])
    dist = mll.og2dist(signal)

    for i in range(signal.shape[0] - 1):

        # Check D crossing
        if signal[i][0] <= v0 and signal[i+1][0] > v0 and signal[i][1] <= wc:
            t = i * dt
            psi = w0 - signal[i][1]
            break
        
        # Check E crossing
        if signal[i][0] > v0 and dist[i] <= r and dist[i+1] > r:
            t = i * dt
            newvec = mll.og2new(np.array([signal[i]]), np.array([t]))[0]
            theta = np.arctan2(newvec[1], newvec[0])
            break

    return psi, theta, t


# alpha = starting point on ellipse
# Return: psi = exit psi on D (NaN if exits on E)
#         theta = exit psi on E (NaN if exits on D)
#         t = exit time
def cycle(neuron, v0, w0, r, tmax, dt, mll, alpha):

    # 1. Generate signal
    x0 = mll.new2og(np.array([r * np.array([np.cos(alpha), np.sin(alpha)])]), np.array([0]))[0]

    iters = 0

    while True:

        signal = neuron.signal(tmax, x0)
        psi, theta, t = _get_exit(signal, v0, w0, r, dt, mll)

        if not (np.isnan(psi) and np.isnan(theta)):
            break
        
        iters += 1
        x0 = signal[-1]

    t += iters * tmax

    return np.array([alpha, psi, theta, t])


def _cycle_wrapper(args):
    return cycle(*args)


def cycles(neuron, v0, w0, r, tmax, dt, mll, alpha, trials=1):
    with Pool(THREADS) as p:
        return np.array(list(p.map(_cycle_wrapper, [(neuron, v0, w0, r, tmax, dt, mll, alpha)] * trials)))


def main():

    dt = 0.1

    I_ampl = 90
    phi = 0.04
    
    Nk = 1000

    tmax1 = 500.0
    tmax2 = 550.0

    # 1. Generate neurons
    neuron = gen_neuron(dt, tmax2, I_ampl, phi, Nk)
    mll = neuron.gen_model(MorrisLecarLin)

    # 2. Find fixed point
    eq0 = np.array([-30, 0.13])
    dist = np.array([0.4, 0.005])
    eq = get_fixed_pt(neuron, eq0, tmax1, tmax2, dist, dt)

    if eq is None:
        print("Failed to find fixed point!")
        return
    
    v0, w0 = eq
    print(f'Fixed point: {eq}')

    dv = 0.2
    dw = 0.004
    mll.init(eq, dv, dw)

    # 3. Trials TODO

    r = 30
    alpha = 0.0

    tmax = 500.0
    trials = 1200

    assert tmax <= tmax2

    data = cycles(neuron, v0, w0, r, tmax, dt, mll, alpha, trials)
    data_d = data[np.nonzero(np.logical_not(np.isnan(data[:,1])))][:,(0,1,3)]
    data_e = data[np.nonzero(np.logical_not(np.isnan(data[:,2])))][:,(0,2,3)]

    # 4. Plotting

    plt.scatter(data_d[:,1], data_d[:,2])
    plt.title(f'D Map | $\\alpha = {alpha}$ | I = {I_ampl} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('$\\psi$')
    plt.ylabel('Exit Time')
    plt.figure()

    plt.scatter(data_e[:,1], data_e[:,2])
    plt.title(f'E Map | $\\alpha = {alpha}$ | I = {I_ampl} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('$\\theta$')
    plt.ylabel('Exit Time')

    plt.show()


if __name__ == "__main__":
    main()

