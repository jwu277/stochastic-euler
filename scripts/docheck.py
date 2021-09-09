from multiprocessing import Pool
from neurons.ml import MorrisLecar
from neurons.mll import MorrisLecarLin
from util import current
import numpy as np
from util import plot
from util.ml import *
from util.dyn import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time


THREADS = 12

cmap_raw = cm.get_cmap('hsv')
cmap = lambda n: cmap_raw(n % 256)


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


def _get_dist(x, t, mll):
    return mll.og2dist(np.array([x]))[0]


# returns validb, distance of offending point (in transformed coordinates)
def _is_valid(signal, v0, w0, dt, tmin, mode, mll):

    idxmin = int(tmin / dt)

    # previous crossing type
    prev = mode

    idx2 = 0 # previous crossing index
    idx = 0

    while idx < signal.shape[0] - 1:

        if signal[idx][0] < v0 and signal[idx+1][0] >= v0:

            # Crossing left to right

            if signal[idx][1] < w0 and not prev:
                prev = True
                idx += idxmin
                idx2 = idx
            else:
                return False, min(_get_dist(signal[idx], idx * dt, mll), _get_dist(signal[idx2], idx2 * dt, mll))
        
        elif signal[idx][0] > v0 and signal[idx+1][0] <= v0:

            # Crossing right to left

            if signal[idx][1] > w0 and prev:
                prev = False
                idx += idxmin
                idx2 = idx
            else:
                return False, min(_get_dist(signal[idx], idx * dt, mll), _get_dist(signal[idx2], idx2 * dt, mll))

        idx += 1

    return True, -1


# tmin = noisy backwash filter time
# mode = True if L, False if L'
def cycle(neuron, v0, w0, tmax, dt, tmin, psi, mode, mll):

    # 1. Generate signal
    x0 = np.array([v0, w0 + (-1 if mode else 1) * psi])
    signal = neuron.signal(tmax, x0)

    # 2. Determine if L, L' crossings are valid
    validb, dist = _is_valid(signal, v0, w0, dt, tmin, mode, mll)

    return [psi, mode, validb, dist]


def _cycle_wrapper(args):
    return cycle(*args)


def cycles(neuron, v0, w0, tmax, dt, tmin, psi_arr, mode, mll, trials=1):

    n = len(psi_arr) * trials

    with Pool(THREADS) as p:
        return np.array(list(p.map(_cycle_wrapper, zip([neuron]*n, [v0]*n, [w0]*n, [tmax]*n, [dt]*n, [tmin]*n, list(psi_arr) * trials, [mode]*n, [mll]*n))))


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

    # 3. Trials

    tmin = 0.2
    trials = 200

    tmax = 120.0

    # 3a. L section
    mode = True
    psi_range = np.linspace(0.0, 0.02, 10)

    t = time.time()
    ldata = cycles(neuron, v0, w0, tmax, dt, tmin, psi_range, mode, mll, trials)
    ldata = ldata[ldata[:,0].argsort()]
    print(f'L time: {time.time() - t}')

    # 3b. L' section
    mode = False
    psip_range = np.linspace(0, 0.02, 10)

    t = time.time()
    lpdata = cycles(neuron, v0, w0, tmax, dt, tmin, psip_range, mode, mll, trials)
    lpdata = lpdata[lpdata[:,0].argsort()]
    print(f'L\' time: {time.time() - t}')

    # 3c. Offending distances
    offdist = np.concatenate((ldata[:,3], lpdata[:,3]))
    offdist = offdist[offdist >= 0]

    # 4. Plotting

    plt.scatter(ldata[::trials,0], np.average(np.reshape(ldata[:,2], (-1, trials)), axis=1))
    plt.title(f'L section | I = {I_ampl} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('$\\psi$')
    plt.ylabel('Proportion of valid oscillations')
    plt.figure()

    plt.scatter(lpdata[::trials,0], np.average(np.reshape(lpdata[:,2], (-1, trials)), axis=1))
    plt.title(f'L\' section | I = {I_ampl} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('$\\psi$')
    plt.ylabel('Proportion of valid oscillations')
    plt.figure()

    plt.hist(offdist)
    plt.title(f'I = {I_ampl} | $N_k$ = {Nk} | Trials = {trials * (len(psi_range) + len(psip_range))}')
    plt.xlabel('Offending Distance')
    plt.ylabel('Occurences')

    plt.show()


if __name__ == "__main__":
    main()

