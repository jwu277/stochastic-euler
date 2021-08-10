## TODO: determine L and L' cutoffs ##

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
    return np.linalg.norm(mll.og2new(np.array([x]), np.array([t]))[0])


# returns validb, distance of offending point (in transformed coordinates)
def _is_valid(signal, v0, w0, dt, tmin, mode, mll):

    idxmin = int(tmin / dt)

    # previous crossing type
    prev = mode

    pts = [np.concatenate((signal[0], [mode]))]

    idx2 = 0 # previous crossing index
    idx = 0

    while idx < signal.shape[0] - 1:

        if signal[idx][0] < v0 and signal[idx+1][0] >= v0:
            pts.append(np.concatenate((signal[idx], [not prev])))
            # Crossing left to right

            if signal[idx][1] < w0 and not prev:
                prev = True
                idx += idxmin
                idx2 = idx
            else:
                dist = min(_get_dist(signal[idx], idx * dt, mll), _get_dist(signal[idx2], idx2 * dt, mll))
                if dist > 200 and abs(w0 - 0.12936) > 0.00:
                    plt.title(f'A {mode} | {w0}')
                    plot.pp(signal)
                    
                    for j in range(0, int(30 / dt), int(1 / dt)):
                        plt.plot(*signal[j], marker='+', c=cmap(int(30 * j / int(1 / dt))))

                    i = 0
                    for pt in pts:
                        i += 1
                        plt.annotate(str(i), (pt[0], pt[1]))
                        plt.plot(pt[0], pt[1], 'ro' if pt[2] else 'go')
                    plt.show()
                return False, dist
                # return False, _get_dist(signal[idx], idx * dt, mll)
        
        elif signal[idx][0] > v0 and signal[idx+1][0] <= v0:
            pts.append(np.concatenate((signal[idx], [not prev])))
            # Crossing right to left

            if signal[idx][1] > w0 and prev:
                prev = False
                idx += idxmin
                idx2 = idx
            else:
                dist = min(_get_dist(signal[idx], idx * dt, mll), _get_dist(signal[idx2], idx2 * dt, mll))
                if dist > 200 and abs(w0 - 0.12936) > 0.00:
                    plt.title(f'B {mode} | {w0}')
                    plot.pp(signal)

                    for j in range(0, int(30 / dt), int(1 / dt)):
                        plt.plot(*signal[j], marker='+', c=cmap(int(30 * j / int(1 / dt))))

                    i = 0
                    for pt in pts:
                        i += 1
                        plt.annotate(str(i), (pt[0], pt[1]))
                        plt.plot(pt[0], pt[1], 'ro' if pt[2] else 'go')
                    plt.show()
                return False, dist
                # return False, _get_dist(signal[idx], idx * dt, mll)

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

    # print(np.linalg.norm(mll.og2new(np.array([eq + [0, 0.002]]), np.array([0]))))
    # print(np.linalg.norm(mll.og2new(np.array([eq + [0, 0.005]]), np.array([0]))))
    # print(np.linalg.norm(mll.og2new(np.array([eq + [0, 0.01]]), np.array([0]))))
    # print(np.linalg.norm(mll.og2new(np.array([eq + [0, 0.02]]), np.array([0]))))
    # print(np.linalg.norm(mll.og2new(np.array([eq + [0, 0.04]]), np.array([0]))))

    # return

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
    psi_range = np.linspace(0, 0.1, 10)

    t = time.time()
    lpdata = cycles(neuron, v0, w0, tmax, dt, tmin, psi_range, mode, mll, trials)
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
    plt.title(f'I = {I_ampl} | $N_k$ = {Nk} | Trials = {trials}')
    plt.xlabel('Offending Transformed Distance')
    plt.ylabel('Occurences')

    plt.show()


if __name__ == "__main__":
    main()

