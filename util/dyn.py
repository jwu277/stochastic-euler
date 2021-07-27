import numpy as np


# Gets fixed point for neuron
# x0 = starting pt
# tmax1 = pseudo-Cauchy sequence start time
# tmax2 = pseudo-Cauchy sequence end time
# dist = distance threshold to establish limit
def get_fixed_pt(neuron, x0, tmax1, tmax2, dist, dt):

    neuron.set_noise(False)

    # 1. Check for stable eq
    x = neuron.signal(tmax2, x0)
    eq = _detect_eq(x[int(tmax1 / dt):], dist)

    if eq is not None:
        neuron.set_noise(True)
        return eq

    # 2. Check for unstable eq
    neuron.set_time_dir(-1)
    x = neuron.signal(tmax2, x0)
    eq = _detect_eq(x[int(tmax1 / dt):], dist)

    neuron.set_time_dir(1)
    neuron.set_noise(True)

    return eq


# Detects equilibrium point based on Cauchy sequencing
# dist = threshold distance
# Returns None if no equilibrium detected
def _detect_eq(signal, dist):

    for i in range(signal.shape[0]):
        for j in range(i+1, signal.shape[0]):
            if np.any(abs(signal[i] - signal[j]) > dist):
                return None

    return signal[-1]

