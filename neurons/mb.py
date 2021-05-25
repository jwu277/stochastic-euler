### Markov Binary Neuron Model ###


import ito
import numpy as np
from scipy.stats import uniform


class MarkovBinary:

    # pqs = quiescence --> spike transition probability
    # psq = spike --> quiescence transition probability
    # dt = time increment between transitions
    def __init__(self, pqs, psq, dt):
        self.pqs = pqs
        self.psq = psq
        self.dt = dt


    # tmax = time to simulate up to
    def spike_times(self, tmax):

        times = []
        state = 0

        rvs = uniform.rvs(size=(tmax // self.dt))

        for i in range(0, tmax // self.dt):

            if state == 0:
                if rvs[i] < self.pqs:
                    times.append(i * self.dt)
                    state = 1

            elif state == 1:
                if rvs[i] < self.psq:
                    state = 0

        return np.array(times)

