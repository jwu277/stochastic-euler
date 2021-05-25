### Binary Markov Neuron Model ###


from util import ito
import numpy as np
from scipy.stats import uniform


class BinaryMarkov:

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

        rvs = uniform.rvs(size=int(tmax / self.dt))

        for i in range(0, int(tmax / self.dt)):

            if state == 0:
                if rvs[i] < self.pqs:
                    state = 1

            elif state == 1:
                if rvs[i] < self.psq:
                    state = 0
                    times.append(i * self.dt)

        return np.array(times)

