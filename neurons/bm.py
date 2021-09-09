### Binary Markov Neuron Model ###


import numpy as np
from scipy.stats import uniform


class BinaryMarkov:


    # pqs = quiescence --> spike transition probability
    # psq = spike --> quiescence transition probability
    # dt = time increment between transitions
    def __init__(self, pqs, psq, dt):
        self._pqs = pqs
        self._psq = psq
        self._dt = dt


    # tmax = time to simulate up to
    # Alternates between -1 and +1
    def signal(self, tmax):

        sig = [-1.0]

        rvs = uniform.rvs(size=int(tmax / self._dt))

        for i in range(0, int(tmax / self._dt)):
            if sig[-1] < 0:
                sig.append(1.0 if rvs[i] < self._pqs else -1.0)
            else:
                sig.append(-1.0 if rvs[i] < self._psq else 1.0)

        return np.array(sig)

