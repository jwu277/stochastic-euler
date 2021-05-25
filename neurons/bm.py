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
    # Alternates between -1 and +1
    def signal(self, tmax):

        sig = [-1.0]

        rvs = uniform.rvs(size=int(tmax / self.dt))

        for i in range(0, int(tmax / self.dt)):
            if sig[-1] < 0:
                sig.append(1.0 if rvs[i] < self.pqs else -1.0)
            else:
                sig.append(-1.0 if rvs[i] < self.psq else 1.0)

        return np.array(sig)

