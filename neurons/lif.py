### Binary Markov Neuron Model ###


from util import ito
import numpy as np
from scipy.stats import uniform


class LIF:


    ## Neuron governed by following equation:
    ## tau dV/dt = -lam * V + ae * Se - ai * Si
    ## vt is the threshold voltage
    ## Se and Si are Poisson point process signals
    ## te_avg and ti_avg are the mean Poisson times for those signals, respectively
    def __init__(self, vt, tau, lam, ae, ai, te_avg, ti_avg, dt):
        self._vt = vt
        self._tau = tau
        self._lam = lam
        self._ae = ae
        self._ai = ai
        self._te_avg = te_avg
        self._ti_avg = ti_avg
        self._dt = dt


    # tmax = time to simulate up to
    def signal(self, tmax):

        te = uniform.rvs(size=int(tmax / self._dt) + 1) < (self._dt / self._te_avg)
        ti = uniform.rvs(size=int(tmax / self._dt) + 1) < (self._dt / self._ti_avg)

        a = lambda t, x: -x / self._dt if x >= self._vt else -self._lam * x / self._tau + self._ae / self._tau / self._dt * te[int(t / self._dt)] - self._ai / self._tau / self._dt * ti[int(t / self._dt)]
        b = lambda t, x: 0

        return ito.sim(a, b, tmax, self._dt)

