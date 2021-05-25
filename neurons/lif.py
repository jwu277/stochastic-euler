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
        self.vt = vt
        self.tau = tau
        self.lam = lam
        self.ae = ae
        self.ai = ai
        self.te_avg = te_avg
        self.ti_avg = ti_avg
        self.dt = dt


    # tmax = time to simulate up to
    def signal(self, tmax):

        te = uniform.rvs(size=int(tmax / self.dt) + 1) < (self.dt / self.te_avg)
        ti = uniform.rvs(size=int(tmax / self.dt) + 1) < (self.dt / self.ti_avg)

        a = lambda t, x: -x / self.dt if x >= self.vt else -self.lam * x / self.tau + self.ae / self.tau / self.dt * te[int(t / self.dt)] - self.ai / self.tau / self.dt * ti[int(t / self.dt)]
        b = lambda t, x: 0

        return ito.sim(a, b, tmax, self.dt)

