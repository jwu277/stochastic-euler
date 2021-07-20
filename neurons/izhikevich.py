### Izhikevich Neuron Model ###


from util import ito
import numpy as np


class Izhikevich:


    A = 0.04
    B = 5.0
    C = 140.0
    V_TH = 0.03


    ## Neuron governed by following equations:
    ## dV/dt = AV^2 + BV + C - u + I
    ## du/dt = -a(bV - u)
    ## I is e.g. synaptic input
    ## Voltage resets once a threshold is reached
    def __init__(self, a, b, c, d, I, dt):
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self._I = I
        self._dt = dt


    # tmax = time to simulate up to
    def signal(self, tmax):

        a = lambda t, x: np.array([(self._c - x[0]) / self._dt, self._d / self._dt] if x[0] >= self.V_TH else [self.A * x[0] * x[0] + self.B * x[0] + self.C - x[1] + self._I[int(t / self._dt)], -self._a * (self._b * x[0] - x[1])])
        b = lambda t, x: 0

        return ito.sim(a, b, tmax, self._dt, np.array([self._c, 0]))

