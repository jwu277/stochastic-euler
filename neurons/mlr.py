### Model of Morris-Lecar neuron near fixed point ###


import numpy as np
from scipy.stats import norm


class MorrisLecarR:


    ## Neuron governed by following equations:
    ## m = 0.5 * (1 + tanh((v - V1) / V2))
    ## alpha = 0.5 * phi * cosh((v - V3) / (2 * V4)) * (1 + tanh((v - V3) / V4))
    ## beta = 0.5 * phi * cosh((v - V3) / (2 * V4)) * (1 - tanh((v - V3) / V4))
    ## dv = f(v, w) dt
    ## dw = g(v, w) dt + h(v, w) dBt
    def __init__(self, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, Nk):

        self._phi = phi
        self._C = C
        self._gL = gL
        self._gCa = gCa
        self._gK = gK
        self._VL = VL
        self._VCa = VCa
        self._VK = VK
        self._V1 = V1
        self._V2 = V2
        self._V3 = V3
        self._V4 = V4
        self._dt = dt
        self._Nk = Nk


    ## x = (v, w)
    ## I does not affect the partial derivatives, so it is excluded
    def _f(self, x):

        v = x[0]
        w = x[1]

        m = 0.5 * (1 + np.tanh((v - self._V1) / self._V2))

        return (-self._gCa * m * (v - self._VCa) - self._gK * w * (v - self._VK) - self._gL * (v - self._VL)) / self._C


    ## x = (v, w)
    def _g(self, x):

        v = x[0]
        w = x[1]

        alpha = 0.5 * self._phi * np.cosh((v - self._V3) / (2 * self._V4)) * (1 + np.tanh((v - self._V3) / self._V4))
        beta = 0.5 * self._phi * np.cosh((v - self._V3) / (2 * self._V4)) * (1 - np.tanh((v - self._V3) / self._V4))

        return alpha * (1 - w) - beta * w


    ## x = (v, w)
    def _h(self, x):

        v = x[0]
        w = x[1]

        alpha = 0.5 * self._phi * np.cosh((v - self._V3) / (2 * self._V4)) * (1 + np.tanh((v - self._V3) / self._V4))
        beta = 0.5 * self._phi * np.cosh((v - self._V3) / (2 * self._V4)) * (1 - np.tanh((v - self._V3) / self._V4))

        return np.sqrt(alpha * (1 - w) + beta * w) / np.sqrt(self._Nk)


    ## Deterministic part of R eqn
    def _a(self, r):
        return 1 / (2 * r) - r


    ## Initializes MLR model parameters
    ## eq = stable fixed point
    ## dv = v increment for pderiv
    ## dw = w increment for pderiv
    ## psic = cutoff psi value
    def init(self, eq, dv, dw):

        self._M = np.array([[(self._f(eq + [dv, 0]) - self._f(eq)) / dv, (self._f(eq + [0, dw]) - self._f(eq)) / dw],
            [(self._g(eq + [dv, 0]) - self._g(eq)) / dv, (self._g(eq + [0, dw]) - self._g(eq)) / dw]])

        self._sigma = self._h(eq)
        self._G = np.array([[0, 0], [0, self._sigma]])

        eig = np.linalg.eig(self._M)[0][0]
        self._lambda = -np.real(eig)
        self._omega = np.imag(eig)
        self._tau = np.sqrt(- self._sigma ** 2 * self._M[0][1] / (2 * self._omega ** 2 * self._M[1][0]))

        self._Qinv = np.array([[-1/self._omega, -(self._M[0][0] + self._lambda) / (self._omega * self._M[1][0])], [0, 1/self._M[1][0]]])
        

    ## One trial to get spiking time
    ## psi0 = initial psi
    ## psic = psi cutoff
    def trial(self, psi0, psic, gensize=10000):

        R0 = (np.sqrt(self._lambda) / self._tau) * np.linalg.norm(np.matmul(self._Qinv, [0, -psi0]))
        Rthresh = (np.sqrt(self._lambda) / self._tau) * np.linalg.norm(np.matmul(self._Qinv, [0, -psic]))

        return self._trial_r(R0, Rthresh, gensize)


    ## One trial to get spiking time
    ## R0 = initial R
    ## Rthresh = R threshold
    ## gensize = number of Gaussian RVs to generate at once
    def _trial_r(self, R0, Rthresh, gensize):
        
        rvs = norm.rvs(size=gensize)
        ctr = 0

        r = R0
        sqdt = np.sqrt(self._dt)

        while r < Rthresh:
            
            r += self._a(r) * self._dt + rvs[ctr % gensize] * sqdt

            ctr += 1

            if ctr % gensize == 0:
                rvs = norm.rvs(size=gensize)
                ctr = 0

        return ctr * self._dt

