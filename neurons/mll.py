### Linear model of Morris-Lecar neuron near fixed point ###


import numpy as np
from util import ito


class MorrisLecarLin:


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


    ## Initializes MLQ model parameters
    ## eq = stable fixed point
    ## dv = v increment for pderiv
    ## dw = w increment for pderiv
    ## psic = cutoff psi value
    def init(self, eq, dv, dw):

        self._eq = eq

        self._M = np.array([[(self._f(eq + [dv, 0]) - self._f(eq)) / dv, (self._f(eq + [0, dw]) - self._f(eq)) / dw],
            [(self._g(eq + [dv, 0]) - self._g(eq)) / dv, (self._g(eq + [0, dw]) - self._g(eq)) / dw]])

        self._sigma = self._h(eq)
        G = np.array([[0, 0], [0, self._sigma]])

        eig = np.linalg.eig(self._M)[0][0]
        self._lambda = -np.real(eig)
        self._omega = np.imag(eig)

        self._Q = np.array([[-self._omega, self._M[0][0] + self._lambda], [0, self._M[1][0]]])
        self._Qinv = np.linalg.inv(self._Q)

        self._C = np.matmul(self._Qinv, G)

        # Ellipse stretching and rotation parameters
        u, s, vh = np.linalg.svd(self._Q, full_matrices=True)
        self._el1 = s[0]
        self._el2 = s[1]
        print(np.linalg.cond(self._Q))
        print(s)
        print(u)
        print(self._Q)
        print(np.matmul(u, np.matmul(np.diag(s), vh)))
        self._elphi = np.arctan2(u[1][0], u[0][0])
        print(self._elphi)
        

    # Deterministic Euler part
    # x is a 2-array
    def _a(self, t, x):
        return -self._lambda * x


    # 2D rotation matrix
    # Returns array of rotation matrices if theta is an array
    def _rot2D(self, theta):

        c, s = np.cos(theta), np.sin(theta)
        arr = np.array([[c, -s], [s, c]])

        if np.isscalar(theta):
            return arr
        
        return np.swapaxes(np.swapaxes(arr, 0, 2), 1, 2)


    def _b(self, t, x):
        return np.matmul(self._rot2D(self._omega * t), self._C)


    # Converts og coordinates to new
    # x = array of phase vectors
    # t = array of times
    # Returns array of transformed vectors
    def og2new(self, x, t):
        return np.einsum('ni,nji->nj', np.dot(x - self._eq, np.transpose(self._Qinv)), self._rot2D(self._omega * t))
    

    # Converts new coordinates to og
    # x = array of phase vectors
    # t = array of times
    # Returns array of transformed vectors
    def new2og(self, x, t):
        return np.dot(np.einsum('ni,nji->nj', x, self._rot2D(-self._omega * t)), np.transpose(self._Q)) + self._eq


    # Converts og coordinates to 2-norms in new coordinates
    # Returns array of 2-norms
    def og2dist(self, x):
        return np.linalg.norm(np.dot(x - self._eq, np.transpose(self._Qinv)), axis=1)


    # tmax = time to simulate up to
    # x0 = [v0, w0] = IC
    def signal(self, tmax, x0):

        a = lambda t, x: self._a(t, x)
        b = lambda t, x: self._b(t, x)

        # Essentially apply og2new on a scalar
        x0new = self.og2new(np.array([x0]), np.array([0]))[0]

        xv = ito.sim(a, b, tmax, self._dt, x0new, mat=True)
        tv = np.arange(xv.shape[0]) * self._dt

        return self.new2og(xv, tv)
    

    def get_Q(self):
        return np.copy(self._Q)
    

    def get_Qinv(self):
        return np.copy(self._Qinv)

    
    def get_ellipse_params(self):
        return self._el1, self._el2, self._elphi

