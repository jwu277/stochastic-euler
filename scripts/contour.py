from neurons.ml import MorrisLecar
from neurons.mll import MorrisLecarLin
from util import current

import numpy as np

from util import plot
from util.ml import *
from util.dyn import *

import matplotlib.pyplot as plt

import time


THREADS = 12


def gen_neuron(dt, tmax, I_ampl, phi, Nk):

    # C ~ uF
    # g ~ mS
    # V ~ mV
    # t ~ ms
    
    C = 20
    gL = 2.0
    gCa = 4.4
    gK = 8
    VL = -60
    VCa = 120
    VK = -84
    V1 = -1.2
    V2 = 18.0
    V3 = 2.0
    V4 = 30.0

    I = current.constant(tmax, dt, I_ampl)

    return MorrisLecar(I, phi, C, gL, gCa, gK, VL, VCa, VK, V1, V2, V3, V4, dt, stochastic='euler', Nk=Nk)


def main():

    dt = 0.1

    I_ampl = 90
    phi = 0.04
    
    Nk = 1000

    tmax = 2000.0 # all simulations must be less than tmax in duration

    tmax1 = 500.0
    tmax2 = 550.0

    assert tmax2 <= tmax

    # 1. Generate neurons
    neuron = gen_neuron(dt, tmax, I_ampl, phi, Nk)
    mll = neuron.gen_model(MorrisLecarLin)

    # 2. Find fixed point
    eq0 = np.array([-30, 0.13])
    dist = np.array([0.4, 0.005])
    eq = get_fixed_pt(neuron, eq0, tmax1, tmax2, dist, dt)

    if eq is None:
        print("Failed to find fixed point!")
        return
    
    v0, w0 = eq
    print(f'Fixed point: {eq}')

    # 3. Initialize MLL

    dv = 0.2
    dw = 0.004
    mll.init(eq, dv, dw)

    # 4. Plot ellipse

    fig, ax = plt.subplots()

    ax.set_xlabel('v')
    ax.set_ylabel('w')
    
    ax.set_xlim(-40, -10)
    ax.set_ylim(0.075, 0.25)

    cdist = 30
    ax.set_title(f'ULC and FPE Contours | r = {cdist}')

    numpts = 1000
    theta = np.linspace(0, 2 * np.pi, numpts)
    circ = cdist * np.transpose(np.array([np.cos(theta), np.sin(theta)]))
    elpts = mll.new2og(circ, np.zeros(numpts))

    ax.plot(elpts[:,0], elpts[:,1], ls='dotted', c='blue', label='FPE')

    # 5. Plot ULC

    # Run signal in reverse time from t1 to t2
    x0 = np.array([v0, w0 - 0.017])
    t1 = 500
    t2 = 610

    assert t2 <= tmax

    neuron.set_time_dir(-1)
    signal = neuron.signal(t2, x0)[int(t1/dt):]
    neuron.set_time_dir(1)

    plot.pp(signal, ax=ax, label='ULC', c='k', ls='--')

    # 6. Plot equilibrium point
    ax.scatter(*eq, marker='x', c='r', label='Fixed Point')

    # 7. Plot misc points

    # 7a. Psi cutoff point
    psic = cdist / np.linalg.norm(mll.get_Qinv()[:,1])
    ax.scatter(v0, w0 - psic, marker='x', c='orange', label='Psi Cutoff Point')

    # 7b. Alpha points
    alpha = np.array([-3.0, -2.0, 0.0, 1.2])
    alpha_xy = mll.new2og(cdist * np.transpose(np.array([np.cos(alpha), np.sin(alpha)])), np.zeros(alpha.shape))
    ax.scatter(alpha_xy[:,0], alpha_xy[:,1], marker='x', c='g', label=f'$\\alpha$ points', zorder=1.2)

    for i in range(len(alpha)):
        ax.annotate(f'{alpha[i]}', tuple(alpha_xy[i]), zorder=1.2)

    ax.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()

