import numpy as np
from util import plot
from util.ml import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def main():

    dt = 0.1

    psection = -30 # Poincare section
    sthresh = 0 # spiking threshold
    vdiff = 2.5 # orbit amplitude minimum

    t1 = int(7700 / dt)
    t2 = int(8100 / dt)

    sig = np.load('picklejar/phi_noise_t2.npy')

    ot = get_orbit_times2(sig, vdiff)

    plot.tr(sig[:,0], dt)
    plt.xlabel('t')
    plt.ylabel('v')

    spiked = True

    for i in range(len(ot) - 1):
        if is_spike2(sig[ot[i]:ot[i+1]], sthresh):
            plt.axvline(ot[i] * dt, c='r', ls='--')
            spiked = True
        elif spiked:
            plt.axvline(ot[i] * dt, c='g', ls='--')
            spiked = False
    
    # Animated pp
    fig, ax = plt.subplots()

    t = np.arange(t1, t2) * dt
    line, = ax.plot(sig[t1:t2,0], sig[t1:t2,1])

    def animate(t_idx):
        line.set_xdata(sig[t1:t_idx,0])
        line.set_ydata(sig[t1:t_idx,1])

    plt.xlabel('v')
    plt.ylabel('w')
    
    ani = animation.FuncAnimation(fig, animate, np.arange(t1, t2, 100))

    plt.show()


if __name__ == "__main__":
    main()

