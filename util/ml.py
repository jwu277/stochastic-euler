import numpy as np


# Gets orbit times (indices) by looking at V-crossings
def get_orbit_times(signal, psection):

    t = []
    v = signal[:,0]

    for i in range(len(v) - 1):
        if v[i] <= psection and v[i+1] > psection:
            t.append(i)
    
    return np.array(t)



# Gets orbit times (indices) by looking at V-maxima
# w should be increasing at these maxima
# vdiff = minimum difference between maxima and minima
def get_orbit_times2(signal, vdiff):

    t = []
    v = signal[:,0]
    w = signal[:,1]

    for i in range(len(v) - 2):
        if v[i] < v[i+1] and v[i+1] > v[i+2]:

            # w should be decreasing at next minima for a valid orbit
            j = 0

            while True:

                if v[i+j] > v[i+j+1] and v[i+j+1] < v[i+j+2]:
                    if v[i+1] - v[i+j+1] > vdiff:
                        t.append(i + 1)
                    break

                j += 1
    
    return np.array(t)


# Detects whether an orbit is a spike or quiescence
def is_spike(orbit, sthresh):
    return np.any(orbit[:,0] > sthresh)


# Detects whether an orbit is a spike or quiescence
# Convention that orbits start at maximum value
def is_spike2(orbit, sthresh):
    return orbit[0,0] > sthresh

