import numpy as np 
import matplotlib.pyplot as plt 
import h5py


h5_files = ['/home/tim/research/oct23_data/massgui_export_20231014_0006.h5',
            '/home/tim/research/oct23_data/massgui_export_20231015_0000.h5',
            '/home/tim/research/oct23_data/massgui_export_20231016_0001.h5',
            '/home/tim/research/oct23_data/massgui_export_20231017_0000.h5',
            '/home/tim/research/oct23_data/massgui_export_20231017_0001.h5',
            '/home/tim/research/oct23_data/massgui_export_20231018_0000.h5']

def load_states_from_h5(h5_file, states):
    h5 = h5py.File(h5_file, 'r')
    n = 0
    for key in h5.keys():
        for state in states:
            n += len(h5[key][state]['energy'])

    energies = np.zeros(n)
    unixnanos = np.zeros(n)
    seconds_after_last_external_trigger = np.zeros(n)

    i0 = 0
    for key in h5.keys():
        for state in states:
            g = h5[key][state]
            n = len(g['energy'])
            energies[i0:i0+n] = g['energy']
            unixnanos[i0:i0+n] = g['unixnano']
            seconds_after_last_external_trigger[i0:i0+n] = g['seconds_after_last_external_trigger']
            i0+=n
    return np.vstack([energies,unixnanos,seconds_after_last_external_trigger])

def midpoints(x):
    return x[:-1]+(x[1]-x[0])/2

print(h5py.File(h5_files[0], 'r')[1])