import numpy as np 
import matplotlib.pyplot as plt 
import h5py

h5_file = '/home/tim/research/oct23_data/massgui_export_20231016_0001.h5'

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

def pcm_edges(x):
    return np.concatenate((x,[x[1]-x[0]]))

# check for cal overlap
states = ['S','F','W']
state_labels = ['Nd','Pr','Cal']

fig,ax = plt.subplots(1,1)
fig.suptitle('Nd 2.25 kV, Pr 2.15 kV, Cal 5 kV')

binsize = 1
e_bin_edges = np.arange(750,2000,binsize)

for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_states_from_h5(h5_file, state)

    counts,_ = np.histogram(data_arr[0,:],e_bin_edges)
    centers = midpoints(e_bin_edges)
    int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
    counts = counts/int_time

    ax.plot(centers,counts,label=f'{sl}')
    ax.set_ylabel(f'[counts /s /{binsize} eV bin]')
    ax.legend()
    ax.minorticks_on()
plt.xlabel('Energy [eV]')

plt.show()