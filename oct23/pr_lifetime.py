import numpy as np 
import matplotlib.pyplot as plt
import my_utils as utils

h5_file = '/home/tim/research/oct23_data/20231016_0001.h5'

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
    data_arr = utils.load_state_from_h5(h5_file, state)

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