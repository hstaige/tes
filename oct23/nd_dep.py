import numpy as np 
import matplotlib.pyplot as plt 
import h5py
# plt.ion()
# plt.close('all')

h5_file1 = '/home/tim/research/oct23_data/massgui_export_20231016_0001.h5'
h5_file2 = '/home/tim/research/oct23_data/massgui_export_20231017_0000.h5'

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

t_bin_edges = np.arange(0,1.003,0.003)
e_bin_edges = np.arange(750,2000,1)
state1 = ['S','T','U','X']
state2 = ['B','C']
data_arr = np.hstack((load_states_from_h5(h5_file1, state1),load_states_from_h5(h5_file2,state2)))
counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
#plt.plot(e_bin_edges[:-1],np.sum(counts,axis=1))
plt.pcolormesh(t_bin_edges,e_bin_edges,counts)


# Current density dependance
states = ['B','C','D','Z','Y']
state_labels = ['10mA','15mA','20mA','30mA','40mA']

fig,ax = plt.subplots(len(states),1)
fig.suptitle('Nd Density Dependance')
e_slices = [842,1203,1241]
e_labels = ['E2M3','Ni-like','Co-like']
binsize = .003
t_bin_edges = np.arange(0,1.003,binsize)

for i,state,sl in zip(range(len(states)),states,state_labels):

    if i < 3: # not proud of this
        data_arr = load_states_from_h5(h5_file2, state)
    else:
        data_arr = load_states_from_h5(h5_file1, state)

    for e,l in zip(e_slices,e_labels):
        ind = (data_arr[0,:]>(e-10)) & (data_arr[0,:]<(e+10))
        data_arr_sliced = data_arr[:,ind]
        counts,_ = np.histogram(data_arr_sliced[2,:],t_bin_edges)
        end_avg = np.mean(counts[int(len(counts)/2):])
        int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
        counts = counts/int_time
        ax[i].plot(midpoints(t_bin_edges),counts,label=f'{sl} {l}')

    ax[i].set_ylabel(f'[counts /s /{binsize} s bin]')
    ax[i].legend()
    ax[i].minorticks_on()
plt.xlabel('Time since trigger [s]')


# Energy dependence
states = ['S','T','U','X']
state_labels = ['2.25kV','2.05kV','1.8kV','1.4kV']

fig,ax = plt.subplots(len(states),1)
fig.suptitle('Nd Energy Dependance')
e_slices = [842,1203,1241]
e_labels = ['E2M3','Ni-like','Co-like']
binsize = .003
t_bin_edges = np.arange(0,1.003,binsize)

for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_states_from_h5(h5_file1, state)

    for e,l in zip(e_slices,e_labels):
        ind = (data_arr[0,:]>(e-10)) & (data_arr[0,:]<(e+10))
        data_arr_sliced = data_arr[:,ind]
        counts,_ = np.histogram(data_arr_sliced[2,:],t_bin_edges)
        end_avg = np.mean(counts[int(len(counts)/2):])

        int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
        counts = counts/int_time
        ax[i].plot(midpoints(t_bin_edges),counts,label=f'{sl} {l}')

    ax[i].set_ylabel(f'[counts /s /{binsize} s bin]')
    ax[i].legend()
    ax[i].minorticks_on()
plt.xlabel('Time since trigger [s]')


# Current density dep spectra
states = ['B','C','D','Z','Y']
state_labels = ['10mA','15mA','20mA','30mA','40mA']

fig,ax = plt.subplots(len(states),1)
fig.suptitle('Nd Density Dependent Spectra')

binsize = 1
e_bin_edges = np.arange(750,2000,binsize)
ratio = [1203,1241]

for i,state,sl in zip(range(len(states)),states,state_labels):
    if i < 3: # not proud of this
        data_arr = load_states_from_h5(h5_file2, state)
    else:
        data_arr = load_states_from_h5(h5_file1, state)

    counts,_ = np.histogram(data_arr[0,:],e_bin_edges)
    centers = midpoints(e_bin_edges)
    int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
    counts = counts/int_time

    ni_c = np.sum(counts[(centers>(ratio[0]-5)) & (centers<(ratio[0]+5))])
    co_c = np.sum(counts[(centers>(ratio[1]-5)) & (centers<(ratio[1]+5))])
    ax[i].plot(centers,counts,label=f'{sl} Ni/Co={ni_c/co_c:.2f}')
    ax[i].set_ylabel(f'[counts /s /{binsize} eV bin]')
    ax[i].legend()
    ax[i].minorticks_on()
plt.xlabel('Energy [eV]')


# Energy dep spectra
states = ['S','T','U','X']
state_labels = ['2.25kV','2.05kV','1.8kV','1.4kV']

fig,ax = plt.subplots(len(states),1)
fig.suptitle('Nd Energy Dependent Spectra')

binsize = 1
e_bin_edges = np.arange(750,2000,binsize)
ratio = [1203,1241]

for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_states_from_h5(h5_file1, state)

    counts,_ = np.histogram(data_arr[0,:],e_bin_edges)
    centers = midpoints(e_bin_edges)
    int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
    counts = counts/int_time

    ni_c = np.sum(counts[(centers>(ratio[0]-5)) & (centers<(ratio[0]+5))])
    co_c = np.sum(counts[(centers>(ratio[1]-5)) & (centers<(ratio[1]+5))])
    ax[i].plot(centers,counts,label=f'{sl} Ni/Co={ni_c/co_c:.2f}')
    ax[i].set_ylabel(f'[counts /s /{binsize} eV bin]')
    ax[i].legend()
    ax[i].minorticks_on()
plt.xlabel('Energy [eV]')

plt.show()