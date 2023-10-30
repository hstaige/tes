import numpy as np 
import matplotlib.pyplot as plt 
import h5py
# plt.ion()
# plt.close('all')

h5_file = '/home/tim/research/oct23_data/massgui_export_20231016_0001.h5'

def load_state_from_h5(h5_file, state):
    h5 = h5py.File(h5_file, 'r')
    n = 0
    for key in h5.keys():
        n += len(h5[key][state]['energy'])

    energies = np.zeros(n)
    unixnanos = np.zeros(n)
    seconds_after_last_external_trigger = np.zeros(n)

    i0 = 0
    for key in h5.keys():
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
state = ['K','M','N','O','Q','F','G','H','J']
data_arr = load_state_from_h5(h5_file, state)
counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
plt.pcolormesh(t_bin_edges,e_bin_edges,counts)


# Current density dependance
states = ['K','M','N','O','Q']
state_labels = ['10mA','15mA','20mA','30mA','40mA']

fig,ax = plt.subplots(len(states),1)
fig.suptitle('Pr Density Dependance')
e_slices = [798,1142,1182]
e_labels = ['E2M3','Ni-like','Co-like']
binsize = .003
t_bin_edges = np.arange(0,1.003,binsize)

for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_state_from_h5(h5_file, state)

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
states = ['F','G','H','J']
state_labels = ['2.15kV','1.95kV','1.7kV','1.4kV']

fig,ax = plt.subplots(len(states),1)
fig.suptitle('Pr Energy Dependance')
e_slices = [798,1142,1182]
e_labels = ['E2M3','Ni-like','Co-like']
binsize = .003
t_bin_edges = np.arange(0,1.003,binsize)

for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_state_from_h5(h5_file, state)

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


# # Current density ratio
# states = ['K','M','N','O','Q']
# state_labels = ['10mA','15mA','20mA','30mA','40mA']

# fig,ax = plt.subplots(len(states),1)
# fig.suptitle('Pr Density Dependance')
# e_slices = [1142,1182]
# e_labels = ['Ni-like','Co-like']
# t_bin_edges = np.arange(.5,.9,0.003)

# for i,state,sl in zip(range(len(states)),states,state_labels):
#     data_arr = load_states_from_h5(h5_file, state)

#     indni = (data_arr[0,:]>(e_slices[0]-10)) & (data_arr[0,:]<(e_slices[0]+10))
#     indco = (data_arr[0,:]>(e_slices[1]-10)) & (data_arr[0,:]<(e_slices[1]+10))
#     data_arr_slicedni = data_arr[:,indni]
#     data_arr_slicedco = data_arr[:,indco]
#     countsni,_ = np.histogram(data_arr_slicedni[2,:],t_bin_edges)
#     countsco,_ = np.histogram(data_arr_slicedco[2,:],t_bin_edges)

#     ax[i].plot(midpoints(t_bin_edges),countsni/countsco,label=f'{sl} Ni/Co ratio')
#     ax[i].set_ylabel('Counts')
#     ax[i].legend()
#     ax[i].minorticks_on()
# plt.xlabel('Time since trigger [s]')

# # Energy ratio
# states = ['F','G','H','J']
# state_labels = ['2.15kV','1.95kV','1.7kV','1.4kV']

# fig,ax = plt.subplots(len(states),1)
# fig.suptitle('Pr Energy Dependance')
# e_slices = [1142,1182]
# e_labels = ['Ni-like','Co-like']
# t_bin_edges = np.arange(0.5,.95,0.003)

# for i,state,sl in zip(range(len(states)),states,state_labels):
#     data_arr = load_states_from_h5(h5_file, state)

#     indni = (data_arr[0,:]>(e_slices[0]-10)) & (data_arr[0,:]<(e_slices[0]+10))
#     indco = (data_arr[0,:]>(e_slices[1]-10)) & (data_arr[0,:]<(e_slices[1]+10))
#     data_arr_slicedni = data_arr[:,indni]
#     data_arr_slicedco = data_arr[:,indco]
#     countsni,_ = np.histogram(data_arr_slicedni[2,:],t_bin_edges)
#     countsco,_ = np.histogram(data_arr_slicedco[2,:],t_bin_edges)

#     ax[i].plot(midpoints(t_bin_edges),countsni/countsco,label=f'{sl} Ni/Co ratio')
#     ax[i].set_ylabel('Counts')
#     ax[i].legend()
#     ax[i].minorticks_on()
# plt.xlabel('Time since trigger [s]')


# Current density dep spectra
states = ['K','Q','O','N','M']
state_labels = ['10mA','15mA','20mA','30mA','40mA']

fig,ax = plt.subplots(len(states),1)
fig.suptitle('Pr Density Dependent Spectra')

binsize = 1
e_bin_edges = np.arange(750,2000,binsize)
ratio = [1142,1182]

for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_state_from_h5(h5_file, state)

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
states = ['F','G','H','J']
state_labels = ['2.15kV','1.95kV','1.7kV','1.4kV']

fig,ax = plt.subplots(len(states),1)
fig.suptitle('Pr Energy Dependent Spectra')

binsize = 1
e_bin_edges = np.arange(750,2000,binsize)
ratio = [1142,1182]

for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_state_from_h5(h5_file, state)

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