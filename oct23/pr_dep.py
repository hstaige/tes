import numpy as np 
import matplotlib.pyplot as plt 
import h5py
import tools

plt.rcParams.update({'font.size': 16})

h5_file1 = '/home/tim/research/oct23_data/20231016_0001.h5'
h5_file2 = '/home/tim/research/oct23_data/20231017_0000.h5'

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

# t_bin_edges = np.arange(0,1.003,0.003)
# e_bin_edges = np.arange(750,2000,1)
# state1 = ['S','T','U','X']
# state2 = ['B','C']
# data_arr = np.hstack((load_state_from_h5(h5_file1, state1),load_state_from_h5(h5_file2,state2)))
# counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
# #plt.plot(e_bin_edges[:-1],np.sum(counts,axis=1))
# plt.pcolormesh(t_bin_edges,e_bin_edges,counts)


# Current density dependance
states = ['K','O','N','M']
state_labels = ['10mA','20mA','30mA','40mA']

#fig,ax = plt.subplots(len(states),1)
fig, ax = plt.subplots(len(states),1,figsize=(8.09*2,5*2))
#fig.suptitle('Pr Density Dependance')
e_slices = [798,1142,1182]
e_labels = ['E2/M3','Ni-like','Co-like']
binsize = .003
t_bin_edges = np.arange(0,1.003,binsize)

for i,state,sl in zip(range(len(states)),states,state_labels):

    data_arr = load_state_from_h5(h5_file1, state)

    colors = iter([tools.blue,tools.red,tools.green])
    for e,l in zip(e_slices,e_labels):
        ind = (data_arr[0,:]>(e-10)) & (data_arr[0,:]<(e+10))
        data_arr_sliced = data_arr[:,ind]
        counts,_ = np.histogram(data_arr_sliced[2,:],t_bin_edges)
        end_avg = np.mean(counts[int(len(counts)/2):])
        int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
        counts = counts/int_time
        ax[i].plot(midpoints(t_bin_edges),counts,label=f'{sl} {l}', color=next(colors))

    #ax[i].set_ylabel(f'[counts /s /{binsize} s bin]')
    ax[i].legend(loc='upper right')
    ax[i].minorticks_on()

fig.supylabel(f'counts / s / {binsize*1e3:.0f} ms bin')
plt.xlabel('Time since trigger [s]')
plt.tight_layout()
plt.minorticks_on()


# Energy dependence
states = ['F','G','H','J']
state_labels = ['2.15kV','1.95kV','1.7kV','1.4kV']

fig, ax = plt.subplots(len(states),1,figsize=(8.09*2,5*2))
#fig.suptitle('Pr Energy Dependance')
e_slices = [798,1142,1182]
e_labels = ['E2/M3','Ni-like','Co-like']
binsize = .003
t_bin_edges = np.arange(0,1.003,binsize)

for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_state_from_h5(h5_file1, state)

    colors = iter([tools.blue,tools.red,tools.green])
    for e,l in zip(e_slices,e_labels):
        ind = (data_arr[0,:]>(e-10)) & (data_arr[0,:]<(e+10))
        data_arr_sliced = data_arr[:,ind]
        counts,_ = np.histogram(data_arr_sliced[2,:],t_bin_edges)
        end_avg = np.mean(counts[int(len(counts)/2):])

        int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
        counts = counts/int_time
        ax[i].plot(midpoints(t_bin_edges),counts,label=f'{sl} {l}', color=next(colors))

    ax[i].legend(loc='upper right')
    ax[i].minorticks_on()

plt.xlabel('Time since trigger [s]')
fig.supylabel(f'counts / s / {binsize*1e3:.0f} ms bin')
plt.tight_layout()
plt.minorticks_on()


# Current density dep spectra
states = ['K','O','N','M']
state_labels = ['10mA','20mA','30mA','40mA']

fig, ax1 = plt.subplots(figsize=(8.09*2,5*2))
#fig.suptitle('Pr Density Dependent Spectra')

binsize = 1
e_bin_edges = np.arange(750,1250,binsize)
ratio = [1142,1182]

colors = iter([tools.blue, tools.green, tools.yellow, tools.red,'k'])
for i,state,sl in zip(range(len(states)),states,state_labels):

    data_arr = load_state_from_h5(h5_file1, state)

    counts,_ = np.histogram(data_arr[0,:],e_bin_edges)
    centers = midpoints(e_bin_edges)
    int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
    #int_time = np.max(counts)
    counts = counts/int_time

    ni_c = np.sum(counts[(centers>(ratio[0]-5)) & (centers<(ratio[0]+5))])
    co_c = np.sum(counts[(centers>(ratio[1]-5)) & (centers<(ratio[1]+5))])
    c = next(colors)
    ax1.plot(centers,counts,label=f'{sl} Ni/Co={ni_c/co_c:.2f}',color=c)

plt.annotate('Ni-like',(1144,7),ha='center')
plt.annotate('Co-like',(1182,2.7),ha='center')
plt.annotate('E2/M3',(798,1.7),ha='center')
ax1.set_ylabel(f'[counts / s / {binsize} eV bin]')
# ax1.set_ylabel('Normalized counts')
ax1.legend(loc='upper left',title='Pr')
ax1.minorticks_on()
ax1.set_xlabel('Energy [eV]')
plt.tight_layout()

# Energy dep spectra
states = ['F','G','H','J']
state_labels = ['2.15kV','1.95kV','1.7kV','1.4kV']

fig, ax1 = plt.subplots(figsize=(8.09*2,5*2))
#fig.suptitle('Pr Energy Dependent Spectra')

binsize = 1
e_bin_edges = np.arange(750,1250,binsize)
ratio = [1142,1182]

colors = iter([tools.blue, tools.green, tools.yellow, tools.red])
for i,state,sl in zip(range(len(states)),states,state_labels):
    data_arr = load_state_from_h5(h5_file1, state)

    counts,_ = np.histogram(data_arr[0,:],e_bin_edges)
    centers = midpoints(e_bin_edges)
    int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9
    counts = counts/int_time

    ni_c = np.sum(counts[(centers>(ratio[0]-5)) & (centers<(ratio[0]+5))])
    co_c = np.sum(counts[(centers>(ratio[1]-5)) & (centers<(ratio[1]+5))])
    c = next(colors)
    ax1.plot(centers,counts,label=f'{sl} Ni/Co={ni_c/co_c:.2f}',color=c)

plt.annotate('Ni-like',(1144,5.6),ha='center')
plt.annotate('Co-like',(1182,6.2),ha='center')
plt.annotate('E2/M3',(798,1.3),ha='center')
ax1.set_ylabel(f'[counts / s / {binsize} eV bin]')
ax1.legend(loc='upper left',title='Pr')
ax1.minorticks_on()
ax1.set_xlabel('Energy [eV]')
plt.tight_layout()

plt.show()