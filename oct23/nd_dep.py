import os
import re
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.signal import savgol_filter
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

# Current density dependance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
states = ['B','AA','Z','Y']
state_labels = ['10mA','20mA','30mA','40mA']

#fig,ax = plt.subplots(len(states),1)
fig, ax = plt.subplots(len(states),1,figsize=(8.09*2,5*2))
#fig.suptitle('Nd Density Dependance')
e_slices = [842,1203,1241]
e_labels = ['E2/M3','Ni-like','Co-like']
binsize = .003
t_bin_edges = np.arange(0,1.003,binsize)

for i,state,sl in zip(range(len(states)),states,state_labels):

    if i < 1: # not proud of this
        data_arr = load_state_from_h5(h5_file2, state)
    else:
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


# Energy dependence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
states = ['S','T','U','X']
state_labels = ['2.25kV','2.05kV','1.8kV','1.4kV']

fig, ax = plt.subplots(len(states),1,figsize=(8.09*2,5*2))
#fig.suptitle('Nd Energy Dependance')
e_slices = [842,1203,1241]
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


# Current density dep spectra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
states = ['B','AA','Z','Y']
state_labels = ['10mA','20mA','30mA','40mA']

fig, ax1 = plt.subplots(figsize=(8.09*2,5*2))
#fig.suptitle('Nd Density Dependent Spectra')

binsize = 1
e_bin_edges = np.arange(800,1350,binsize)
ratio = [1203,1241]

colors = iter([tools.blue, tools.green, tools.yellow, tools.red,'k'])
for i,state,sl in zip(range(len(states)),states,state_labels):
    if i < 1: # not proud of this
        data_arr = load_state_from_h5(h5_file2, state)
    else:
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
plt.annotate('Ni-like',(1203,4.4),ha='center')
plt.annotate('Co-like',(1241,3),ha='center')
plt.annotate('E2/M3',(842,1.2),ha='center')
ax1.set_ylabel(f'[counts / s / {binsize} eV bin]')
# ax1.set_ylabel('Normalized counts')
ax1.legend(loc='upper left',title="Nd")
ax1.minorticks_on()
ax1.set_xlabel('Energy [eV]')
plt.tight_layout()


# Energy dep spectra ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
states = ['S','T','U','X']
state_labels = ['2.25kV','2.05kV','1.8kV','1.4kV']

fig, ax1 = plt.subplots(figsize=(8.09*2,5*2))

#fig.suptitle('Nd Energy Dependent Spectra')

binsize = 1
e_bin_edges = np.arange(800,1350,binsize)
ratio = [1203,1241]

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
plt.annotate('Ni-like',(1203,4.1),ha='center')
plt.annotate('Co-like',(1241,5.2),ha='center')
plt.annotate('E2/M3',(842,1.2),ha='center')
ax1.set_ylabel(f'[counts / s / {binsize} eV bin]')
ax1.legend(loc='upper left',title='Nd')
ax1.minorticks_on()
ax1.set_xlabel('Energy [eV]')
plt.tight_layout()
plt.show()


# Th ratios ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
nd_th_dir = '/home/tim/research/dec22_data/theory/td_th/Nd/'

th_files = os.listdir(nd_th_dir)
r = re.compile(r'noCX.*[1-9]e[1-9][1-2]')
th_files = np.array(list(filter(r.match,th_files)))
th_dens = [float(re.search(f'[1-9]e[0-9]+',f)[0]) for f in th_files]
order = np.argsort(th_dens)
th_files = th_files[order]

fig, ax1 = plt.subplots(figsize=(8.09*2,5*2))
color = iter(cm.rainbow(np.linspace(0, 1, len(th_files))))
for f in th_files:
    spec = np.loadtxt(nd_th_dir+f)
    plt.plot(spec[:,0],spec[:,6]/spec[:,7],color=next(color),label='NOMAD '+re.search(f'[1-9]e[0-9]+',f)[0]+r' cm$^{-3}$')


# Current density dependance
# states = ['B','AA','Z','Y']
# state_labels = ['10 mA','20 mA','30 mA','40 mA']
states = ['AA','Z','Y']
state_labels = ['20 mA','30 mA','40 mA']

#fig,ax = plt.subplots(len(states),1)
#fig.suptitle('Nd Density Dependance')
e_slices = [1203,1241]
e_labels = ['Ni-like','Co-like']
binsize = .008
t_bin_edges = np.arange(0.035,1,binsize)

#colors = iter([tools.blue, tools.green, tools.yellow, tools.red])
colors = iter([tools.blue, tools.green, tools.red])
for i,state,sl in zip(range(len(states)),states,state_labels):

    if i < 0: # not proud of this
        data_arr = load_state_from_h5(h5_file2, state)
    else:
        data_arr = load_state_from_h5(h5_file1, state)

    #t_bin_edges -= 0.002

    niind = (data_arr[0,:]>(e_slices[0]-10)) & (data_arr[0,:]<(e_slices[0]+10))
    data_arr_sliced = data_arr[:,niind]
    nicounts,_ = np.histogram(data_arr_sliced[2,:],t_bin_edges)
    nicounts = savgol_filter(nicounts, 10, 3)
    coind = (data_arr[0,:]>(e_slices[1]-10)) & (data_arr[0,:]<(e_slices[1]+10))
    data_arr_sliced = data_arr[:,coind]
    cocounts,_ = np.histogram(data_arr_sliced[2,:],t_bin_edges)
    err = (nicounts/cocounts)**.5
    #cocounts = savgol_filter(cocounts, 50, 2)
    c = next(colors)
    ax1.plot(midpoints(t_bin_edges),nicounts/cocounts*.4,label=f'Experiment {sl}', color=c,marker='o',linestyle=':')
    ax1.errorbar(midpoints(t_bin_edges),(nicounts/cocounts),yerr=err, color=c, linestyle='None',alpha=0)

plt.ylim([0,10])
plt.xlim([0.04,1])
plt.xscale('log')
plt.ylabel('Ni/Co-like line ratio')
plt.xlabel('Time since trigger [s]')
plt.minorticks_on()
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()