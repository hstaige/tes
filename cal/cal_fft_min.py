import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.rcParams.update({'font.size': 16})

states = ['A','B']
data_dir = '/home/tim/research/tes/cal_data'
day = '20230728'
run = '0005'

for state in states:
    
    file = f'{data_dir}/{day}_{run}_{state}photonlist.csv'
    binsize_t = 1e-4
    e_bounds = [2617,2626]
    

    data = np.loadtxt(file,delimiter=',',skiprows=1)
    data[:,1] -= data[0,1]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]

    t_bounds = [np.min(data[:,2]),np.max(data[:,2])]
    print(t_bounds)
    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize_t, binsize_t)
    counts, _ = np.histogram(data[:,2], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize_t/2

    fig, ax = plt.subplots(2,1)
    ax[0].set_title(f'State {state}'+ r' Cl K$\alpha$')
    ax[0].scatter(bin_centers,counts)
    ax[0].set_xlabel('Time since trigger [s]')
    ax[0].set_ylabel(f'Counts per {binsize_t} s bin')
    ax[0].minorticks_on()
    N = len(counts)
    T = binsize_t
    xf = fftfreq(N,T)[1:N//2]

    ax[1].plot(xf,2/N*np.abs(fft(counts)[1:N//2]))
    ax[1].set_xlabel('Freq [Hz]')
    ax[1].set_ylabel('Magnitude (arb)')
    ax[1].minorticks_on()
plt.show()