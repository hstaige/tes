import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.rcParams.update({'font.size': 16})


data_dir = '/home/tim/research/tes/cal_data/20221221_0002_Aphotonlist.csv'
#data_dir = '/home/tim/research/tes/cal_data/20230803_0007_Aphotonlist.csv'

plot_type = 3
# 0: energy
# 1: fft
# 2: scatter all
# 3: td slice

if plot_type==0:
    binsize_e = 1
    e_bounds = [0,10000]

    data = np.loadtxt(data_dir,delimiter=',',skiprows=1)
    print(data.shape)
    data[:,1] -= data[0,1]
    print(data[-1,1]*1e-9/3600)
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]

    bin_edges = np.arange(e_bounds[0], e_bounds[1]+binsize_e, binsize_e)
    counts, _ = np.histogram(data[:,0], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize_e/2

    plt.plot(bin_centers,counts)
    plt.show()

elif plot_type==1:

    binsize_t = 1e-4
    e_bounds = [2617,2626]
    e_bounds = [2610,2640]
    #e_bounds = [1187,1223]
    data = np.loadtxt(data_dir,delimiter=',',skiprows=1)
    data[:,1] -= data[0,1]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]

    t_bounds = [np.min(data[:,2]),np.max(data[:,2])]
    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize_t, binsize_t)
    counts, _ = np.histogram(data[:,2], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize_t/2

    fig, ax = plt.subplots(2,1)
    ax[0].set_title('20231221_0002_T')
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

elif plot_type==2:

    data = np.loadtxt(data_dir,delimiter=',',skiprows=1)
    plt.scatter(data[:,2],data[:,0])
    plt.show()

elif plot_type==3:
    binsize_t = 2e-3
    e_bounds = [1187,1223]

    data = np.loadtxt(data_dir,delimiter=',',skiprows=1)
    print(data[-1,1]*1e-9/3600)
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]

    t_bounds = [np.min(data[:,2]),np.max(data[:,2])]
    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize_t, binsize_t)
    counts, _ = np.histogram(data[:,2], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize_t/2

    plt.plot(bin_centers,counts)
    plt.show()