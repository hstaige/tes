import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.stats import binned_statistic_2d
from scipy.signal import lfilter, butter

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/tes/td_data/'
file = '20221220_'
states = ['0000_A','0000_B','0000_K']


plot_type = 3
# 0: summed states fft
# 1: 2d binned
# 2: seperate state fft
# 3: 

if plot_type == 0:

    data = np.empty((1,2))
    for state in states:
        data = np.vstack((data,np.load(dir+file+state+'.npy')))

    binsize = .001
    e_bounds = [2610,2630]
    t_bounds = [0,1]

    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
    counts, _ = np.histogram(data[:,1], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    N = len(counts)
    T = binsize
    xf = fftfreq(N,T)[1:N//2]
    plt.plot(xf,2/N*np.abs(fft(counts)[1:N//2]))
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Magnitude (arb)')
    plt.minorticks_on()
    plt.show()

elif plot_type == 1:

    data = np.empty((1,2))
    for state in states:
        data = np.vstack((data,np.load(dir+file+state+'.npy')))

    e_binsize = 1
    t_binsize = .001 

    e_bounds = [500,5000]
    t_bounds = [0,1.000]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    x_bins = np.arange(np.min(data[:,0]),np.max(data[:,0]+e_binsize),e_binsize)
    y_bins = np.arange(np.min(data[:,1]),np.max(data[:,1]+t_binsize),t_binsize)
    bin_data,x_edges,y_edges,_ = binned_statistic_2d(data[:,0], data[:,1], None, statistic='count', bins=[x_bins,y_bins])

    plt.pcolormesh(y_edges,x_edges,bin_data)
    plt.xlabel('Time since trigger [ms]')
    plt.ylabel('Photon energy [eV]')
    plt.minorticks_on()
    plt.show()

if plot_type == 2:
    for state in states:

        data = np.load(dir+file+state+'.npy')

        binsize = .001
        e_bounds = [2610,2630]
        t_bounds = [0,1]

        data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
        data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

        bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
        counts, _ = np.histogram(data[:,1], bins=bin_edges)
        bin_centers = bin_edges[:-1]+binsize/2

        N = len(counts)
        T = binsize
        xf = fftfreq(N,T)[1:N//2]
        plt.plot(xf,2/N*np.abs(fft(counts)[1:N//2]), label=state)

    plt.xlabel('Freq [Hz]')
    plt.ylabel('Magnitude (arb)')
    plt.minorticks_on()
    plt.legend()
    plt.show()

elif plot_type == 3:

    def butter_bandstop_filter(data, lowcut, highcut, fs, order):


        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        i, u = butter(order, [low, high], btype='bandstop')
        y = lfilter(i, u, data)
        return y

    data = np.empty((1,2))
    for state in states:
        data = np.vstack((data,np.load(dir+file+state+'.npy')))

    e_binsize = 1
    t_binsize = .001 

    e_bounds = [2610,2630]
    t_bounds = [0,1.000]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    x_bins = np.arange(np.min(data[:,0]),np.max(data[:,0]+e_binsize),e_binsize)
    y_bins = np.arange(np.min(data[:,1]),np.max(data[:,1]+t_binsize),t_binsize)
    bin_data,x_edges,y_edges,_ = binned_statistic_2d(data[:,0], data[:,1], None, statistic='count', bins=[x_bins,y_bins])
    oddata = np.sum(bin_data,axis=0)
    plt.plot(y_bins[:-1],oddata)
    filtered = butter_bandstop_filter(oddata,19,21,len(oddata),3)
    plt.plot(y_bins[:-1],filtered,color='k')
    plt.xlabel('Time since trigger [ms]')
    plt.ylabel('Counts')
    plt.minorticks_on()
    plt.show()
