import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.rcParams.update({'font.size': 16})

data_dir1 = '/home/tim/research/tes/cal_data/20230728_0005_Aphotonlist.csv'
data_dir2 = '/home/tim/research/tes/cal_data/20230803_0007_Bphotonlist.csv'

eff_curve_data1 = '/home/tim/research/tes/cal_data/n2_trans.txt'
eff_curve_data2 = '/home/tim/research/tes/cal_data/h2o_trans.txt'

plot_type = 1
# 1: energies
# 2: differences at peaks

if plot_type == 0:

    binsize_e = .5
    e_bounds = [0,10000]

    data1 = np.loadtxt(data_dir1,delimiter=',',skiprows=1)

    data1[:,1] -= data1[0,1]
    print(data1[-1,1]*1e-9/3600)
    data1 = data1[(data1[:,0]>e_bounds[0]) & (data1[:,0]<e_bounds[1])]

    bin_edges1 = np.arange(e_bounds[0], e_bounds[1]+binsize_e, binsize_e)
    counts1, _ = np.histogram(data1[:,0], bins=bin_edges1)
    bin_centers1 = bin_edges1[:-1]+binsize_e/2

    data2 = np.loadtxt(data_dir2,delimiter=',',skiprows=1)

    data2[:,1] -= data2[0,1]
    print(data2[-1,1]*1e-9/3600)
    data2 = data2[(data2[:,0]>e_bounds[0]) & (data2[:,0]<e_bounds[1])]

    bin_edges2 = np.arange(e_bounds[0], e_bounds[1]+binsize_e, binsize_e)
    counts2, _ = np.histogram(data2[:,0], bins=bin_edges2)
    bin_centers2 = bin_edges2[:-1]+binsize_e/2

    counts2 = counts2*(np.max(counts1)/np.max(counts2))
    plt.plot(bin_centers1,counts1)
    plt.plot(bin_centers2,counts2,linestyle=':')
    #plt.plot(bin_centers1,counts1/counts2)
    plt.show()

elif plot_type == 1:

    binsize_e = 1
    e_bounds = [0,10000]

    data1 = np.loadtxt(data_dir1,delimiter=',',skiprows=1)

    data1[:,1] -= data1[0,1]
    print(data1[-1,1]*1e-9/3600)
    data1 = data1[(data1[:,0]>e_bounds[0]) & (data1[:,0]<e_bounds[1])]

    bin_edges1 = np.arange(e_bounds[0], e_bounds[1]+binsize_e, binsize_e)
    counts1, _ = np.histogram(data1[:,0], bins=bin_edges1)
    bin_centers1 = bin_edges1[:-1]+binsize_e/2

    peaks, _ = find_peaks(counts1, prominence=500)

    data2 = np.loadtxt(data_dir2,delimiter=',',skiprows=1)

    data2[:,1] -= data2[0,1]
    print(data2[-1,1]*1e-9/3600)
    data2 = data2[(data2[:,0]>e_bounds[0]) & (data2[:,0]<e_bounds[1])]

    bin_edges2 = np.arange(e_bounds[0], e_bounds[1]+binsize_e, binsize_e)
    counts2, _ = np.histogram(data2[:,0], bins=bin_edges2)
    bin_centers2 = bin_edges2[:-1]+binsize_e/2
    fig,ax = plt.subplots(2,1)

    counts2 = counts2*(1/np.max(counts2))
    counts1 = counts1*(1/np.max(counts1))
    ax[0].plot(bin_centers1,counts1, label='2023_07_28')
    # for i in peaks:
    #     ax[0].axvline(bin_centers1[i],color='k',alpha=.5)
    ax[0].plot(bin_centers2,counts2,linestyle=':',label='2023_08_03')
    ax[0].set_xlim([0,8200])
    ax[0].set_ylabel('Normalized Counts')
    ax[0].minorticks_on()
    ax[0].legend()

    eff_curve1 = np.loadtxt(eff_curve_data1, skiprows=2)
    eff_curve2 = np.loadtxt(eff_curve_data2, skiprows=2)

    diff = counts2[peaks]/counts1[peaks]
    ax[1].scatter(bin_centers1[peaks],diff)
    ax2 = ax[1].twinx()
    ax2.plot(eff_curve1[:,0], eff_curve1[:,1],linestyle=':',label='1 um N2 transmission',color='k')
    ax2.plot(eff_curve2[:,0], eff_curve2[:,1],linestyle=':',label='1 um H2O transmission',color='r')
    ax2.legend(loc='lower right')
    ax2.set_ylabel('Transmission')
    ax[1].set_xlim([0,8200])
    ax[1].set_ylabel('Peak Amp Ratio')
    ax[1].set_xlabel('Energy [eV]')
    
    ax[1].minorticks_on()
    plt.show()