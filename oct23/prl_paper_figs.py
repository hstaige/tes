import numpy as np
import matplotlib.pyplot as plt
import tools

plt.rcParams.update({'font.size': 16})

plot_num = 0

# 0: Pr example spectrum

if plot_num == 0:
    datafile = '/home/tim/research/EBIT-TES-Data/data_by_state/20231017_0001_D.npy'
    e_slices = [798,1142,1182]
    e_labels = ['E2M3','Ni-like','Co-like']
    hw = 5
    time_bin = 0.003

    def plot_data(datafile, ax):
        data = np.load(datafile).T

        data_ni = tools.bound(data, data[:,0], e_slices[1]-hw, e_slices[1]+hw)
        data_co = tools.bound(data, data[:,0], e_slices[2]-hw, e_slices[2]+hw)

        times = np.arange(0.01,1+time_bin,time_bin)
        data_ni,_ = np.histogram(data_ni[:,2],bins=times)
        data_co,_ = np.histogram(data_co[:,2],bins=times)

        ni_uncert = data_ni**.5
        co_uncert = data_co**.5

        ax.errorbar(tools.midpoints(times),data_ni,ni_uncert,linestyle='',marker='.', c=tools.red, label=e_labels[1])
        ax.errorbar(tools.midpoints(times),data_co,co_uncert,linestyle='',marker='.', c=tools.blue, label=e_labels[2])
        #ax.set_xscale('log')
        return (np.max(data_ni), np.max(data_co))

    fig, ax = plt.subplots(figsize=(8.09,5))
    norms = plot_data(datafile,ax)
    plt.ylabel('Counts')
    plt.xlabel('Time [S]')
    fig.tight_layout()
    ax.minorticks_on()
    plt.legend()
    plt.show()
