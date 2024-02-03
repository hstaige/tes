import numpy as np
import matplotlib.pyplot as plt
import my_utils as utils

plot_num = 0

if plot_num == 0:
    datafile1 = '/home/tim/research/EBIT-TES-Data/data_by_state/20231014_0006_D.npy'
    datafile2 = '/home/tim/research/EBIT-TES-Data/data_by_state/20231014_0006_I.npy'
    e_slices = [842,1203,1241]
    e_labels = ['E2M3','Ni-like','Co-like']
    hw = 5

    def plot_data(datafile, ax):
        data = np.load(datafile1).T
        data_ni = data[(data[:,0]>(e_slices[1]-hw))&(data[:,0]<(e_slices[1]+hw))]
        data_co = data[(data[:,0]>(e_slices[2]-hw))&(data[:,0]<(e_slices[2]+hw))]

        times = np.linspace(0.01,1.002,500)
        data_ni,_ = np.histogram(data_ni[:,2],bins=times)
        data_co,_ = np.histogram(data_co[:,2],bins=times)

        ax.scatter(utils.midpoints(times),data_ni)
        ax.scatter(utils.midpoints(times),data_co)
        ax.set_xscale('log')

    fig,ax = plt.subplots(2,1)
    plot_data(datafile1,ax[0])
    plot_data(datafile2,ax[1])
    plt.show()