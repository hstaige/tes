import numpy as np
import matplotlib.pyplot as plt
import tools

plt.rcParams.update({'font.size': 16})

plot_num = 0

# 0: Nd time dependance ni/co
# 1: Nd example spectrum

if plot_num == 0:
    datafile1 = '/home/tim/research/EBIT-TES-Data/data_by_state/20231014_0006_D.npy'
    datafile2 = '/home/tim/research/EBIT-TES-Data/data_by_state/20231014_0006_I.npy'
    theoryfile1 = '/home/tim/research/dec22_data/theory/td_th/Nd/CX3e12_3e12_iontotpp.dat'
    theoryfile2 = '/home/tim/research/dec22_data/theory/td_th/Nd/M3_HF_A/CX3e12/2e12/iontotpp.dat'
    e_slices = [842,1203,1241]
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


        ax.errorbar(tools.midpoints(times),data_ni,ni_uncert,linestyle='',marker='.', c=tools.red)
        ax.errorbar(tools.midpoints(times),data_co,co_uncert,linestyle='',marker='.', c=tools.blue)
        ax.set_xscale('log')
        return (np.max(data_ni), np.max(data_co))

    def plot_theory(theoryfile, ax, norm):
        th = np.loadtxt(theoryfile,skiprows=1)
        th = tools.bound(th,th[:,0],0.01,1)
        th_t = th[:,0]
        th_co = th[:,7]*norms[1]/np.max(th[:,7])
        th_ni = th[:,6]*norms[0]/np.max(th[:,6])

        ax.plot(th_t,th_ni, c=tools.red)
        ax.plot(th_t,th_co, c=tools.blue)

    fig,ax = plt.subplots(2,1)
    norms = plot_data(datafile1,ax[0])
    plot_theory(theoryfile1,ax[0],norms)
    plot_data(datafile2,ax[1])
    plt.show()

if plot_num == 1:
    datafile1 = '/home/tim/research/EBIT-TES-Data/data_by_state/20231014_0006_D.npy'
    datafile2 = '/home/tim/research/EBIT-TES-Data/data_by_state/20231014_0006_I.npy'
    energy_bin = .5

    def plot_data(datafile, ax, color, label):
        data = np.load(datafile).T
        print(data)
        energies = np.arange(750,1630,energy_bin)
        bin_data,_ = np.histogram(data[:,0],bins=energies)

        int_time = (np.max(data[:,1])-np.min(data[:,1]))*1e-9
        print(int_time)
        data_uncert = bin_data**.5

        ax.plot(tools.midpoints(energies),bin_data/int_time,marker='',c=color,label=label)

    def plot_inset(datafile, ax, color, label):
        data = np.load(datafile).T
        print(data)
        energies = np.arange(843-20,843+20,energy_bin)
        bin_data,_ = np.histogram(data[:,0],bins=energies)

        int_time = (np.max(data[:,1])-np.min(data[:,1]))*1e-9
        print(int_time)
        data_uncert = bin_data**.5

        ax.plot(tools.midpoints(energies),bin_data/int_time,marker='',c=color,label=label)

    fig, ax = plt.subplots(figsize=(8.09,5))
    plot_data(datafile1,ax,tools.red,'60 mA')
    plot_data(datafile2,ax,tools.blue, '30 mA')
    plt.xlabel('Energy [eV]')
    plt.ylabel(f'Count rate [s$^{{{-1}}}$]')
    plt.legend()
    plt.tight_layout()
    plt.minorticks_on()

    left, bottom, width, height = [0.15, 0.65, 0.25, 0.25]
    ax2 = fig.add_axes([left, bottom, width, height])
    plot_inset(datafile1,ax2,tools.red,'60 mA')
    plot_inset(datafile2,ax2,tools.blue, '30 mA')
    plt.minorticks_on()
    plt.show()