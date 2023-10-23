import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, filtfilt, lfilter
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update({'font.size': 20})

dir = "/home/tim/research/tes/"
file = 'td_data/'
# states = ['0002_V','0002_X','0002_Z','0002_AB','0002_AD','0002_AF','0002_T']
# current_labels = [18.4, 9.2, 13.8, 23, 27.6, 32.2, 36.8]


fig, axs = plt.subplots(2)

for plot in range(2):
    bins = .0001
    t_bounds = [0,3]

    if plot==0:
        Z = 'Pr'
        states = ['20221221_0002_K']
        th_files = ['Pr/M3_HF_A/CX3e12/3e12/iontotpp.dat','Pr/CX3e12/3e12/iontotpp.dat']
        th_labels = ['HF 3e12 $e^-⋅cm^{-3}$', 'No HF 3e12 $e^-⋅cm^{-3}$']
        voltage_labels = [1.93]
        current_labels = [32]
        e_bounds_ni = [1140,1146]
        e_bounds_co = [1182,1188]
    elif plot==1:
        Z = 'Pr'
        states = ['20221221_0002_M']
        th_files = ['Pr/M3_HF_A/CX3e12/2e12/iontotpp.dat','Pr/CX3e12/2e12/iontotpp.dat']
        th_labels = ['HF 2e12 $e^-⋅cm^{-3}$', 'No HF 2e12 $e^-⋅cm^{-3}$']
        voltage_labels = [1.93]
        current_labels = [16]
        e_bounds_ni = [1140,1146]
        e_bounds_co = [1182,1188]

    bin_edges = np.logspace(np.log10(0.01),np.log10(3.0), 500)

    def data_slice_nans(dir, file, state, bin_edges, e_range, t_range):

        data = np.load(dir+file+state+'.npy')

        data = data[(data[:,1]>t_range[0]) & (data[:,1]<t_range[1])]
        data = data[(data[:,0]>e_range[0]) & (data[:,0]<e_range[1])]

        # constant bins first pass and filter
        bins = .0005
        bin_edges_const = np.arange(t_bounds[0], t_bounds[1]+bins, bins) # constant bins
        bin_cents_const = bin_edges_const[:-1]+bins/2
        counts, _ = np.histogram(data[:,1], bins=bin_edges_const)
        counts_unfiltered = counts

        f0 = 20
        Q = 3
        a, b = iirnotch(f0, Q, 1 / bins)
        counts = lfilter(a, b, counts)

        counts[counts < 0] = 0

        log_counts = []
        log_uncertainties = []
        for lb, rb in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (bin_cents_const >= lb) & (bin_cents_const < rb)
            if np.any(mask):
                avg_count = np.sum(1 / (counts[mask] + 1) * counts[mask]) / np.sum(1 / (counts[mask] + 1))
                #avg_count = np.sum((counts[mask] + 1) * counts[mask]) / np.sum(counts[mask] + 1)
                log_counts.append(avg_count)
                log_uncertainties.append(1 / np.sqrt(np.sum(1 / (counts[mask] + 1))) if avg_count else 0)
            else:
                log_counts.append(0)
                log_uncertainties.append(0)

        counts = np.array(log_counts)
        uncert = np.array(log_uncertainties)

        return counts, uncert

    bin_centers = np.array([(bin+binp1)/2 for bin, binp1 in zip(bin_edges[:-1], bin_edges[1:])])

    df_out = pd.DataFrame({'time_since_trigger': bin_centers})

    for state, I, V in zip(states, current_labels, voltage_labels):
        counts_ni, uncert_ni = data_slice_nans(dir, file, state, bin_edges, e_bounds_ni, t_bounds)
        df_out[f'{Z}_ni_{I:.1f}mA_{V}kV'] = counts_ni
        df_out[f'{Z}_ni_{I:.1f}mA_{V}kV_uncert'] = uncert_ni

        counts_co, uncert_co = data_slice_nans(dir, file, state, bin_edges, e_bounds_co, t_bounds)
        df_out[f'{Z}_co_{I:.1f}mA_{V}kV'] = counts_co
        df_out[f'{Z}_co_{I:.1f}mA_{V}kV_uncert'] = uncert_co

        # counts_e2m3, uncert_e2m3 = data_slice_nans(dir, file, state, bin_edges, e_bounds_e2m3, t_bounds)
        # df_out[f'e2m3_{I:.1f}mA'] = counts_e2m3
        # df_out[f'e2m3_{I:.1f}mA_uncert'] = uncert_e2m3

    df_out.to_csv(dir+f'td_data/filt_slices_{Z}{voltage_labels[0]}kV{current_labels[0]}mA.csv')

    axs[plot].errorbar(bin_centers,counts_co,uncert_co,ls='none',color='#0077bb', alpha = 0.75)
    axs[plot].scatter(bin_centers,counts_co,label=f'Co-like',color='#0077bb', s=16)

    axs[plot].errorbar(bin_centers,counts_ni,uncert_ni,ls='none',color='#cc3311', alpha = 0.75)
    axs[plot].scatter(bin_centers,counts_ni,label=f'Ni-like',color='#cc3311', s=16)

    #th_files = os.listdir(dir+'theory/td_th/Nd/')
    time_avg_cut = 1
    avg0 = np.mean(counts_ni[bin_centers>time_avg_cut])
    avg1 = np.mean(counts_co[bin_centers>time_avg_cut])
    for i in range(len(th_files)):
        qstate_ni = 6
        qstate_co = 7
        th = np.loadtxt(f'{dir}theory/td_th/{th_files[i]}',skiprows=1)
        #th = th[th[:,0]<1]
        th[:,0] += -0.0

        thavg0 = np.mean(th[th[:,0]>time_avg_cut,qstate_ni])
        thavg1 = np.mean(th[th[:,0]>time_avg_cut,qstate_co])
        th[:,qstate_ni] *= avg0/thavg0
        th[:,qstate_co] *= avg1/thavg1

        if i==0:
            axs[plot].plot(th[:,0],th[:,qstate_ni], linestyle='--',linewidth=3,color='r', label=th_labels[i])
            axs[plot].plot(th[:,0], th[:,qstate_co], linestyle='--',linewidth=3,color='b')
        elif i==1:
            axs[plot].plot(th[:,0],th[:,qstate_ni], linestyle=':',linewidth=3,color='r', label=th_labels[i])
            axs[plot].plot(th[:,0], th[:,qstate_co], linestyle=':',linewidth=3,color='b')

    axs[plot].set_xscale('log')
    axs[plot].xaxis.set_minor_formatter(FormatStrFormatter("%.0e"))
    axs[plot].tick_params(axis='both', which='minor', labelsize=10, rotation=45)
    axs[plot].tick_params(axis='both', which='major', labelsize=16, rotation=45)
    axs[plot].set_xlim([1e-2,t_bounds[1]])
    axs[plot].set_xlabel('Time since trigger [s]')
    axs[plot].set_ylabel('Counts')
    axs[plot].minorticks_on()
    axs[plot].legend()
    axs[plot].set_title(f'{Z}    {voltage_labels[0]} kV   {current_labels[0]} mA')
    leg = axs[plot].get_legend()
    leg.legendHandles[2].set_color('k')
    leg.legendHandles[3].set_color('k')
plt.subplots_adjust(hspace=0.5)
plt.show()
