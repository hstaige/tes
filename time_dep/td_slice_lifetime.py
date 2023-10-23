import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import iirnotch, filtfilt, lfilter
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size': 20})

dir = "/home/tim/research/tes/"
file = 'td_data/'
# states = ['0002_V','0002_X','0002_Z','0002_AB','0002_AD','0002_AF','0002_T']
# current_labels = [18.4, 9.2, 13.8, 23, 27.6, 32.2, 36.8]

co_color = '#0077bb'
ni_color = '#cc3311'
e2m3_color = '#117733'

plots = 0

if plots==0:
    Z = 'Nd'
    states = ['20221220_0000_H','20221220_0000_J','20221220_0000_L']
    th_files = ['Nd/CX3e12_3e12_iontotpp.dat']
    voltage_labels = [2.25]
    current_labels = [35.9]
    e_bounds_ni = [1199,1209]
    e_bounds_co = [1236,1246]
    e_bounds_e2m3 = [836,846]

data = np.empty((1,2))
for state in states:
    data = np.vstack((data,np.load(dir+file+state+'.npy')))

bins = .0001
t_bounds = [0,1]

bin_edges = np.logspace(np.log10(0.01),np.log10(3.0), 500)

def data_slice(data, bin_edges, e_range, t_range):


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
    counts_ni, uncert_ni = data_slice(data, bin_edges, e_bounds_ni, t_bounds)
    df_out[f'{Z}_ni_{I:.1f}mA_{V}kV'] = counts_ni
    df_out[f'{Z}_ni_{I:.1f}mA_{V}kV_uncert'] = uncert_ni

    counts_co, uncert_co = data_slice(data, bin_edges, e_bounds_co, t_bounds)
    df_out[f'{Z}_co_{I:.1f}mA_{V}kV'] = counts_co
    df_out[f'{Z}_co_{I:.1f}mA_{V}kV_uncert'] = uncert_co

    counts_e2m3, uncert_e2m3 = data_slice(data, bin_edges, e_bounds_e2m3, t_bounds)
    df_out[f'e2m3_{I:.1f}mA'] = counts_e2m3
    df_out[f'e2m3_{I:.1f}mA_uncert'] = uncert_e2m3

df_out.to_csv(dir+f'td_data/filt_slices_{Z}{voltage_labels[0]}kV{current_labels[0]}mA.csv')

fig, ax = plt.subplots()

ax.errorbar(bin_centers,counts_co,uncert_co,ls='none',color=co_color, alpha = 0.75)
ax.scatter(bin_centers,counts_co,label=f'Co-like',color=co_color, s=16)

ax.errorbar(bin_centers,counts_ni,uncert_ni,ls='none',color=ni_color, alpha = 0.75)
ax.scatter(bin_centers,counts_ni,label=f'Ni-like',color=ni_color, s=16)

ax.errorbar(bin_centers,counts_e2m3,uncert_ni,ls='none',color=e2m3_color, alpha = 0.75)
ax.scatter(bin_centers,counts_e2m3,label=f'E2/M3',color=e2m3_color, s=16)

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.25, 0.6, 0.2, 0.2]
#ax2 = fig.axes([left, bottom, width, height])
ax2 = inset_axes(ax,width='30%', height='30%', loc=1)

def inset_plot(axis,data):
    
    binsize_e2m3 = .0005
    binsize_e1 = 0.00025

    e_bounds_ni = [1199,1209]
    e_bounds_co = [1236,1246]
    e_bounds_e2m3 = [836,846]
    t_bounds = [.974,.989]

    def decay_func(t,A,B,tau):
        return A*np.exp(-(t-B)/tau)


    #t_bounds = [.975,1]
    data_e2m3 = data[(data[:,0]>e_bounds_e2m3[0]) & (data[:,0]<e_bounds_e2m3[1])]
    data_e2m3 = data_e2m3[(data_e2m3[:,1]>t_bounds[0]) & (data_e2m3[:,1]<t_bounds[1])]

    data_co = data[(data[:,0]>e_bounds_co[0]) & (data[:,0]<e_bounds_co[1])]
    data_co = data_co[(data_co[:,1]>t_bounds[0]) & (data_co[:,1]<t_bounds[1])]

    data_ni = data[(data[:,0]>e_bounds_ni[0]) & (data[:,0]<e_bounds_ni[1])]
    data_ni = data_ni[(data_ni[:,1]>t_bounds[0]) & (data_ni[:,1]<t_bounds[1])]

    bin_edges_e2m3 = np.arange(t_bounds[0], t_bounds[1]+binsize_e2m3, binsize_e2m3)
    bin_edges_e1 = np.arange(t_bounds[0], t_bounds[1]+binsize_e1, binsize_e1)
    counts_e2m3, _ = np.histogram(data_e2m3[:,1], bins=bin_edges_e2m3)
    counts_co, _ = np.histogram(data_co[:,1], bins=bin_edges_e1)
    counts_ni, _ = np.histogram(data_ni[:,1], bins=bin_edges_e1)
    bin_centers_e2m3 = bin_edges_e2m3[:-1]+binsize_e2m3/2
    bin_centers_e1 = bin_edges_e1[:-1]+binsize_e1/2

    # remove zero count bins and scale t to ms
    bin_centers_e2m3 = bin_centers_e2m3*1000
    bin_centers_e1 = bin_centers_e1*1000
    #counts = counts[counts>0]

    counts_e2m3 = np.array([count if count>0 else 0.1 for count in counts_e2m3])
    counts_co = np.array([count if count>0 else 0.1 for count in counts_co])
    counts_ni = np.array([count if count>0 else 0.1 for count in counts_ni])

    weights_e2m3 = [(count**(1/2)) if count>0.1 else 0 for count in counts_e2m3]
    weights_co = [(count**(1/2)) if count>0.1 else 0 for count in counts_co]
    weights_ni = [(count**(1/2)) if count>0.1 else 0 for count in counts_ni]
    
    guess = [16,978,3]
    bin_centers_fit = bin_centers_e2m3[bin_centers_e2m3>979]
    counts_e2m3_fit = counts_e2m3[bin_centers_e2m3>979]
    popt,pcov = curve_fit(decay_func,bin_centers_fit,counts_e2m3_fit,p0=guess)

    #plt.plot(bin_centers,decay_func(bin_centers,guess[0],guess[1],guess[2]))
    fit_x = np.linspace(979,989,100)
    axis.plot(fit_x,decay_func(fit_x,popt[0],popt[1],popt[2]),label='Exp Fit',zorder=10,color='k')
    
    axis.scatter(bin_centers_e2m3,counts_e2m3,color=e2m3_color,zorder=100)
    axis.errorbar(bin_centers_e2m3,counts_e2m3,weights_e2m3,color=e2m3_color,ls='none',zorder=100)

    axis.scatter(bin_centers_e1,counts_co,color=co_color)
    axis.errorbar(bin_centers_e1,counts_co,weights_co,color=co_color,ls='none')  

    axis.scatter(bin_centers_e1,counts_ni,color=ni_color)
    axis.errorbar(bin_centers_e1,counts_ni,weights_ni,color=ni_color,ls='none')

inset_plot(ax2,data)


ax.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylim([.1,150])
ax.xaxis.set_minor_formatter(FormatStrFormatter("%.0e"))
ax.tick_params(axis='both', which='minor', labelsize=10, rotation=45)
ax.set_xlim([1e-2,t_bounds[1]])
ax.set_ylim([0,450])
ax.tick_params(axis='both', which='major', labelsize=16, rotation=45)
ax2.tick_params(axis='both', which='major', labelsize=10, rotation=45)
ax.set_title(f'{Z}    {voltage_labels[0]} kV   {current_labels[0]} mA')
ax.set_xlabel('Time since trigger [s]')
ax.set_ylabel('Counts')
ax2.set_xlabel('Time since trigger [ms]')
ax2.set_ylabel('Counts')
plt.minorticks_on()
ax.legend(loc='upper left')
ax2.legend()

plt.show()