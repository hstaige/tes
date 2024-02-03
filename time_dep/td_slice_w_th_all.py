import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, lfilter

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/dec22_data/'
file = 'td_data/20221221_'
states = ['0002_V','0002_X','0002_Z','0002_AB','0002_AD','0002_AF','0002_T']
current_labels = [18.4, 9.2, 13.8, 23, 27.6, 32.2, 36.8]

bins = .002

#e_bounds0 = [1140,1146]
#e_bounds1 = [1182,1188]

e_bounds_ni = [1199,1209]
e_bounds_co = [1236,1246]
e_bounds_e2m3 = [836,846]
t_bounds = [0,3]

th_files = os.listdir(dir+'theory/td_th/Nd/to_plot/')
th_files.sort()

bin_edges = np.arange(t_bounds[0], t_bounds[1]+bins, bins) # constant bins

def data_slice_nans(dir, file, state, bin_edges, e_range, t_range):

    data = np.load(dir+file+state+'.npy')

    data = data[(data[:,1]>t_range[0]) & (data[:,1]<t_range[1])]
    data = data[(data[:,0]>e_range[0]) & (data[:,0]<e_range[1])]
    counts, _ = np.histogram(data[:,1], bins=bin_edges)

    uncert = [(count**(1/2)) for count in counts]

    f0 = 20
    Q = 3
    a, b = iirnotch(f0, Q, 1 / bins)
    counts = lfilter(a, b, counts)

    return counts, uncert

bin_centers = [(bin+binp1)/2 for bin, binp1 in zip(bin_edges[:-1], bin_edges[1:])]

df_out = pd.DataFrame({'time_since_trigger': bin_centers})
print(df_out)

for state, I in zip(states, current_labels):
    counts_ni, uncert_ni = data_slice_nans(dir, file, state, bin_edges, e_bounds_ni, t_bounds)
    df_out[f'ni_{I:.1f}'] = counts_ni

    counts_co, uncert_co = data_slice_nans(dir, file, state, bin_edges, e_bounds_co, t_bounds)
    df_out[f'co_{I:.1f}'] = counts_co

    counts_e2m3, uncert_e2m3 = data_slice_nans(dir, file, state, bin_edges, e_bounds_e2m3, t_bounds)
    df_out[f'e2m3_{I:.1f}'] = counts_e2m3

df_out.to_csv(dir+'td_data/filt_slices_decay.csv')


plt.errorbar(bin_centers,counts_ni,uncert_ni,ls='none',color='r')
plt.scatter(bin_centers,counts_ni,label=f'{e_bounds_ni} Ni-like',color='r')

plt.errorbar(bin_centers,counts_co,uncert_co,ls='none',color='g')
plt.scatter(bin_centers,counts_co,label=f'{e_bounds_co} Co-like',color='g')

plt.errorbar(bin_centers,counts_e2m3,uncert_e2m3,ls='none',color='b')
plt.scatter(bin_centers,counts_e2m3,label=f'{e_bounds_e2m3} e2/m3',color='b')

plt.xscale('log')
plt.title('Nd   2.05 kV   36.8 mA')
plt.xlim([0.01,3])
plt.xlabel('Time since trigger [ms]')
plt.ylabel(f'Counts')
plt.minorticks_on()
plt.legend()
plt.show()
