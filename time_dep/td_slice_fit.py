import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, lfilter

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/tes/'
file = 'td_data/20221221_'
states = ['0002_T','0002_V']
states_labels = ['36.8mA','18.4mA']

bins = .002

#e_bounds0 = [1140,1146]
#e_bounds1 = [1182,1188]
e_bounds_ni = [1199,1209]
e_bounds_co = [1236,1246]
t_bounds = [0,3]

def data_slice(data, binsize, e_range, t_range):
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]
    data = data[(data[:,0]>e_range[0]) & (data[:,0]<e_range[1])]

    bin_edges = np.arange(t_range[0], t_range[1]+binsize, binsize)
    counts, _ = np.histogram(data[:,1], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    bin_centers = bin_centers[(counts>0)]*1000
    counts = counts[counts>0]

    f0 = 20
    Q = 3
    a, b = iirnotch(f0, Q, 1 / bins)
    counts = lfilter(a, b, counts)
    return bin_centers, counts

th_files = os.listdir(dir+'theory/td_th/Nd/')
th_files =  np.array([i for i in th_files if re.match(r'CX.+\.dat',i)])
e_dens = np.array([float(re.search(r'CX3e12_([0-9]e[0-9]+)_',th_file)[1]) for th_file in th_files])

sorted_e_dens = np.argsort(e_dens)
e_dens = e_dens[sorted_e_dens]
th_files = th_files[sorted_e_dens]

th_data = np.loadtxt(f'{dir}theory/td_th/Nd/{th_files[0]}',skiprows=1)[:,(0,6,7)]
th_data[:,0] *= 1e3
th_data[:,0] += 18
for th_file in th_files[1:]:
    th_data = np.hstack((th_data,np.loadtxt(f'{dir}theory/td_th/Nd/{th_file}',skiprows=1)[:,(6,7)]))

print(th_data.shape)
print(len(e_dens))
for state,label in zip(states,states_labels):
    data = np.load(dir+file+state+'.npy')
    bin_centers_ni, counts_ni = data_slice(data, bins, e_bounds_ni, t_bounds)
    bin_centers_co, counts_co = data_slice(data, bins, e_bounds_co, t_bounds)

    time_avg_cut = 2000
    avg_ni = np.mean(counts_ni[bin_centers_ni>time_avg_cut])
    avg_co = np.mean(counts_co[bin_centers_co>time_avg_cut])

    ni_resids = np.zeros(len(th_data[0,1::2]))
    for i,ni_th in enumerate(th_data[:,1::2].T):
        ni_th_avg = np.mean(ni_th[th_data[:,0]>time_avg_cut])
        ni_th *= avg_ni/ni_th_avg
        ni_th_interp = interp1d(th_data[:,0],ni_th)
        ni_resids[i] = np.sum([(y-ni_th_interp(t)) for t,y in zip(bin_centers_ni,counts_ni)])
    plt.plot(e_dens,ni_resids,label=f'{label} ni-like')

    co_resids = np.zeros(len(th_data[0,1::2]))
    for i,co_th in enumerate(th_data[:,2::2].T):
        co_th_avg = np.mean(co_th[th_data[:,0]>time_avg_cut])
        co_th *= avg_co/co_th_avg
        co_th_interp = interp1d(th_data[:,0],co_th)
        co_resids[i] = np.sum([(y-co_th_interp(t)) for t,y in zip(bin_centers_co,counts_co)])
    plt.plot(e_dens,co_resids,label=f'{label} co-like')

plt.ylabel('residual (data-theory)')
plt.xlabel('e- density')
plt.axhline(0,linestyle=':',color='k')
plt.legend()
plt.xscale('log')
plt.show()