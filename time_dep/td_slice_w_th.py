import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from scipy.signal import iirnotch, lfilter

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/tes/'
file = 'td_data/20221221_'
states = ['0002_V']

bins = .002

#e_bounds0 = [1140,1146]
#e_bounds1 = [1182,1188]
e_bounds0 = [1199,1209]
e_bounds1 = [1236,1246]
t_bounds = [0,3]



data = np.empty((1,2))
for state in states:
    data = np.vstack((data,np.load(dir+file+state+'.npy')))

th_files = os.listdir(dir+'theory/td_th/Nd/to_plot/')
th_files.sort()

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


bin_centers_ni, counts_ni = data_slice(data, bins, e_bounds0, t_bounds)
bin_centers_co, counts_co = data_slice(data, bins, e_bounds1, t_bounds)

weights0 = [(count**(1/2)) for count in counts_ni]
weights1 = [(count**(1/2)) for count in counts_co]

plt.errorbar(bin_centers_ni,counts_ni,weights0,ls='none',color='k')
plt.scatter(bin_centers_ni,counts_ni,label=f'{e_bounds0} Ni-like',color='k')

plt.errorbar(bin_centers_co,counts_co,weights1,ls='none',color='grey')
plt.scatter(bin_centers_co,counts_co,label=f'{e_bounds1} Co-like',color='grey')


time_avg_cut = 1000
avg0 = np.mean(counts_ni[bin_centers_ni>time_avg_cut])
avg1 = np.mean(counts_co[bin_centers_co>time_avg_cut])

color = iter(cm.rainbow(np.linspace(0, 1, len(th_files))))
for file in th_files:
    qstate1 = 6
    qstate2 = 7
    th = np.loadtxt(f'{dir}theory/td_th/Nd/to_plot/{file}',skiprows=1)
    #th = th[th[:,0]<1]
    th[:,0] *= 1e3
    th[:,0] += 18

    thavg0 = np.mean(th[th[:,0]>time_avg_cut,qstate1])
    thavg1 = np.mean(th[th[:,0]>time_avg_cut,qstate2])
    th[:,qstate1] *= avg0/thavg0
    th[:,qstate2] *= avg1/thavg1

    th1_f = interp1d(th[:,0],th[:,qstate1])
    th2_f = interp1d(th[:,0],th[:,qstate2])
    ls1 = (np.sum([(th1_f(t)-y)**2 for t,y in zip(bin_centers_ni,counts_ni)]))**.5
    ls2 = (np.sum([(th2_f(t)-y)**2 for t,y in zip(bin_centers_co,counts_co)]))**.5
    print('LS',file,qstate1,ls1)
    print('LS',file,qstate2,ls2)

    c = next(color)
    plt.plot(th[:,0],th[:,qstate1], color=c, linestyle=':')
    plt.plot(th[:,0], th[:,qstate2], color=c, linestyle='-',label=file[:-13] + f' {ls1:.1f}, {ls2:.1f}')

plt.title('Nd   2.05 kV   36.8 mA')
#plt.xlim([0,500])
plt.xlabel('Time since trigger [ms]')
plt.ylabel(f'Counts')
plt.minorticks_on()
plt.legend()
plt.show()

#np.savetxt(dir+f'Pr_20221221_CoNi_K_{binsize*1000}ms.csv', out, fmt='%.1f', delimiter=',')
