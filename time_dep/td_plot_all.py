
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/tes/td_data/'

states = ['20221221_0002_T',
          '20221221_0002_V',
          '20221221_0002_X',
          '20221221_0002_Z',
          '20221221_0002_AB',
          '20221221_0002_AD',
          '20221221_0002_AF']

for state in states:
    data = np.load(dir+state+'.npy')
    binsize = 1

    e_bounds = [500,2000]
    t_bounds = [0,1]

    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    bin_edges = np.arange(e_bounds[0], e_bounds[1]+binsize, binsize)
    counts, _ = np.histogram(data[:,0], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    plt.plot(bin_centers,counts,label=state)
plt.xlabel('Photon Energy [eV]')
plt.ylabel(f'Counts ({e_bounds} eV)')
plt.minorticks_on()
plt.legend()
plt.show()