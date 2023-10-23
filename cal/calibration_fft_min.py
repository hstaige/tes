import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/tes/td_data/'
file = '20221220_0000_'
states = ['A','B','K']

binsize = .001
e_bounds = [2610,2630]
t_bounds = [0,1]

data = np.empty((1,2))
for state in states:
    data = np.vstack((data,np.load(dir+file+state+'.npy')))

data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
counts, _ = np.histogram(data[:,1], bins=bin_edges)
bin_centers = bin_edges[:-1]+binsize/2

N = len(counts)
T = binsize
xf = fftfreq(N,T)[1:N//2]

plt.plot(xf,2/N*np.abs(fft(counts)[1:N//2]))
plt.xlabel('Freq [Hz]')
plt.ylabel('Magnitude (arb)')
plt.minorticks_on()
plt.show()
