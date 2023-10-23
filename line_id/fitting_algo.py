import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d, expon

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/tes/td_data/'
file = '20221220_'
states = ['0000_H','0000_J','0000_L']

data = np.empty((1,2))
for state in states:
    data = np.vstack((data,np.load(dir+file+state+'.npy')))

def MLE_fit(data, shape, *params):
    if shape=='gaussian':
        A = params[0]
        mu = params[1]
        sigma = params[2]

        def gaussian_pdf(x,A,mu,sigma):
            return A/(sigma*math.sqrt(2*math.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

        def gaussian_logpdf(x,A,mu,sigma):
            return np.log(A/(sigma*math.sqrt(2*math.pi))) - (x-mu)**2/(2*sigma**2)

    if shape=='expon':
        tau = params[0]

binsize = .0005

e_bounds = [837,847]
t_bounds = [.979,.989]

data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

def decay_func(t,A,B,tau):
    return A*np.exp(-(t-B)/tau)

bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
counts, _ = np.histogram(data[:,1], bins=bin_edges)
bin_centers = bin_edges[:-1]+binsize/2

data_time = data[:,1]

x0 = data_time[0]
tau = (len(data_time))/np.sum(data_time)

x = np.linspace(np.min(bin_centers),np.max(bin_centers),1000)

plt.scatter(bin_centers,counts/np.max(counts),zorder=0)
plt.plot(x,decay_func(x,1,x0,tau),label=r' $e^{-(t-t_0)/ \tau}$'+r'   $\tau$'+f'={tau*1000:.2f}'+' ms',zorder=10)

plt.title('Nd')
plt.xlabel('Time since trigger [ms]')
plt.ylabel(f'Counts ({e_bounds} eV) norm')
plt.minorticks_on()
plt.legend()
plt.show()