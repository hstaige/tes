import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/tes/td_data/'
# file = '20221221_0002_'
# states = ['T','AF','AD','AB','V','Z','X']
# Nislices = [[852,872],[918,952],[1138,1178],[1197,1211],[1349,1367],[1515,1552],[1701,1743]]
# Coslices = [[872,917],[970,1018],[1224,1272],[1372,1406],[1560,1618]]
# M3slices = [[827,851]]

file = '20221221_0002_'
states = ['K','M']
Nislices = [[1140,1146]]
Coslices = [[1140,1146]]
M3slices = []

binsize = .003
t_bounds = [0,3]

bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
bin_centers = bin_edges[:-1]+binsize/2

out = np.zeros((len(bin_centers),len(states)*3+1))
summed = np.zeros_like(bin_centers)

t = bin_centers*1000
out[:,0] = t

header='Time,'

i = 1
for state in states:
    data = np.load(dir+file+state+'.npy')


    summed = np.zeros_like(bin_centers)
    for slice in Nislices:
        temp_data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]
        temp_data = temp_data[(temp_data[:,0]>slice[0]) & (temp_data[:,0]<slice[1])]

        counts, _ = np.histogram(temp_data[:,1], bins=bin_edges)
        summed += counts
    #out[:,i] = summed
    max = np.max(summed)
    #plt.plot(t,(summed/max+(i-1)*.3),color='b')
    asummed = np.pad(summed,50,mode='edge')
    asummed = np.convolve(asummed, np.ones(50)/50, mode='valid')
    summed = [i if (i>(a-.1)) else np.NaN for i,a in zip(summed,asummed[:-51])]
    out[:,i] = np.array(asummed[:-51])*max
    #out[:,i] = np.array(summed)*max
    header += f'{state} Ni,'

    #plt.plot(t,(summed/max+(i-1)*.3),color='b')


    summed = np.zeros_like(bin_centers)
    for slice in Coslices:
        temp_data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]
        temp_data = temp_data[(temp_data[:,0]>slice[0]) & (temp_data[:,0]<slice[1])]

        counts, _ = np.histogram(temp_data[:,1], bins=bin_edges)
        summed += counts
    #out[:,i+1] = summed
    max = np.max(summed)
    #plt.plot(t,(summed/max+(i-1)*.3),color='g')
    asummed = np.pad(summed,50,mode='edge')
    asummed = np.convolve(asummed, np.ones(50)/50, mode='valid')
    summed = [i if (i>(a-.1)) else np.NaN for i,a in zip(summed,asummed[:-51])]
    #plt.plot(t,(summed/max+(i-1)*.3),color='g')
    out[:,i+1] = np.array(asummed[:-51])*max
    #out[:,i+1] = np.array(summed)*max
    header += f'{state} Co,'


    summed = np.zeros_like(bin_centers)
    for slice in M3slices:
        temp_data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]
        temp_data = temp_data[(temp_data[:,0]>slice[0]) & (temp_data[:,0]<slice[1])]

        counts, _ = np.histogram(temp_data[:,1], bins=bin_edges)
        summed += counts
    out[:,i+2] = summed
    max = np.max(summed)
    #plt.plot(t,(summed/max+(i-1)*.3),color='b')
    asummed = np.pad(summed,50,mode='edge')
    asummed = np.convolve(asummed, np.ones(50)/50, mode='valid')
    summed = [i if (i>(a-.1)) else np.NaN for i,a in zip(summed,asummed[:-51])]
    out[:,i+2] = np.array(asummed[:-51])
    #out[:,i+2] = np.array(summed)*max
    header += f'{state} E2M3,'

    i += 3
        
plt.show()
np.savetxt(dir+f'20221221_CoNi_{binsize*1000}ms_ma.csv', out, fmt='%.1f', delimiter=',', header=header)