import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#floc = 'C:/Users/ahosi/OneDrive/Desktop/calibratedTES_Dec2022'
floc = '/home/tim/research/tes/calibratedTES_Dec2022'
#ddest = 'C:/data/Line_ID_Nd'

ddest = '/home/tim/research/tes/line_id'
date = '202212'
day = '21'
runnum = '0002'
#statelist = ['E','G','K','M','Q','R']
statelist = ['E']
minenergy = 600
maxenergy = 3000

mintime = 0
maxtime = 3
binsize = 1

file = f'{floc}/{date}{day}_{runnum}'
data = np.loadtxt(f'{file}_{statelist[0]}photonlist.csv', skiprows=1, delimiter=',')
if len(statelist)>1:
    for state in statelist[1:]:
        data = np.vstack((data,np.loadtxt(f'{file}_{state}photonlist.csv', skiprows=1, delimiter=',')))

data = data[(data[:,0]>=minenergy) & (data[:,0]<=maxenergy)]
#bin_edges = np.arange(mintime, maxtime+binsize, binsize)
bin_edges = np.arange(minenergy, maxenergy+binsize, binsize)

counts, _ = np.histogram(data, bins=bin_edges)
bin_centers = bin_edges[:-1]+binsize/2
#plt.plot(data[:,1],data[:,0])
plt.plot(bin_centers,counts,marker='.', zorder=5, label='Data')
plt.show()