import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

file = '/home/pcuser/Desktop/Periodicity_Testing/20230727_0011_Aphotonlist.csv'

data = np.loadtxt(file, skiprows=1, delimiter=',')

print(f'Working on {len(data)} points')
print(f'First two are {data[:2, 1]}')

data[:, 1] -= data[0,1]

print()
print(f'First two after shift are {data[:2, 1]}, highest is {data[-1, 1]}')

data = data[:, 1] * 1e-9 #put time in seconds

data = data[(1800<data) & (data<1803)]

print(f'Starting bins for {len(data)} data points')
binsize = 5e-3
bins = np.arange(0,data[-1], binsize) #millisecond bins from start to end
y, _ = np.histogram(data,bins)
print('Finished binning')

f, ax = plt.subplots()
plt.title(f'Data with binsize of {binsize}')
plt.hist(data, bins)
plt.show()


yf = fft(y)

N=len(y)
T = 1 / binsize
xf = fftfreq(N, T)[:N//2]
print(xf[:5])

print(N, yf[:6])

plt.plot(xf[1:], np.abs(yf[1:N//2]))
plt.show()