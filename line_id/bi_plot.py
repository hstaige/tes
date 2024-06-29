import re
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def gaussian(x, A, mu, sigma):
    return A/(sigma * math.sqrt(2*math.pi)) * np.exp(-(x-mu)**2/(2*sigma**2))

def midpoints(x):
    return (x[:-1]+x[1:])/2

def format(line):
    line = re.sub(r'([0-9]+[a-z][0-9]+)\s([0-9]+[a-z][0-9]+)',r'\1.\2',line)
    line = re.sub(r'(0)\s([0-9]+[a-z][0-9]+)',r'\1.\2',line)
    line = re.sub(r'(0)\s([0-9]+[a-z]*)',r'\1.\2',line)
    line = re.sub(r'J=([0-9]+)',r'\1',line)
    line = re.sub(r'\s+',',',line)
    line = re.sub(r'([0-9]+)e',r'\1E',line)
    line = line.split(r',')
    return line

def theory_csv(input_dat):
    with open(input_dat, 'r') as file:
        theory = file.readlines()

    theory = [re.sub(r':|-\s|\s\|','',line) for line in re.split(r'\|',''.join([i for sb in theory for i in sb]))]
    theory = [format(line) for line in theory]

    for i,line in enumerate(theory):
        if len(line)<12:
            theory[i][:0]=theory[i-1][:3]

    theory = [list(filter(None, line[:-1])) for line in theory]

    theory = pd.DataFrame(theory[:-1], columns=['Energy','Total_intensity','Charge','Lower_config','Lower_index','Lower_J','Upper_config','Upper_index','Upper_J','Intensity'])
    return theory[['Energy','Total_intensity']].to_numpy()


dir = '/home/tim/research/apr24_data/theory'
th_data = theory_csv(f'{dir}/out.m.qn')

dir = '/home/tim/research/apr24_data/data_by_state/'
run = '20240419_0000'
states = ['B_OFF','C_OFF','D_OFF']
data = np.empty((2,0))
for state in states:
    data = np.hstack((data,np.load(f'{dir}{run}_{state}.npy')))
data = data[(0,1),:].T

for line in th_data:
    center = float(line[0])
    amp = float(line[1])
    x = np.arange(center-10,center+10,0.5)
    plt.plot(x,gaussian(x,amp*800,center,2),color='k')
    plt.annotate(f'{center:.1f}',(center,amp*800*.39/2))

data, bins = np.histogram(data[:,0],np.arange(2000,6000,1))
plt.plot(midpoints(bins),data)
plt.show()
