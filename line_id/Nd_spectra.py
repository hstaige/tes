import math
import time
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#file1 = '/home/tim/research/tes/processed_NO_RMS/20221220_0000'
#states = ['H','J','L','P']

file1 = '/home/tim/research/tes/processed_NO_RMS/20221220_0001'
states = ['N','F']

summed = np.load(f'{file1}_{states[0]}.npy')
if len(states)>1:
    for state in states[1:]:
        summed = np.vstack((summed,np.load(f'{file1}_{state}.npy')))


plt.scatter(summed[:,1],summed[:,0],marker='.')
plt.show()