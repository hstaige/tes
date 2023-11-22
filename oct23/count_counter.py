import os
import numpy as np 

dir = '/home/tim/research/EBIT-TES-Data/data_by_state/'

files = os.listdir(dir)

def en_slice_counts(data_arr,energy):
    return np.sum((data_arr[0,:]>(energy-5)) & (data_arr[0,:]<(energy+5)))

int_time_list = []
for file in files:
    data_arr = np.load(f'{dir}{file}')

    int_time = (np.max(data_arr[1,:]) - np.min(data_arr[1,:]))*1e-9 # total integration time [s]
    int_time_list.append(int_time)
    # cycle_time = round(np.max(data_arr[2,:])*1e3) # cycle time [ms]
    # total_counts = np.sum(data_arr[1,:]) # total counts
    # pr_ni_counts = en_slice_counts(data_arr,1142)
    # pr_co_counts = en_slice_counts(data_arr,1182)
    # nd_ni_counts = en_slice_counts(data_arr,1203)
    # nd_co_counts = en_slice_counts(data_arr,1241)
ind = np.argsort(int_time_list)[-10:]
for i in ind:
    print(f'{int_time_list[i]:.0f}',files[i])