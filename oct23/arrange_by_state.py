import re
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import my_utils as utils

h5_files = ['/home/tim/research/oct23_data/20231014_0006.h5',
            '/home/tim/research/oct23_data/20231015_0000.h5',
            '/home/tim/research/oct23_data/20231016_0001.h5',
            '/home/tim/research/oct23_data/20231017_0000.h5',
            '/home/tim/research/oct23_data/20231017_0001.h5',
            '/home/tim/research/oct23_data/20231018_0000.h5']

save_dir = '/home/tim/research/EBIT-TES-Data/data_by_state/'

for h5_file in h5_files:
    h5 = h5py.File(h5_file, 'r')
    day, run_num = re.search(r'([0-9]+)_([0-9]+).h5',h5_file).group(1,2)
    print(day,run_num)
    chan_keys = list(h5.keys())
    state_keys = list(h5[chan_keys[0]].keys())
    state_keys = [i for i in state_keys if (bool(re.match(r'[A-Z][A-Z]?',i))) & (len(i)<3)]
    for state in state_keys:
        data = utils.load_state_from_h5(h5_file,state)
        np.save(f'{save_dir}{day}_{run_num}_{state}',data)