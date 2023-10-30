import re
import h5py
import numpy as np 
import matplotlib.pyplot as plt 
import my_utils as utils

h5_files = ['/home/tim/research/oct23_data/massgui_export_20231014_0006.h5',
            '/home/tim/research/oct23_data/massgui_export_20231015_0000.h5',
            '/home/tim/research/oct23_data/massgui_export_20231016_0001.h5',
            '/home/tim/research/oct23_data/massgui_export_20231017_0000.h5',
            '/home/tim/research/oct23_data/massgui_export_20231017_0001.h5',
            '/home/tim/research/oct23_data/massgui_export_20231018_0000.h5']


states = [[] for i in range(len(h5_files))]
files_like_states = [[] for i in range(len(h5_files))]

for i,h5_file in enumerate(h5_files):
    h5 = h5py.File(h5_file, 'r')
    chan_keys = list(h5.keys())
    state_keys = list(h5[chan_keys[0]].keys())
    states[i].append([i for i in state_keys if (bool(re.match(r'[A-Z][A-Z]?',i))) & (len(i)<3)])
    files_like_states[i].append([h5_file for i in state_keys if (bool(re.match(r'[A-Z][A-Z]?',i))) & (len(i)<3)])
    