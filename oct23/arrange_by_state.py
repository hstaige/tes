import re
import h5py
import numpy as np 
import matplotlib.pyplot as plt 

h5_files = ['/home/tim/research/apr24_data/20240419/0000/20240419_0000.hdf5']

save_dir = '/home/tim/research/apr24_data/data_by_state/'

def load_state_from_h5(h5_file, state):
    h5 = h5py.File(h5_file, 'r')
    n = 0
    for key in h5.keys():
        n += len(h5[key][state]['energy'])

    energies = np.zeros(n)
    unixnanos = np.zeros(n)

    i0 = 0
    for key in h5.keys():
        g = h5[key][state]
        n = len(g['energy'])
        energies[i0:i0+n] = g['energy']
        unixnanos[i0:i0+n] = g['unixnano']
        i0+=n
    return np.vstack([energies,unixnanos])

for h5_file in h5_files:
    h5 = h5py.File(h5_file, 'r')
    #day, run_num = re.search(r'([0-9]+)_([0-9]+).h5',h5_file).group(1,2)
    day = '20240419'
    run_num = '0000'
    print(day,run_num)
    chan_keys = list(h5.keys())
    state_keys = list(h5[chan_keys[0]].keys())
    print(state_keys)
    state_keys = [i for i in state_keys if (bool(re.match(r'[A-Z]_(OFF)',i)))]
    for state in state_keys:
        data = load_state_from_h5(h5_file,state)
        np.save(f'{save_dir}{day}_{run_num}_{state}',data)