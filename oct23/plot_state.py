import numpy as np
import matplotlib.pyplot as plt
import my_utils as utils

dir = '/home/tim/research/EBIT-TES-Data/data_by_state/'
files = ['20231015_0000_H.npy','20231015_0000_G.npy','20231015_0000_I.npy','20231017_0001_D.npy']
labels = ['Pr; 0.5s; 10ms off','Pr; 1s; 10ms off','Pr; 1s; 10ms off','Pr/Ne; 1s; 50ms off']
t_ranges = [[.479,.489],[.979,.989],[.979,.989],[.939,.989]]
e_range = [788,808]
t_binsize = 0.001
e_binsize = 1
#summed = np.zeros((len()))
for t_range, file, label in zip(t_ranges,files,labels):
    plt.figure()
    data_arr = np.load(dir+file)

    data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
    data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]

    t_bin_edges = np.arange(t_range[0],t_range[1],t_binsize)
    e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    print(np.sum(counts))
    print(counts.shape)
    plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
    plt.title(f'{file}; {label}')


# t_range = [0,1]
# e_range = [500,1250]
# t_binsize = 0.0005
# e_binsize = 1

# for file, label in zip(files,labels):
#     plt.figure()
#     data_arr = np.load(dir+file)

#     data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
#     data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

#     t_bin_edges = np.arange(t_range[0],t_range[1],t_binsize)
#     e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

#     counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
#     plt.title(f'{file}; {label}')
#     plt.pcolormesh(t_bin_edges,e_bin_edges,counts)

plt.show()
quit()