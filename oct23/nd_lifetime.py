import numpy as np
import matplotlib.pyplot as plt
import my_utils as utils

dir = '/home/tim/research/EBIT-TES-Data/data_by_state/'
files = ['20231014_0006_I.npy','20231014_0006_K.npy']
labels = ['Nd; 1s; 10ms off','Nd; 1s; 10ms off']
t_ranges = [[.979,.989],[.979,.989]]
e_range = [832,852]
t_binsize = 0.001
e_binsize = 1

for t_range, file, label in zip(t_ranges,files,labels):
    plt.figure()
    data_arr = np.load(dir+file)

    data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
    data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]

    t_bin_edges = np.arange(t_range[0],t_range[1],t_binsize)
    e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    print(np.sum(counts))
    plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
    plt.title(f'{file}; {label}')


t_range = [0,1]
e_range = [500,1250]
t_binsize = 0.0005
e_binsize = 1

for file, label in zip(files,labels):
    plt.figure()
    data_arr = np.load(dir+file)

    data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
    data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

    t_bin_edges = np.arange(t_range[0],t_range[1],t_binsize)
    e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    plt.title(f'{file}; {label}')
    plt.pcolormesh(t_bin_edges,e_bin_edges,counts)


plt.figure()
t_ranges = [[.979,.989],[.979,.989]]
e_range = [832,852]
t_binsize = 0.0005
e_binsize = 1

all_states_en = np.empty((0))
all_states_t = np.empty((0))
for t_range, file in zip(t_ranges,files):
    data_arr = np.load(dir+file)

    data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
    data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]
    all_states_en = np.append(all_states_en, data_arr[0,:])
    data_arr[2,:] -= np.min(data_arr[2,:])
    all_states_t = np.append(all_states_t, data_arr[2,:])

t_bin_edges = np.arange(0,.01,t_binsize)
e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
print(np.sum(counts))
plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
plt.xlabel('Energy [eV]')
plt.title('All states summed')


plt.figure()
t_ranges = [[0,1],[0,1]]
e_range = [832,852]
t_binsize = 0.0005
e_binsize = 1

all_states_en = np.empty((0))
all_states_t = np.empty((0))
for t_range, file in zip(t_ranges,files):
    data_arr = np.load(dir+file)

    data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
    data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]
    all_states_en = np.append(all_states_en, data_arr[0,:])
    #data_arr[2,:] -= np.min(data_arr[2,:])
    all_states_t = np.append(all_states_t, data_arr[2,:])

t_bin_edges = np.arange(0,1,t_binsize)
e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
print(np.sum(counts))
plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
plt.xlabel('Energy [eV]')
plt.title('All states summed')


plt.figure()
t_ranges = [[.979,.989],[.979,.989]]
e_range = [832,852]
t_binsize = 0.0005
e_binsize = 1

all_states_en = np.empty((0))
all_states_t = np.empty((0))
for t_range, file in zip(t_ranges,files):
    data_arr = np.load(dir+file)

    data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
    data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]
    all_states_en = np.append(all_states_en, data_arr[0,:])
    data_arr[2,:] -= np.min(data_arr[2,:])
    all_states_t = np.append(all_states_t, data_arr[2,:])

t_bin_edges = np.arange(0,.01,t_binsize)
e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
print(np.sum(counts))
plt.plot(utils.midpoints(t_bin_edges),np.sum(counts,axis=0))
plt.xlabel('Time since anode switch [s]')
plt.title('All states summed')
plt.show()

plt.show()
quit()