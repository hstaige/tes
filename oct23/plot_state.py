import numpy as np
import matplotlib.pyplot as plt
import my_utils as utils

dir = '/home/tim/research/EBIT-TES-Data/data_by_state/'
file = '20231015_0000_H.npy'

data_arr = np.load(dir+file)
e_range = [500,1250]
t_range = [0,.995]

e_range = [788,808]
t_range = [.479,.489]

data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

t_bin_edges = np.arange(t_range[0],t_range[1],0.001)
e_bin_edges = np.arange(e_range[0],e_range[1],1)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
print(np.sum(counts))
plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
plt.title(file)
#plt.pcolormesh(t_bin_edges,e_bin_edges,counts)

plt.figure()
data_arr = np.load(dir+file)
e_range = [500,1250]
t_range = [0,.995]

data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

t_bin_edges = np.arange(t_range[0],t_range[1],0.001)
e_bin_edges = np.arange(e_range[0],e_range[1],1)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
plt.title(file)
plt.pcolormesh(t_bin_edges,e_bin_edges,counts)


plt.figure()
file = '20231015_0000_G.npy'

data_arr = np.load(dir+file)
e_range = [500,1250]
t_range = [0,.995]

e_range = [788,808]
t_range = [.979,.989]

data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

t_bin_edges = np.arange(t_range[0],t_range[1],0.001)
e_bin_edges = np.arange(e_range[0],e_range[1],1)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
print(np.sum(counts))
plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
plt.title(file)
#plt.pcolormesh(t_bin_edges,e_bin_edges,counts)

plt.figure()
file = '20231015_0000_G.npy'

data_arr = np.load(dir+file)
e_range = [500,1250]
t_range = [0,.995]

data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

t_bin_edges = np.arange(t_range[0],t_range[1],0.001)
e_bin_edges = np.arange(e_range[0],e_range[1],1)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
plt.title(file)
plt.pcolormesh(t_bin_edges,e_bin_edges,counts)


plt.figure()
file = '20231015_0000_I.npy'

data_arr = np.load(dir+file)
e_range = [500,1250]
t_range = [0,.995]

e_range = [788,808]
t_range = [.979,.989]

data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

t_bin_edges = np.arange(t_range[0],t_range[1],0.001)
e_bin_edges = np.arange(e_range[0],e_range[1],1)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
print(np.sum(counts))
plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
plt.title(file)
#plt.pcolormesh(t_bin_edges,e_bin_edges,counts)
plt.figure()
file = '20231015_0000_I.npy'

data_arr = np.load(dir+file)
e_range = [500,1250]
t_range = [0,.995]

data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

t_bin_edges = np.arange(t_range[0],t_range[1],0.001)
e_bin_edges = np.arange(e_range[0],e_range[1],1)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])

plt.title(file)
plt.pcolormesh(t_bin_edges,e_bin_edges,counts)

plt.figure()
file = '20231017_0001_D.npy'

data_arr = np.load(dir+file)
e_range = [500,1250]
t_range = [0,.995]

e_range = [788,808]
t_range = [.939,.989]

data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

t_bin_edges = np.arange(t_range[0],t_range[1],0.001)
e_bin_edges = np.arange(e_range[0],e_range[1],1)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
print(np.sum(counts))
plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
plt.title(file)
#plt.pcolormesh(t_bin_edges,e_bin_edges,counts)
plt.figure()
file = '20231017_0001_D.npy'

data_arr = np.load(dir+file)
e_range = [500,1250]
t_range = [0,.995]

data_arr = data_arr[:,(data_arr[0,:]>(e_range[0])) & (data_arr[0,:]<(e_range[1]))]
data_arr = data_arr[:,(data_arr[2,:]>(t_range[0])) & (data_arr[2,:]<(t_range[1]))]

t_bin_edges = np.arange(t_range[0],t_range[1],0.001)
e_bin_edges = np.arange(e_range[0],e_range[1],1)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])

plt.title(file)
plt.pcolormesh(t_bin_edges,e_bin_edges,counts)


plt.show()