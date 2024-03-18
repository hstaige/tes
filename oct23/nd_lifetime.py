import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, fit_report
from lmfit import Model
from mass import MLEModel
import tools

dir = '/home/tim/research/EBIT-TES-Data/data_by_state/'
files = ['20231014_0006_I.npy','20231014_0006_K.npy']
labels = ['Nd; 1s; 10ms off','Nd; 1s; 10ms off']

# t_ranges = [[.979,.989],[.979,.989]]
# e_range = [832,852]
# t_binsize = 0.001
# e_binsize = 1

# for t_range, file, label in zip(t_ranges,files,labels):
#     plt.figure()
#     data_arr = np.load(dir+file)

#     data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
#     data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]

#     t_bin_edges = np.arange(t_range[0],t_range[1],t_binsize)
#     e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

#     counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
#     print(np.sum(counts))
#     plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
#     plt.title(f'{file}; {label}')


t_range = [0,.989]
e_range = [500,1250]
t_binsize = 0.0005
e_binsize = 1

all_states_en = np.empty((0))
all_states_t = np.empty((0))
for file in files:
    data_arr = np.load(dir+file)

    data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
    data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]
    all_states_en = np.append(all_states_en, data_arr[0,:])
    data_arr[2,:] -= np.min(data_arr[2,:])
    all_states_t = np.append(all_states_t, data_arr[2,:])

t_bin_edges = np.arange(t_range[0],t_range[1],t_binsize)
e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
plt.pcolormesh(t_bin_edges,e_bin_edges,counts)


plt.figure()
t_ranges = [[.979,.989],[.979,.989]]
e_range = [832,852]
t_binsize = 0.0005
e_binsize = .1

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
e_kern = np.zeros(len(e_bin_edges))
for e in all_states_en:
    e_kern+=tools.gaussian(e_bin_edges,1,e,2)

counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])

def twogauss(e, A1, mu1, sigma1, A2, mu2, sigma2):
    return A1*tools.gaussian(e,A1,mu1,sigma1)+A2*tools.gaussian(e,A2,mu2,sigma2)

model = MLEModel(tools.gaussian)
params = model.make_params()
params["A"].set(30, min=0)
params["sigma"].set(2, min=1.9,max=2.1)
params["mu"].set(840, min=0)

x = tools.midpoints(e_bin_edges)

out = model.fit(np.sum(counts,axis=1), x=x, params=params)
print(fit_report(out))
plt.plot(x,tools.gaussian(x,out.params['A'],out.params['mu'],out.params['sigma']))
plt.plot(x,np.sum(counts,axis=1))
plt.xlabel('Energy [eV]')
plt.title('All states summed')
plt.plot(e_bin_edges,e_kern)
plt.show()


# plt.figure()
# t_ranges = [[0,1],[0,1]]
# e_range = [832,852]
# t_binsize = 0.0005
# e_binsize = 1

# all_states_en = np.empty((0))
# all_states_t = np.empty((0))
# for t_range, file in zip(t_ranges,files):
#     data_arr = np.load(dir+file)

#     data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
#     data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]
#     all_states_en = np.append(all_states_en, data_arr[0,:])
#     #data_arr[2,:] -= np.min(data_arr[2,:])
#     all_states_t = np.append(all_states_t, data_arr[2,:])

# t_bin_edges = np.arange(0,1,t_binsize)
# e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

# counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
# print(np.sum(counts))
# plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
# plt.xlabel('Energy [eV]')
# plt.title('All states summed')


# plt.figure()
# t_ranges = [[.979,.989],[.979,.989]]
# e_range = [832,852]
# t_binsize = 0.0005
# e_binsize = 1

# all_states_en = np.empty((0))
# all_states_t = np.empty((0))
# for t_range, file in zip(t_ranges,files):
#     data_arr = np.load(dir+file)

#     data_arr = data_arr[:,(data_arr[0,:]>e_range[0]) & (data_arr[0,:]<e_range[1])]
#     data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]
#     all_states_en = np.append(all_states_en, data_arr[0,:])
#     data_arr[2,:] -= np.min(data_arr[2,:])
#     all_states_t = np.append(all_states_t, data_arr[2,:])

# t_bin_edges = np.arange(0,.01,t_binsize)
# e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

# counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
# print(np.sum(counts))
# plt.plot(utils.midpoints(t_bin_edges),np.sum(counts,axis=0))
# plt.xlabel('Time since anode switch [s]')
# plt.title('All states summed')
# plt.show()

plt.figure()
t_ranges = [[0,.989],[0,.989]]
t_binsize = 0.001
e_binsize = .1

all_states_en = np.empty((0))
all_states_t = np.empty((0))
for t_range, file in zip(t_ranges,files):
    data_arr = np.load(dir+file)

    data_arr = data_arr[:,(data_arr[2,:]>t_range[0]) & (data_arr[2,:]<t_range[1])]
    all_states_en = np.append(all_states_en, data_arr[0,:])
    all_states_t = np.append(all_states_t, data_arr[2,:])

t_bin_edges = np.arange(.03,.989,t_binsize)
nie_bin_edges = np.arange(1193,1213,e_binsize)
mse_bin_edges = np.arange(832,852,e_binsize)

nicounts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [nie_bin_edges,t_bin_edges])
mscounts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [mse_bin_edges,t_bin_edges])
nicounts = np.sum(nicounts,axis=0)
mscounts = np.sum(mscounts,axis=0)



ts = tools.midpoints(t_bin_edges)
plt.plot(ts,nicounts)
plt.plot(ts,mscounts)
plt.show()
tau = np.zeros(len(ts))
ceqni = np.sum(nicounts[int(len(t_bin_edges)/2):])
ceqms = np.sum(nicounts[int(len(t_bin_edges)/2):])
for i,t in enumerate(ts):
    cms = mscounts[i]
    Cni = np.sum(nicounts[:i])
    Cms = np.sum(mscounts[:i])
    tau[i] = cms/(Cni-ceqni/ceqms*Cms)

plt.plot(ts,tau)
plt.show()