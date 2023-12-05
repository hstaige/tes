import numpy as np
import matplotlib.pyplot as plt
import lmfit
from mass import MLEModel
import my_utils as utils

dir = '/home/tim/research/EBIT-TES-Data/data_by_state/'
files = ['20231015_0000_H.npy','20231015_0000_G.npy','20231015_0000_I.npy','20231017_0001_D.npy']
labels = ['Pr; 0.5s; 10ms off','Pr; 1s; 10ms off','Pr; 1s; 10ms off','Pr/Ne; 1s; 50ms off']

plot = 7
# 0: 1d, energy on x, ea state
# 1: 2d, ea state
# 2: 1d, energy on x, anode off and on
# 3: 1d, time on x
# 4: 1d, time on x, cumsum, lmfit
# 5: 1d, time on x, cumsum, mass fit
# 6: 1d, time on x, mass fit
# 7: 1d, time on x, small bins, mass fit

if plot==0:
    t_ranges = [[.479,.489],[.979,.989],[.979,.989],[.939,.989]]
    e_range = [788,808]
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

elif plot==1:
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

elif plot==2:
    plt.figure()
    t_ranges = [[.479,.488],[.979,.988],[.979,.988],[.939,.988]]
    e_range = [788,808]
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

    t_bin_edges = np.arange(0,.025,t_binsize)
    e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

    counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
    counts = np.sum(counts,axis=1)
    counts /= np.max(counts)
    print(np.sum(counts))
    plt.plot(utils.midpoints(e_bin_edges),counts,label='Beam off')

    t_ranges = [[0,1],[0,1],[0,1],[0,1]]
    e_range = [788,808]
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

    t_bin_edges = np.arange(0,1,t_binsize)
    e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

    counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
    counts = np.sum(counts,axis=1)
    counts /= np.max(counts)
    print(np.sum(counts))
    plt.plot(utils.midpoints(e_bin_edges),counts,label='Beam on')
    plt.xlabel('Energy [eV]')
    plt.title('All states summed')
    plt.legend()

elif plot==3:
    plt.figure()
    t_ranges = [[.479,.488],[.979,.988],[.979,.988],[.939,.988]]
    e_range = [788,808]
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

    t_bin_edges = np.arange(0,.025,t_binsize)
    e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

    counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
    print(np.sum(counts))
    plt.plot(utils.midpoints(t_bin_edges),np.sum(counts,axis=0))
    plt.xlabel('Time since anode switch [s]')
    plt.title('All states summed')
    plt.show()

elif plot==4:
    plt.figure()
    t_ranges = [[.479,.488],[.979,.988],[.979,.988],[.939,.988]]
    e_range = [788,808]
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

    all_states_t.sort()
    counts = [1 for i in all_states_t]
    tot_counts = np.sum(counts)
    counts_sum = tot_counts-np.cumsum(counts)
    uncertainties = [(i)**.5 if i!=0 else 1 for i in counts_sum]
    def residual(params, x, data, uncertainty):
        amp1 = params['amp1']
        tau1 = params['tau1']
        amp2 = params['amp2']
        tau2 = params['tau2']

        model = amp1*np.exp(-x/tau1) + amp2*np.exp(-x/tau2)
        return (data-model)/uncertainty
    
    params = lmfit.create_params(amp1=100, amp2=100, tau1=1e-4, tau2=1e-3)
    out = lmfit.minimize(residual, params, args=(all_states_t, counts_sum, uncertainties))
    print(out.params.pretty_print())
    plt.plot(all_states_t, counts_sum)
    plt.xlabel('Time since anode switch [s]')
    plt.title('Cumulative Sum')
    plt.show()

elif plot==5:
    plt.figure()
    t_ranges = [[.479,.488],[.979,.988],[.979,.988],[.939,.988]]
    e_range = [788,808]
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

    all_states_t.sort()
    counts = [1 for i in all_states_t]
    tot_counts = np.sum(counts)
    counts_sum = tot_counts-np.cumsum(counts)
    uncertainties = [(i)**.5 if i!=0 else 1 for i in counts_sum]


    def twoexp(t, A1, tau1, A2, tau2):
        return A1/tau1*np.exp(-t/tau1)+A2/tau2*np.exp(-t/tau2)
    model = MLEModel(twoexp)
    params = model.make_params()
    params["A1"].set(100, min=0)
    params["tau1"].set(10e-5, min=10e-6)
    params["A2"].set(10, min=0)
    params["tau2"].set(1e-3, min=2e-3)

    result = model.fit(counts_sum, t=all_states_t, params=params)
    result.plot()
    print(result.fit_report())
    plt.show()

elif plot==6:
    plt.figure()
    t_ranges = [[.479,.488],[.979,.988],[.979,.988],[.939,.988]]
    e_range = [788,808]
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

    t_bin_edges = np.arange(0,.025,t_binsize)
    e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

    counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
    counts = np.sum(counts,axis=0)

    def twoexp(t, A1, tau1, A2, tau2):
        return A1/tau1*np.exp(-t/tau1)+A2/tau2*np.exp(-t/tau2)
    model = MLEModel(twoexp)
    params = model.make_params()
    params["A1"].set(1, min=0)
    params["tau1"].set(1e-3, min=10e-6)
    params["A2"].set(1, min=0)
    params["tau2"].set(1e-3, min=2e-3)

    result = model.fit(all_states_en, t=all_states_t, params=params)
    result.plot()
    print(result.fit_report())
   
    plt.xlabel('Time since anode switch [s]')
    plt.title('Cumulative Sum')
    plt.show()

elif plot==7:
    plt.figure()
    t_ranges = [[.479,.488],[.979,.988],[.979,.988],[.939,.988]]
    e_range = [788,808]
    t_binsize = 0.0001
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

    t_bin_edges = np.arange(0,.025,t_binsize)
    e_bin_edges = np.arange(e_range[0],e_range[1],e_binsize)

    counts,_,_ = np.histogram2d(all_states_en,all_states_t,bins = [e_bin_edges,t_bin_edges])
    counts = np.sum(counts,axis=0)

    def twoexp(t, A1, tau1, A2, tau2):
        return A1/tau1*np.exp(-t/tau1)+A2/tau2*np.exp(-t/tau2)
    model = MLEModel(twoexp)
    params = model.make_params()
    params["A1"].set(1, min=0)
    params["tau1"].set(1e-3, min=10e-6)
    params["A2"].set(1, min=0)
    params["tau2"].set(1e-3, min=2e-3)

    result = model.fit(counts, t=utils.midpoints(t_bin_edges), params=params)
    result.plot()
    #print(result.params['tau1'])
    print(result.fit_report())
   
    plt.xlabel('Time since anode switch [s]')
    plt.title('Cumulative Sum')
    plt.show()