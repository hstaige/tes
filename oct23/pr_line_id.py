import os
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter
import my_utils as utils
import lmfit

file = '/home/tim/research/EBIT-TES-Data/data_by_state/'
run = '20231015_0000'
states = ['G','H','I']

plot_type = 2
# 0: determine best th match (dumb)
# 1: etermine best th match (better)
# 2: peak detection

if plot_type==0:
    e_binsize = 0.2
    t_bin_edges = np.arange(0,1,0.001)
    e_bin_edges = np.arange(500-e_binsize/2,2200+e_binsize,e_binsize)
    data_arr = np.empty((3,0))
    for state in states:
        data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))
    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    counts = np.sum(counts,axis=1)
    counts /= np.max(counts)
    energies = utils.midpoints(e_bin_edges)

    def resids(a, data, theory):
        return(data-a*theory)
    params = lmfit.create_params(a=1)

    th_dir = '/home/tim/research/dec22_data/theory/'
    th_subdirs = ['Pr_1e11','Pr_1e13','Pr_3e12']
    min_results = []
    for th_subdir in th_subdirs:
        full_dir = th_dir+th_subdir
        files = os.listdir(full_dir)
        for file in files:
            th = np.loadtxt(f'{th_dir}{th_subdir}/{file}')
            th[:,1] /= np.max(th[:,1])
            #print(th[:,0]-energies)
            result = lmfit.minimize(resids,params,args=(counts,th[:,1]))
            plt.plot(th[:,0],result.params['a']*th[:,1],label=f'{th_subdir} {file}')
            min_results.append([f'{th_subdir} {file}', np.sum(result.residual)])

    min_results = np.array(min_results)
    min_results = min_results[np.argsort(np.float32(min_results[:,1]))]
    print(min_results)
    plt.plot(energies,counts,color='k')
    plt.legend()
    plt.show()

if plot_type==1:
    e_binsize = 0.2
    t_bin_edges = np.arange(0,1,0.001)
    e_bin_edges = np.arange(500-e_binsize/2,2200+e_binsize,e_binsize)
    data_arr = np.empty((3,0))
    for state in states:
        data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))
    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    counts = np.sum(counts,axis=1)
    counts /= np.max(counts)
    energies = utils.midpoints(e_bin_edges)

    def resids(a, data, theory):
        return(data-a*theory)
    params = lmfit.create_params(a=1)

    th_dir = '/home/tim/research/dec22_data/theory/Pr_2keV/'
    min_results = []
    files = os.listdir(th_dir)
    for file in files:
        th = np.loadtxt(f'{th_dir}{file}')
        th[:,1] /= np.max(th[:,1])
        #print(th[:,0]-energies)
        result = lmfit.minimize(resids,params,args=(counts,th[:,1]))
        plt.plot(th[:,0],result.params['a']*th[:,1],label=f'{file}')
        min_results.append([f'{file}', np.sum(result.residual)])

    min_results = np.array(min_results)
    min_results = min_results[np.argsort(np.float32(min_results[:,1]))]
    print(min_results)
    plt.plot(energies,counts,color='k')
    plt.legend()
    plt.show()

if plot_type==2:
    data_arr = np.empty((3,0))
    for state in states:
        data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))

    e_binsize = .5
    t_bin_edges = np.arange(0,1,0.001)
    e_bin_edges = np.arange(500,2000,e_binsize)

    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    counts = np.sum(counts,axis=1)
    energies = utils.midpoints(e_bin_edges)
    plt.plot(energies,counts,label='data')

    counts = savgol_filter(counts,window_length=10,polyorder=9)

    # peaks,_ = find_peaks(counts, prominence=200)
    # for peak in peaks:
    #     plt.axvline(energies[peak],color='grey',linestyle='--')
    filt_counts = gaussian_filter(counts,sigma=1)
    sndder = -np.gradient(np.gradient(counts))
    sndder *= np.max(counts)/np.max(sndder)
    sndder_filt = savgol_filter(sndder,window_length=10,polyorder=6)

    filt_counts = gaussian_filter(counts,sigma=.1)
    fthder = np.gradient(np.gradient(np.gradient(np.gradient(filt_counts))))
    fthder *= np.max(counts)/np.max(fthder)

    plt.plot(energies,fthder,label='4th der')
    plt.plot(energies,sndder_filt,label='2th derivative savgol filt')
    plt.plot(energies,filt_counts,label='gauss filt')
    plt.plot(energies,counts,label='sav filt')
    #plt.ylim([0,np.max(counts)])
    plt.xlabel('Energy [eV]')
    plt.ylabel(f'Counts per {e_binsize} eV bin')
    plt.legend()
    plt.show()