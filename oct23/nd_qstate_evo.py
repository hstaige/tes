import numpy as np 
import matplotlib.pyplot as plt 
import my_utils as utils
from lmfit import Model

file = '/home/tim/research/EBIT-TES-Data/data_by_state/'
run = '20231014_0006'
states = ['I','K']

plot_type = 4

# 0: 2d
# 1: 1d, energy on x, Cu/Zn time window
# 2: 1d, time on x
# 3: 1d, energy on x, Cu/Zn time window, with fit
# 4: 1d, energy on x, Cu/Zn time window compare to SS

slice_half_width = 3
slices = [1203,1196,1186]
labels = ['Ni','Cu','Zn']

data_arr = np.empty((3,0))
for state in states:
    data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))

if plot_type==0:
    t_bin_edges = np.arange(0,1.003,0.003)
    e_bin_edges = np.arange(750,2000,1)

    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    plt.pcolormesh(t_bin_edges,e_bin_edges,counts)

if plot_type==1:
    e_binsize = 1
    t_bin_edges = np.arange(0.02,0.05,0.001)
    e_bin_edges = np.arange(1150,1250,e_binsize)

    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    plt.plot(utils.midpoints(e_bin_edges),np.sum(counts,axis=1))
    plt.xlabel('Energy [eV]')
    plt.ylabel(f'Counts per {e_binsize} eV bin')

if plot_type==2:
    e_binsize = 1
    slice_half_width = 3
    slices = [1203,1196,1186]
    labels = ['Ni','Cu','Zn']

    t_bin_edges = np.arange(0,0.1,0.001)
    for slice,label in zip(slices,labels):
        e_bin_edges = np.arange(slice-slice_half_width,slice+slice_half_width,e_binsize)
        counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
        plt.plot(utils.midpoints(t_bin_edges),np.sum(counts,axis=0),label=label)

    plt.xlabel('Time since trigger [s]')
    plt.ylabel(f'Counts per {e_binsize} eV bin')
    plt.legend()

if plot_type==3:
    e_binsize = 1
    t_bin_edges = np.arange(0.02,0.04,0.001)
    e_bin_edges = np.arange(1180,1210,e_binsize)
    center_guess = [1203,1196,1186]
    center_var = 5

    counts,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    e_mids = utils.midpoints(e_bin_edges)
    e_counts = np.sum(counts,axis=1)

    def gaussian(x,A,sigma,center):
        return A*np.exp(-(x-center)**2/(2*sigma**2))
    
    G1,G2,G3 = Model(gaussian, prefix='Ni_'), Model(gaussian, prefix='Cu_'), Model(gaussian, prefix='Zn_')
    G_tot = G1+G2+G3
    params_tot = G_tot.make_params(Ni_A={'value':800,'min':0}, Ni_sigma={'value':4,'min':0}, Ni_center={'value':center_guess[0],'min':center_guess[0]-center_var,'max':center_guess[0]+center_var},
                                    Cu_A={'value':250,'min':0}, Cu_sigma={'value':4,'min':0}, Cu_center={'value':center_guess[1],'min':center_guess[1]-center_var,'max':center_guess[1]+center_var},
                                    Zn_A={'value':250,'min':0}, Zn_sigma={'value':4,'min':0}, Zn_center={'value':center_guess[2],'min':center_guess[2]-center_var,'max':center_guess[2]+center_var})
    
    out = G_tot.fit(e_counts, params=params_tot, x=e_mids)
    print(out.fit_report())
    out.plot_fit()
    plt.plot(e_mids,e_counts)
    plt.xlabel('Energy [eV]')
    plt.ylabel(f'Counts per {e_binsize} eV bin')

if plot_type==4:
    e_binsize = 3
    e_bin_edges = np.arange(500,2000,e_binsize)
    t_bin_edges = np.arange(0.02,0.04,0.001)
    counts1,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])

    t_bin_edges = np.arange(0.42,0.44,0.001)
    counts2,_,_ = np.histogram2d(data_arr[0,:],data_arr[2,:],bins = [e_bin_edges,t_bin_edges])
    plt.plot(utils.midpoints(e_bin_edges),np.sum(counts1,axis=1)-np.sum(counts2,axis=1))
    plt.ylim([0,400])
    plt.xlabel('Energy [eV]')
    plt.ylabel(f'Counts per {e_binsize} eV bin')


plt.show()