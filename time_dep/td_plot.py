import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
from scipy.stats import binned_statistic_2d, expon, norm, poisson
from scipy.special import voigt_profile
from statsmodels.base.model import GenericLikelihoodModel

plt.rcParams.update({'font.size': 16})

dir = '/home/tim/research/tes/td_data/'
#file = '20221220_'
file = '20221220_'
states = ['0000_H','0000_J','0000_L']
#states = ['0002_T']
#states = ['0002_M']

plot_type = 2
# 0: 2d beam off
# 1: e2/m3 slice
# 2: m3 slice fitted
# 3: 2d
# 4: build up fitted
# 5: Ni/Co compare
# 6: slice fft
# 7: E2/M3 slice lineshift
# 8: 2d binned
# 9: 1d remove buildup
# 10: per time slice fit Ni/Cu
# 11: Cu buildup with residual from 10
# 12:
# 13:
# 14: 1d time integrated

data = np.empty((1,2))
for state in states:
    data = np.vstack((data,np.load(dir+file+state+'.npy')))

if plot_type == 0:
    e_bounds = [600,1000]
    t_bounds = [.975,1.000]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    plt.xlabel('Time since trigger [ms]')
    plt.ylabel('Photon energy [eV]')
    plt.scatter(data[:,1],data[:,0],marker='.')
    plt.show()

elif plot_type == 1:

    binsize = .001

    e_bounds = [830,860]
    e_bounds = [837,847]
    t_bounds = [.979,.989]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    print(len(data))

    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
    counts, _ = np.histogram(data, bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    plt.xlabel('Time since trigger [ms]')
    plt.ylabel(f'Counts ({e_bounds} eV)')
    plt.scatter(bin_centers,counts)
    plt.show()

elif plot_type == 2:

    binsize = .001

    e_bounds = [830,860]
    e_bounds = [837,847]
    t_bounds = [.979,.989]
    #t_bounds = [.975,1]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    print(len(data))

    def decay_func(t,A,B,tau):
        return A*np.exp(-(t-B)/tau)

    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
    counts, _ = np.histogram(data[:,1], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    # remove zero count bins and scale t to ms
    bin_centers = bin_centers[counts>0]*1000
    counts = counts[counts>0]

    weights = [(count**(1/2)) for count in counts]
    guess = [16,979,3]
    popt,pcov = curve_fit(decay_func,bin_centers,counts,p0=guess,sigma=weights)
    uncertainties = np.sqrt(np.diag(pcov))
    
    #plt.plot(bin_centers,decay_func(bin_centers,guess[0],guess[1],guess[2]))
    plt.scatter(bin_centers,counts,zorder=0)
    plt.errorbar(bin_centers,counts,weights,ls='none',zorder=0)
    plt.plot(bin_centers,decay_func(bin_centers,popt[0],popt[1],popt[2]),label=r' $Ae^{-(t-t_0)/ \tau}$'+r'   $\tau$'+f'={popt[2]:.2f}'+r'$\pm$'+f'{uncertainties[2]:.2f}',zorder=10)

    plt.title('Nd')
    plt.xlabel('Time since trigger [ms]')
    plt.ylabel(f'Counts ({e_bounds} eV)')
    plt.minorticks_on()
    plt.legend()
    plt.show()

elif plot_type == 3:
    e_bounds = [0,2000]
    t_bounds = [0,1.000]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    plt.xlabel('Time since trigger [ms]')
    plt.ylabel('Photon energy [eV]')
    plt.scatter(data[:,1],data[:,0],marker='.',s=10)
    #plt.tight_layout()
    plt.minorticks_on()
    plt.show()

elif plot_type == 4:

    binsize = .001

    e_bounds0 = [837,847]
    e_bounds1 = [1199,1209]
    t_bounds = [.02,.1]

    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]
    data0 = data[(data[:,0]>e_bounds0[0]) & (data[:,0]<e_bounds0[1])]
    data1 = data[(data[:,0]>e_bounds1[0]) & (data[:,0]<e_bounds1[1])]

    print(len(data0),len(data1))

    def decay_func(t,A,B,tau):
        return A*(1-np.exp(-(t-B)/tau))
    
    def build_func(t,A,B,tau,M):
        return M/(1+(A*np.exp(-(t-B)*tau*M)))

    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
    counts0, _ = np.histogram(data0[:,1], bins=bin_edges)
    counts1, _ = np.histogram(data1[:,1], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    bin_centers = bin_centers[(counts0>0) & (counts1>0)]*1000
    counts0 = counts0[counts0>0]
    counts1 = counts1[counts1>0]

    weights0 = [(count**(1/2)) for count in counts0]
    guess0 = [150,20,.001,150]
    #popt0,pcov0 = curve_fit(decay_func,bin_centers,counts0,p0=guess0,sigma=weights0)
    popt0,pcov0 = curve_fit(build_func,bin_centers,counts0,p0=guess0,sigma=weights0)
    uncertainties0 = np.sqrt(np.diag(pcov0))

    weights1 = [(count**(1/2)) for count in counts1]
    guess1 = [600,12,.0006,550]
    #popt1,pcov1 = curve_fit(decay_func,bin_centers,counts1,p0=guess1,sigma=weights1)
    popt1,pcov1 = curve_fit(build_func,bin_centers,counts1,p0=guess1,sigma=weights1)
    uncertainties1 = np.sqrt(np.diag(pcov1))

    plt.xlabel('Time since trigger [ms]')
    plt.ylabel(f'Counts)')
    plt.errorbar(bin_centers,counts0,weights0,ls='none',color='g')
    plt.scatter(bin_centers,counts0,label=f'{e_bounds0}',color='g')
    plt.errorbar(bin_centers,counts1,weights1,ls='none',color='b')
    plt.scatter(bin_centers,counts1,label=f'{e_bounds1}',color='b')
    #plt.plot(bin_centers,build_func(bin_centers,guess0[0],guess0[1],guess0[2],guess0[3]),label='guess0')
    #plt.plot(bin_centers,build_func(bin_centers,guess1[0],guess1[1],guess1[2],guess1[3]),label='guess1')
    #plt.plot(bin_centers,build_func(bin_centers,popt0[0],popt0[1],popt0[2],popt0[3]),label=r' $M/(1+Ae^{-r(t-t_0)M})$'+r'   $r$'+f'={popt0[2]:.2E}'+r'$\pm$'+f'{uncertainties0[2]:.2E}',color='g')
    #plt.plot(bin_centers,build_func(bin_centers,popt1[0],popt1[1],popt1[2],popt1[3]),label=r' $M/(1+Ae^{-r(t-t_0)M})$'+r'   $r$'+f'={popt1[2]:.2E}'+r'$\pm$'+f'{uncertainties1[2]:.2E}',color='b')
    #plt.ylim([-5,800])
    plt.minorticks_on()
    plt.legend()
    plt.show()

elif plot_type == 5:

    binsize = .002

    #e_bounds0 = [1199,1209]
    #e_bounds1 = [1236,1246]
    e_bounds0 = [1140,1146]
    e_bounds1 = [1182,1188]
    t_bounds = [0,3]

    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]
    data0 = data[(data[:,0]>e_bounds0[0]) & (data[:,0]<e_bounds0[1])]
    data1 = data[(data[:,0]>e_bounds1[0]) & (data[:,0]<e_bounds1[1])]

    print(len(data0),len(data1))

    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
    counts0, _ = np.histogram(data0[:,1], bins=bin_edges)
    counts1, _ = np.histogram(data1[:,1], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    out = np.stack((bin_centers*1e3,counts0,counts1), axis=-1)

    bin_centers0 = bin_centers[(counts0>0)]*1000
    bin_centers1 = bin_centers[(counts1>0)]*1000
    counts0 = counts0[counts0>0]
    counts1 = counts1[counts1>0]

    weights0 = [(count**(1/2)) for count in counts0]
    weights1 = [(count**(1/2)) for count in counts1]

    plt.errorbar(bin_centers0,counts0,weights0,ls='none',color='b')
    plt.scatter(bin_centers0,counts0,label=f'{e_bounds0} Ni',color='b')
    plt.errorbar(bin_centers1,counts1,weights1,ls='none',color='y')
    plt.scatter(bin_centers1,counts1,label=f'{e_bounds1} Co',color='y')
    
    plt.xlabel('Time since trigger [ms]')
    plt.ylabel(f'Counts')
    plt.minorticks_on()
    plt.legend()
    plt.show()
    
    np.savetxt(dir+f'Pr_20221221_CoNi_K_{binsize*1000}ms.csv', out, fmt='%.1f', delimiter=',')

elif plot_type == 6:

    binsize = .001

    e_bounds = [1199,1209]
    t_bounds = [.1,.95]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    print(len(data))

    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
    counts, _ = np.histogram(data[:,1], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    N = len(counts)
    T = binsize
    xf = fftfreq(N,T)[1:N//2]
    plt.plot(xf,2/N*np.abs(fft(counts)[1:N//2]))
    plt.xlabel('Freq [Hz]')
    plt.ylabel('Magnitude (arb)')
    plt.minorticks_on()
    plt.show()

elif plot_type == 7:

    binsize = 1

    e_bounds = [837,847]
    t_bounds0 = [.8,.9]
    t_bounds1 = [.979,.989]

    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data0 = data[(data[:,1]>t_bounds0[0]) & (data[:,1]<t_bounds0[1])]
    data1 = data[(data[:,1]>t_bounds1[0]) & (data[:,1]<t_bounds1[1])]

    bin_edges = np.arange(e_bounds[0], e_bounds[1]+binsize, binsize)
    counts0, _ = np.histogram(data0[:,0], bins=bin_edges)
    counts1, _ = np.histogram(data1[:,0], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2


    bin_centers0 = bin_centers[(counts0>0)]
    bin_centers1 = bin_centers[(counts1>0)]
    counts0 = counts0[counts0>0]
    counts1 = counts1[counts1>0]

    weights0 = [(count**(1/2)) for count in counts0]
    weights1 = [(count**(1/2)) for count in counts1]

    counts1 = counts1/np.max(counts1)
    counts0 = counts0/np.max(counts0)

    plt.plot(bin_centers0,counts0,ls='none',color='b')
    plt.plot(bin_centers0,counts0,label=f'{t_bounds0}',color='b')
    plt.plot(bin_centers1,counts1,ls='none',color='y')
    plt.plot(bin_centers1,counts1,label=f'{t_bounds1}',color='y')
    
    plt.xlabel('Energy [eV]')
    plt.ylabel(f'Counts')
    plt.minorticks_on()
    plt.legend()
    plt.show()

elif plot_type == 8:
    e_binsize = 1
    t_binsize = .001 

    e_bounds = [500,5000]
    t_bounds = [0,1.000]
    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    x_bins = np.arange(np.min(data[:,0]),np.max(data[:,0]+e_binsize),e_binsize)
    y_bins = np.arange(np.min(data[:,1]),np.max(data[:,1]+t_binsize),t_binsize)
    bin_data,x_edges,y_edges,_ = binned_statistic_2d(data[:,0], data[:,1], None, statistic='count', bins=[x_bins,y_bins])

    plt.pcolormesh(y_edges,x_edges,bin_data)
    plt.xlabel('Time since trigger [s]')
    plt.ylabel('Photon energy [eV]')
    plt.title(f'{file}{states}')
    #plt.tight_layout()
    plt.colorbar()
    plt.minorticks_on()
    plt.show()

elif plot_type == 9:
    e_binsize = 1
    t_binsize = .001 

    e_bounds = [600,2000]
    t_bounds_i = [0,0.97]
    t_bounds = [0.05,0.97]

    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data_uncut = data[(data[:,1]>t_bounds_i[0]) & (data[:,1]<t_bounds_i[1])]
    data_cut = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    x_bins = np.arange(np.min(data[:,0]),np.max(data[:,0]+e_binsize),e_binsize)
    y_bins = np.arange(np.min(data[:,1]),np.max(data[:,1]+t_binsize),t_binsize)
    bin_data_cut,x_edges,y_edges,_ = binned_statistic_2d(data_cut[:,0], data_cut[:,1], None, statistic='count', bins=[x_bins,y_bins])
    bin_data_uncut,x_edges,y_edges,_ = binned_statistic_2d(data_uncut[:,0], data_uncut[:,1], None, statistic='count', bins=[x_bins,y_bins])
    
    plt.plot(x_edges[:-1],np.sum(bin_data_uncut, axis=1),label='Full')
    plt.plot(x_edges[:-1],np.sum(bin_data_cut, axis=1),label='Cut')
    plt.ylabel(f'Counts per {t_binsize} s bin')
    plt.xlabel('Photon energy [eV]')
    #plt.tight_layout()
    plt.minorticks_on()
    plt.legend()
    plt.show()

elif plot_type == 10:
    e_binsize = 1
    t_binsize = .001 

    e_bounds = [1190,1220]
    t_bounds = [0.027,.97]

    def gaussian(x, sigma, center, A):
     out = voigt_profile(x-center, sigma, 0)
     return (A*out)/np.max(out)

    def multi_gaussian(x, *params):
        # params: [sigma, center, spacing, A0, A1]
        y = gaussian(x, params[0], params[1], params[3])
        y = y + gaussian(x, params[0], params[1]+params[2], params[4])
        return y

    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    x_bins = np.arange(np.min(data[:,0]),np.max(data[:,0]+e_binsize),e_binsize)
    y_bins = np.arange(np.min(data[:,1]),np.max(data[:,1]+t_binsize),t_binsize)
    bin_data,x_edges,y_edges,_ = binned_statistic_2d(data[:,0], data[:,1], None, statistic='count', bins=[x_bins,y_bins])

    bin_centers = x_edges[:-1]+e_binsize/2
    interpx = np.linspace(e_bounds[0],e_bounds[1],1000)
    results = np.zeros((len(y_edges[:-1]),2))
    for i in range(len(y_edges[:-1])):
        data_slice = bin_data[:,i]
        guess = [2.5,1204,-6,0,0]
        popt, pcov = curve_fit(multi_gaussian, bin_centers, data_slice, p0=guess, maxfev=5000)
        results[i,0] = popt[1]
        results[i,1] = popt[1]+popt[2]
        #plt.plot(interpx, multi_gaussian(interpx, *popt))
        #plt.plot(bin_centers,data_slice)
        #plt.show()

    plt.plot(results[:,0])
    plt.plot(results[:,1])
    plt.minorticks_on()
    plt.show()

elif plot_type == 11:

    e_binsize = 1
    t_binsize = .001 

    def gaussian(x, sigma, center, A):
     out = voigt_profile(x-center, sigma, 0)
     return (A*out)/np.max(out)

    def multi_gaussian(x, *params):
        # params: [sigma, center, spacing, A0, A1]
        y = gaussian(x, params[0], params[1], params[3])
        y = y + gaussian(x, params[0], params[1]+params[2], params[4])
        return y
    
    def Cu_line(data):
        e_bounds = [1190,1220]
        t_bounds = [0.027,.97]
        t_bounds = [0.027,.5]

        data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
        data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

        x_bins = np.arange(np.min(data[:,0]),np.max(data[:,0]+e_binsize),e_binsize)
        y_bins = np.arange(np.min(data[:,1]),np.max(data[:,1]+t_binsize),t_binsize)
        bin_data,x_edges,y_edges,_ = binned_statistic_2d(data[:,0], data[:,1], None, statistic='count', bins=[x_bins,y_bins])

        bin_centers = x_edges[:-1]+e_binsize/2
        interpx = np.linspace(e_bounds[0],e_bounds[1],1000)
        results = np.zeros((len(y_edges[:-1]),2))
        for i in range(len(y_edges[:-1])):
            data_slice = bin_data[:,i]
            guess = [2.5,1204,-6,0,0]
            popt, pcov = curve_fit(multi_gaussian, bin_centers, data_slice, p0=guess, maxfev=5000)
            # popt[4] = 0
            # resid =  data_slice - multi_gaussian(bin_centers, *popt)
            # results[i,0] = np.sum(resid)
            results[i,0] = popt[4]
            results[i,1] = popt[0]
            # plt.plot(interpx, multi_gaussian(interpx, *popt))
            # plt.plot(bin_centers,resid)
            # plt.plot(bin_centers,data_slice)
            # plt.show()
        return y_edges[:-1], results


    binsize = .001
    e_bounds0 = [1199,1209]
    e_bounds1 = [1236,1246]
    #e_bounds1 = [1195.5,1197]
    #e_bounds1 = [838,846]
    t_bounds = [.02,.5]

    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]
    data0 = data[(data[:,0]>e_bounds0[0]) & (data[:,0]<e_bounds0[1])]
    data1 = data[(data[:,0]>e_bounds1[0]) & (data[:,0]<e_bounds1[1])]

    print(len(data0),len(data1))

    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
    counts0, _ = np.histogram(data0[:,1], bins=bin_edges)
    counts1, _ = np.histogram(data1[:,1], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    bin_centers0 = bin_centers[(counts0>0)]*1000
    bin_centers1 = bin_centers[(counts1>0)]*1000
    counts0 = counts0[counts0>0]
    counts1 = counts1[counts1>0]

    weights0 = [(count**(1/2)) for count in counts0]
    weights1 = [(count**(1/2)) for count in counts1]

    plt.errorbar(bin_centers0,counts0,weights0,ls='none',color='b')
    plt.scatter(bin_centers0,counts0,label=f'{e_bounds0} Ni',color='b')
    plt.errorbar(bin_centers1,counts1,weights1,ls='none',color='y')
    plt.scatter(bin_centers1,counts1,label=f'{e_bounds1} Co',color='y')
    
    centers, results = Cu_line(data)
    weights2 = [(count**(1/2)) for count in results[:,0]]
    cu = [amp*(2*3.1415*sigma**2)**.5 for amp, sigma in results]
    print(len(cu),len(centers))
    #plt.plot(y_edges[:-1]*1e3,results)
    plt.errorbar(centers*1e3,cu,weights2,ls='none',color='g')
    plt.scatter(centers*1e3,cu,label='[1190,1220] Cu',color='g')
    plt.xlabel('Time since trigger [ms]')
    plt.ylabel(f'Counts')
    plt.minorticks_on()
    plt.legend()
    plt.show()

elif plot_type == 12:
    e_binsize = .5
    t_binsize = .005

    e_bounds = [838,848]
    t_bounds = [0.05,.97]

    def gaussian(x, sigma, center, A):
     out = voigt_profile(x-center, sigma, 0)
     return (A*out)/np.max(out)

    def multi_gaussian(x, *params):
        # params: [sigma, center, spacing, A0, A1]
        y = gaussian(x, params[0], params[1], params[3])
        y = y + gaussian(x, params[0], params[1]+params[2], params[4])
        return y

    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    x_bins = np.arange(np.min(data[:,0]),np.max(data[:,0]+e_binsize),e_binsize)
    y_bins = np.arange(np.min(data[:,1]),np.max(data[:,1]+t_binsize),t_binsize)
    bin_data,x_edges,y_edges,_ = binned_statistic_2d(data[:,0], data[:,1], None, statistic='count', bins=[x_bins,y_bins])

    bin_centers = x_edges[:-1]+e_binsize/2
    interpx = np.linspace(e_bounds[0],e_bounds[1],1000)
    results = np.zeros((len(y_edges[:-1]),2))
    for i in range(len(y_edges[:-1])):
        data_slice = bin_data[:,i]
        guess = [2.5,840.7,1.2,0,0]
        bounds = ([1,-np.inf,1,0,0],[5,np.inf,2,np.inf,np.inf])
        try:
            popt, pcov = curve_fit(multi_gaussian, bin_centers, data_slice, p0=guess, bounds=bounds, maxfev=5000)
        except:
            popt = [0,0,0,0,0]
        results[i,0] = popt[1]
        results[i,1] = popt[1]+popt[2]
        # plt.plot(interpx, multi_gaussian(interpx, *popt))
        # plt.plot(bin_centers,data_slice)
        # plt.show()

    plt.scatter(y_edges[:-1],results[:,0])
    plt.scatter(y_edges[:-1],results[:,1])
    plt.minorticks_on()
    plt.ylim([838,846])
    plt.show()

elif plot_type == 13:

    binsize = .0005

    e_bounds = [837,847]
    t_bounds = [.979,.989]

    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    def decay_func(t,A,B,tau):
        return A*np.exp(-(t-B)/tau)


    bin_edges = np.arange(t_bounds[0], t_bounds[1]+binsize, binsize)
    counts, _ = np.histogram(data[:,1], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    data = data[:,1]
    params = expon.fit(data)
    x0,tau = params

    x = np.linspace(np.min(bin_centers),np.max(bin_centers),1000)

    
    disp = [(count-decay_func(bin_center,1,x0,tau)) for bin_center,count in zip(bin_centers,counts)]
    unc = norm.fit(disp)[0]/2e3
    counts = counts/np.max(counts)
    print(unc)
    
    plt.scatter(bin_centers,counts/np.max(counts),zorder=0)
    plt.plot(x,decay_func(x,1,x0,tau),label=r' $e^{-(t-t_0)/ \tau}$'+r'   $\tau$'+f'={tau*1000:.2f}'+' ms',zorder=10)
    plt.plot(x,decay_func(x,1,x0,tau-unc),color='k')
    plt.plot(x,decay_func(x,1,x0,tau+unc),color='k')

    plt.title('Nd')
    plt.xlabel('Time since trigger [ms]')
    plt.ylabel(f'Counts ({e_bounds} eV) norm')
    plt.minorticks_on()
    plt.legend()
    plt.show()


elif plot_type == 14:

    binsize = 1

    e_bounds = [500,2000]
    t_bounds = [0,1]

    data = data[(data[:,0]>e_bounds[0]) & (data[:,0]<e_bounds[1])]
    data = data[(data[:,1]>t_bounds[0]) & (data[:,1]<t_bounds[1])]

    bin_edges = np.arange(e_bounds[0], e_bounds[1]+binsize, binsize)
    counts, _ = np.histogram(data[:,0], bins=bin_edges)
    bin_centers = bin_edges[:-1]+binsize/2

    plt.plot(bin_centers,counts)
    plt.xlabel('Photon Energy [eV]')
    plt.ylabel(f'Counts ({e_bounds} eV)')
    plt.minorticks_on()
    plt.legend()
    plt.show()