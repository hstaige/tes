import os
import math
import time
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, deconvolve, wiener
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
import my_utils as utils
from lmfit import Parameters, minimize, fit_report, Model, create_params

plt.rcParams.update({'font.size': 16})


file = '/home/tim/research/EBIT-TES-Data/data_by_state/'
# run = '20231015_0000'
# states = ['G','H','I']
run = '20231015_0000'
states = ['G','H','I']

plot_type = 1
# 0: determine best th match (dumb)
# 1: determine best th match (better)
# 2: determine th corrections
# 3: peak detection
# 5: weiner deconv
# 6: ideal deconv
# 7: BIC fit

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
    fig, ax = plt.subplots(figsize=(8.09*2,5*2))
    e_binsize = 0.5
    e_bin_edges = np.arange(750, 1800, e_binsize)
    data_arr = np.empty((3,0))
    for state in states:
        data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))

    counts,_ = np.histogram(data_arr[0,:],bins=e_bin_edges)
    counts = counts / np.max(counts)
    energies = utils.midpoints(e_bin_edges)
    

    def resids(params, energies, counts, th_spline):
        amp = params['amp']
        A = params['A']
        B = params['B']
        energies = A*energies+B
        return(counts-amp*th_spline(energies))
    
    th_dir = '/home/tim/research/dec22_data/theory/Pr_2keV/'
    min_results = []
    files = os.listdir(th_dir)
    for file in files:
        th = np.loadtxt(f'{th_dir}{file}')
        th[:,1] /= np.max(th[:,1])
        th_spline = CubicSpline(th[:,0], th[:,1])
        params = Parameters()
        param_list = [('amp', 1, False, 0, None, None, None),
                      ('A', 1, True, None, None, None, None),
                      ('B', 0, True, None, None, None, None)]
        for p in param_list:
            params.add_many(p)
        
        result = minimize(resids, params, args=(energies,counts,th_spline))
        print(fit_report(result))
        plt.plot(energies*result.params['A']+result.params['B'], result.params['amp']*th_spline(energies),label=f'{file}')
        min_results.append([f'{file}', np.sum(result.residual)])

    min_results = np.array(min_results)
    min_results = min_results[np.argsort(np.float32(min_results[:,1]))]
    print(min_results)
    plt.plot(energies,counts,color='k')
    plt.xlabel('Energy [eV]')
    plt.ylabel('Norm Intensity')
    plt.legend()
    plt.tight_layout()
    plt.minorticks_on()
    plt.show()

if plot_type==2:
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
    data_spline = CubicSpline(energies, counts)

    def resids(params, *args):
        data, theory, th_en, energies = args
        amp = params['amp']
        A = params['A']
        B = params['B']
        C = params['C']
        D = params['D']
        new_th_en = A*th_en**2 + B*th_en + C
        th_spline = CubicSpline(new_th_en, theory)
        return(data-amp*th_spline(energies))
    params = lmfit.create_params(amp=1, A=0, B=1, C=0, D=0)

    file = '/home/tim/research/dec22_data/theory/Pr_1e11/conv1.4_SP1000.00_1.00e+11_0_0_2000.dat'
    th = np.loadtxt(file)
    th[:,1] /= np.max(th[:,1])
    result = lmfit.minimize(resids, params, args=([counts,th[:,1],th[:,0],energies]))

    th_en = th[:,0]**2*result.params['D']+th[:,0]**2*result.params['A']+th[:,0]*result.params['B']+result.params['C']
    print(result.params.pretty_print())
    plt.plot(th_en, result.params['amp']*th[:,1],label='Corrected Th')
    plt.plot(th[:,0],th[:,1],label='Th')
    plt.plot(energies,data_spline(energies),color='k',label='Data')
    plt.legend()
    plt.show()

if plot_type==3:
    data_arr = np.empty((3,0))
    for state in states:
        data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))

    e_binsize = .5
    e_bin_edges = np.arange(600,2000,e_binsize)
    e_bin_edges = np.arange(1600,1850,e_binsize)
    binned_counts,_ = np.histogram(data_arr[0,:],bins=e_bin_edges)
    energies = utils.midpoints(e_bin_edges)
    filtered_counts = savgol_filter(binned_counts,window_length=20,polyorder=3)
    #plt.plot(energies,filtered_counts,label='filt')
    sndder = -np.gradient(np.gradient(filtered_counts))
    sndder *= np.max(filtered_counts)/np.max(sndder)
    sndder_filt = savgol_filter(sndder,window_length=10,polyorder=3)
    plt.plot(energies,sndder_filt,label='d2dx2filter')

    peak_inds,_ = find_peaks(sndder_filt,prominence=300,height=100)
    [plt.axvline(energies[peak],color='grey') for peak in peak_inds]

    # e_binsize = .25
    # e_bin_edges = np.arange(500,2000,e_binsize)
    # binned_counts,_ = np.histogram(data_arr[0,:],bins=e_bin_edges)
    # energies = utils.midpoints(e_bin_edges)
    plt.plot(energies,binned_counts,label='data')

    x = energies
    sig = 2
    uncert = binned_counts/binned_counts**.5

    def gaussian(x, A, mu, sigma):
        return A * 1/(sigma*(2*math.pi)**.5) * np.exp(-(x-mu)**2/(2*sigma**2))

    x_len = len(x)
    param_len = len(peak_inds)
    print(param_len)
    
    s2p = math.sqrt(2*math.pi)
    def vect_gauss(x,A,mu,sigma,gamma):
        A_arr = np.tile(A,(x_len,1))
        mu_arr = np.tile(mu,(x_len,1))
        sigma_arr = np.tile(sigma,(x_len,1))
        gamma_arr = np.tile(gamma,(x_len,1))
        x_arr = np.tile(x,(param_len,1)).T
        #print(A_arr.shape,mu_arr.shape,sigma_arr.shape,x_arr.shape)
        gauss = 1/(sigma_arr*s2p)*np.exp(-(x_arr-mu_arr)**2/(2*sigma_arr**2))
        lorentz = 1/math.pi*(gamma_arr/2)/((x_arr-mu_arr)**2+(gamma_arr/2)**2)
        return np.sum(A_arr*(gauss+lorentz),axis=1)

    def resids(params,x,data,uncert):
        params_arr = np.reshape(params,(-1,4))
        return (data-vect_gauss(x,params_arr[:,0],params_arr[:,1],params_arr[:,2],params_arr[:,3]))/uncert

    params_tot = Parameters()
    param_A = [(f'P{x[i]*10:.0f}_A', binned_counts[i]*3, True, 0, None, None, None) for i in peak_inds]
    param_mu = [(f'P{x[i]*10:.0f}_mu', energies[i]+e_binsize, True, energies[i]-2, energies[i]+2, None, None) for i in peak_inds]
    param_sigma = [(f'P{x[i]*10:.0f}_sigma', 1.95, False, 0, None, None, None) for i in peak_inds]
    param_gamma = [(f'P{x[i]*10:.0f}_gamma', 5.37, False, 0, None, None, None) for i in peak_inds]
    #fwhm = 0.5346*gamma + (0.2166*(gamma**2)+sigma**2)**0.5 # doi.org/10.1016/0022-4073(77)90161-3

    for a,m,s,g in zip(param_A, param_mu, param_sigma,param_gamma):
        params_tot.add_many(a)
        params_tot.add_many(m)
        params_tot.add_many(s)
        params_tot.add_many(g)

    results = minimize(resids,params_tot,args=(x,binned_counts,uncert))
    for var in results.params.keys():
        print(results.params[var].value,results.params[var].stderr)
    
    fit_params = np.array(list(results.params.valuesdict().values()))
    fit_params= np.reshape(fit_params,(-1,4))
    plt.plot(x,vect_gauss(x,fit_params[:,0],fit_params[:,1],fit_params[:,2],fit_params[:,3]),label='fit')
    #print(fit_report(results))

    pre_params = np.array(list(params_tot.valuesdict().values()))
    pre_params= np.reshape(pre_params,(-1,4))
    #plt.plot(x,vect_gauss(x,pre_params[:,0],pre_params[:,1],pre_params[:,2],pre_params[:,3]),label='prefit')
    plt.plot(x,binned_counts-vect_gauss(x,fit_params[:,0],fit_params[:,1],fit_params[:,2],fit_params[:,3]),label='resids')
    plt.legend()
    #plt.show()

if plot_type==4:
    data_arr = np.empty((3,0))
    for state in states:
        data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))

    e_binsize = 1
    t_bin_edges = np.arange(0,1,0.001)
    e_bin_edges = np.arange(1610,1650,e_binsize)

    binned_counts,_ = np.histogram(data_arr[0,:],bins=e_bin_edges)
    energies = utils.midpoints(e_bin_edges)
    plt.plot(energies,binned_counts,label='data')

    counts = savgol_filter(binned_counts,window_length=10,polyorder=4)

    sndder = -np.gradient(np.gradient(counts))
    sndder *= np.max(counts)/np.max(sndder)
    sndder_filt = savgol_filter(sndder,window_length=10,polyorder=7)

    peak_inds,_ = find_peaks(sndder_filt+sndder_filt,prominence=10000)

    # plt.plot(energies+e_binsize,sndder_filt,label='2th derivative savgol filt')
    # plt.plot(energies+e_binsize/2,counts,label='sav filt')
    # plt.plot(energies+e_binsize,sndder_filt+sndder_filt)
    # #plt.ylim([0,np.max(counts)])
    for peak in peak_inds:
        plt.axvline(energies[peak]+e_binsize,color='grey')
    # plt.xlabel('Energy [eV]')
    # plt.ylabel(f'Counts per {e_binsize} eV bin')
    # plt.legend()

    x = energies
    sig = 2
    uncert = binned_counts**.5

    def gaussian(x, mu, sigma):
        r1 = (2*math.pi)**.5
        p1 = 1/(sigma * r1)
        p2 = np.exp(-(x-mu)**2/(2*sigma**2))
        return p1 * p2
    
    def lorentzian(x, mu, gamma):
        return 1/math.pi*(gamma/2)/((x-mu)**2+(gamma/2)**2)
    
    x_len = len(x)
    param_len = len(peak_inds)
    print(param_len)

    # A_arr = np.zeros((x_len,param_len))
    # mu_arr = np.zeros((x_len,param_len))
    # sigma_arr = np.zeros((x_len,param_len))
    # x_arr = np.zeros((param_len,x_len)).T

    def voigt(x,A1,A2,mu1,mu2,sigma,gamma):
        p1 = A1 * (gaussian(x,mu1,sigma)+lorentzian(x,mu1,gamma))
        p2 = A2 * (gaussian(x,mu2,sigma)+lorentzian(x,mu2,gamma))
        return p1+p2

    def resids(params,x,data,uncert):
        A1,A2,mu1,mu2,sigma,gamma = list(params.valuesdict().values())
        return (data-voigt(x,A1,A2,mu1,mu2,sigma,gamma))/uncert

    params_tot = Parameters()
    params_tot.add_many(('A1', binned_counts[peak_inds[0]], True, 0, None, None, None),
                        ('A2', binned_counts[peak_inds[1]], True, 0, None, None, None),
                        ('mu1', energies[peak_inds[0]], True, energies[peak_inds[0]]-3, energies[peak_inds[0]]+3, None, None),
                        ('mu2', energies[peak_inds[1]], True, energies[peak_inds[1]]-3, energies[peak_inds[1]]+3, None, None),
                        ('sigma', 2, True, 0, None, None, None),
                        ('gamma', 2, True, 0, None, None, None))

    print(list(params_tot.valuesdict().values()))
    t0 = time.time()
    results = minimize(resids,params_tot,args=(x,binned_counts,uncert))
    print(time.time()-t0)
    fit_params = np.array(list(results.params.valuesdict().values()))
    A1,A2,mu1,mu2,sigma,gamma = fit_params
    plt.plot(x,voigt(x,A1,A2,mu1,mu2,sigma,gamma))
    print(fit_report(results))
    plt.show()

if plot_type==5:

    data_arr = np.empty((3,0))
    for state in states:
        data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))

    e_binsize = .25
    e_bin_edges = np.arange(600,1850,e_binsize)

    filt_width = 2
    binned_counts,_ = np.histogram(data_arr[0,:],bins=e_bin_edges)
    filtered_counts = savgol_filter(binned_counts,window_length=20,polyorder=3)
    gfiltered_counts = gaussian_filter(binned_counts, filt_width/e_binsize)

    x = utils.midpoints(e_bin_edges)
    def gaussian(x, mu, sigma):
        r1 = (2*math.pi)**.5
        p1 = 1/(sigma * r1)
        p2 = np.exp(-(x-mu)**2/(2*sigma**2))
        return p1 * p2
    
    def lorentzian(x, mu, gamma):
        return 1/math.pi*(gamma/2)/((x-mu)**2+(gamma/2)**2)
    
    def voigt(x,A,mu,sigma,gamma):
        return A * (gaussian(x,mu,sigma)+lorentzian(x,mu,gamma))
    
    def wiener_deconvolution(signal, kernel, lambd):
        kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
        H = np.fft.fft(kernel)
        deconvolved = np.real(np.fft.ifft(np.fft.fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2)))
        return deconvolved
    
    def rolling_window(a, window):
        pad = np.ones(len(a.shape), dtype=np.int32)
        pad[-1] = window-1
        pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
        a = np.pad(a, pad,mode='reflect')
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    uncert = binned_counts**.5
    snr = binned_counts/np.var(rolling_window(binned_counts, 2), axis=-1)
    kern_x = np.arange(0,50,e_binsize)
    kern = voigt(kern_x,10,25,1.95,5.37)
    # plt.plot(kern_x,kern)
    # plt.show()
    # plt.figure()
    gkern = gaussian_filter(kern, filt_width/e_binsize)
    deconv = wiener_deconvolution(binned_counts,kern,.5)
    #deconv_g = wiener_deconvolution(gfiltered_counts,gkern,snr)
    deconv =  savgol_filter(deconv,window_length=20,polyorder=1)
    plt.plot(x+25, (deconv-np.min(deconv))/np.max(deconv-np.min(deconv)),label='deconv')
    #plt.plot(x+25,(deconv_g-np.min(deconv_g))/np.max(deconv_g-np.min(deconv_g)),label='gdeconv')
    plt.plot(x,binned_counts/np.max(binned_counts),label='data')
    plt.plot(x,filtered_counts/np.max(filtered_counts),label='filt')
    #plt.plot(x,gfiltered_counts/np.max(gfiltered_counts),label='gfilt')
    plt.legend()
    plt.show()

if plot_type==6:
    def gaussian(x, mu, sigma):
        r1 = (2*math.pi)**.5
        p1 = 1/(sigma * r1)
        p2 = np.exp(-(x-mu)**2/(2*sigma**2))
        return p1 * p2
    
    def lorentzian(x, mu, gamma):
        return 1/math.pi*(gamma/2)/((x-mu)**2+(gamma/2)**2)
    
    def voigt(x,A,mu,sigma,gamma):
        return A * (gaussian(x,mu,sigma)+lorentzian(x,mu,gamma))
    
    binsize = .5
    x = np.arange(1000,1600,binsize)

    binned_counts = voigt(x,10000,1200,1.95,5.37)+voigt(x,10000,1250,1.95,5.37)

    def wiener_deconvolution(signal, kernel, lambd):
        kernel = np.hstack((kernel, np.zeros(len(signal) - len(kernel)))) # zero pad the kernel to same length
        H = np.fft.fft(kernel)
        deconvolved = np.real(np.fft.ifft(np.fft.fft(signal)*np.conj(H)/(H*np.conj(H) + lambd**2)))
        return deconvolved
    
    def rolling_window(a, window):
        pad = np.ones(len(a.shape), dtype=np.int32)
        pad[-1] = window-1
        pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
        a = np.pad(a, pad,mode='reflect')
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    kern_x = np.arange(0,100,binsize)
    kern = voigt(kern_x,100,50,1.95,5.37)
    # plt.plot(kern_x,kern)
    # plt.show()
    # plt.figure()
    gauss = gaussian(kern_x,20,10)
    y = np.zeros(len(x))
    y[int(len(y)/2)] = 10
    binned_counts = np.convolve(y, kern, mode='same')+abs(np.random.normal(0,.5,len(y)))
    deconv = wiener_deconvolution(binned_counts,kern, binned_counts/np.var(rolling_window(binned_counts, 3), axis=-1))
    filt = savgol_filter(deconv,window_length=20,polyorder=4)
    #deconv,_ = deconvolve(binned_counts,gauss)

    plt.plot(x+50,deconv/np.max(deconv),label='deconv')
    plt.plot(x+50,filt/np.max(deconv),label='deconv_filt')
    #plt.plot(deconv)
    plt.plot(x,binned_counts/np.max(binned_counts),label='data')
    #plt.yscale('log')
    plt.legend()
    plt.show()

if plot_type==7:
    data_arr = np.empty((3,0))
    for state in states:
        data_arr = np.hstack((data_arr,np.load(f'{file}{run}_{state}.npy')))

    e_binsize = .25
    e_bin_edges = np.arange(600,1850,e_binsize)
    binned_counts,_ = np.histogram(data_arr[0,:],bins=e_bin_edges)
    energies = utils.midpoints(e_bin_edges)
    filtered_counts = savgol_filter(binned_counts,window_length=20,polyorder=3)

    peak_inds,_ = find_peaks(filtered_counts,prominence=500,height=100)
    #[plt.axvline(energies[peak],color='grey') for peak in peak_inds]
    x = energies
    x_len = len(x)
    uncert = binned_counts**.5

    rcs = 1e6
    to_run = True
    while to_run:
        
        param_len = len(peak_inds)
        print(param_len)
        
        s2p = math.sqrt(2*math.pi)
        def vect_gauss(x,A,mu,sigma,gamma):
            x_len = len(x)
            param_len = len(A)
            A_arr = np.tile(A,(x_len,1))
            mu_arr = np.tile(mu,(x_len,1))
            sigma_arr = np.tile(sigma,(x_len,1))
            gamma_arr = np.tile(gamma,(x_len,1))
            x_arr = np.tile(x,(param_len,1)).T
            #print(A_arr.shape,x_arr.shape)
            #print(A_arr.shape,mu_arr.shape,sigma_arr.shape,x_arr.shape)
            gauss = 1/(sigma_arr*s2p)*np.exp(-(x_arr-mu_arr)**2/(2*sigma_arr**2))
            lorentz = 1/math.pi*(gamma_arr/2)/((x_arr-mu_arr)**2+(gamma_arr/2)**2)
            return np.sum(A_arr*(gauss+lorentz),axis=1)

        def resids(params,x,data,uncert):
            params_arr = np.reshape(params,(-1,4))
            return (data-vect_gauss(x,params_arr[:,0],params_arr[:,1],params_arr[:,2],params_arr[:,3]))/uncert

        params_tot = Parameters()
        param_A = [(f'P{x[i]*10:.0f}_A', binned_counts[i]*3, True, 0, None, None, None) for i in peak_inds]
        param_mu = [(f'P{x[i]*10:.0f}_mu', energies[i]+e_binsize, True, energies[i]-5, energies[i]+5, None, None) for i in peak_inds]
        param_sigma = [(f'P{x[i]*10:.0f}_sigma', 1.95, False, 0, None, None, None) for i in peak_inds]
        param_gamma = [(f'P{x[i]*10:.0f}_gamma', 5.37, False, 0, None, None, None) for i in peak_inds]
        #fwhm = 0.5346*gamma + (0.2166*(gamma**2)+sigma**2)**0.5 # doi.org/10.1016/0022-4073(77)90161-3

        for a,m,s,g in zip(param_A, param_mu, param_sigma,param_gamma):
            params_tot.add_many(a)
            params_tot.add_many(m)
            params_tot.add_many(s)
            params_tot.add_many(g)

        results = minimize(resids,params_tot,args=(x,binned_counts,uncert))
        fit_params = np.array(list(results.params.valuesdict().values()))
        fit_params= np.reshape(fit_params,(-1,4))
        resids = binned_counts-vect_gauss(x,fit_params[:,0],fit_params[:,1],fit_params[:,2],fit_params[:,3])
        smooth_resids = savgol_filter(resids,window_length=20,polyorder=3)
        peak_inds = np.append(peak_inds,np.argmax(abs(smooth_resids)))
        prcs = rcs
        rcs = results.redchi
        print(rcs)
        if (rcs > 1) & (rcs<prcs):
            prev_results = results
        else:
            to_run = False
            results = prev_results

    fit_params = np.array(list(results.params.valuesdict().values()))
    fit_params= np.reshape(fit_params,(-1,4))
    plt.plot(x,vect_gauss(x,fit_params[:,0],fit_params[:,1],fit_params[:,2],fit_params[:,3]),label='fit')
    print(fit_report(results))
    pre_params = np.array(list(params_tot.valuesdict().values()))
    pre_params= np.reshape(pre_params,(-1,4))
    #plt.plot(x,vect_gauss(x,pre_params[:,0],pre_params[:,1],pre_params[:,2],pre_params[:,3]),label='prefit')
    #plt.plot(energies,filtered_counts,label='filt')
    plt.plot(energies,binned_counts,label='data')
    plt.plot(x,abs(smooth_resids),label='resids')
    plt.legend()
    plt.show()