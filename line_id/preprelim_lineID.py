import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.special import voigt_profile
from scipy.optimize import curve_fit

#floc = 'C:/Users/ahosi/OneDrive/Desktop/calibratedTES_Dec2022'
floc = '/home/tim/research/tes/calibratedTES_Dec2022'
theory_dat = '/home/tim/research/tes/theory/Pr.dat'
#ddest = 'C:/data/Line_ID_Nd'
ddest = '/home/tim/research/tes/line_id'

date = '202212'
day = '19'
runnum = '0000'
statelist = ['Y']
minenergy = 600
maxenergy = 2000
binsize = .5

plot_theory = True
peak_prom = 85 # how distinct a peak must be to be included
manual_peaks_ind = [803,1133,1197] # extra peaks to be added [eV]
sigma_guess = 2 # approx gaussian width [eV]
gamma_guess = .1 # approx gaussian width [eV]
up_bound_sigma = 10 # max gaussian width
up_bound_gamma = .5 # max lorentzian width
max_center_shift = 2 # distance the fitted line center can be from the data
max_amp_shift = np.inf # distance the fitted line amplitude can be from the data

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def voigt(x,center,A,sigma,gamma):
     out = voigt_profile(x-center, sigma, gamma)
     return (A*out)/np.max(out)

def multi_voigt(x, *params):
    sigma = params[0]
    gamma = params[1]
    #fwhm = 0.5346*gamma + (0.2166*(gamma**2)+sigma**2)**0.5 # doi.org/10.1016/0022-4073(77)90161-3
    y = np.zeros_like(x)
    for i in range(2,len(params),2):
        y = y + voigt(x,params[i],params[i+1],sigma,gamma)
    return y

bin_edges = np.arange(minenergy, maxenergy+binsize, binsize)

file = f'{floc}/{date}{day}_{runnum}'
data = np.loadtxt(f'{file}_{statelist[0]}photonlist.csv', skiprows=1, delimiter=',')
if len(statelist)>1:
    for state in statelist[1:]:
        data = np.vstack((data,np.loadtxt(f'{file}_{statelist}photonlist.csv', skiprows=1, delimiter=',')))

theory = np.genfromtxt(theory_dat, usecols=(0,1))

counts, _ = np.histogram(data, bins=bin_edges)
bin_centers = bin_edges[:-1]+binsize/2
peaks_ind, _ = find_peaks(counts, prominence=peak_prom) # returns index of detected peaks_ind

man_ind = [np.argmin(abs(peak-bin_centers)) for peak in manual_peaks_ind] # closest bin_center to each specified manual_peaks_ind
peaks_ind = np.concatenate((peaks_ind,man_ind))
peaks_ind = peaks_ind.astype('int')

guess = np.empty(len(peaks_ind)*2+2)
low_bounds = np.empty(len(guess))
up_bounds = np.empty(len(guess))

# gaussian width
guess[0] = sigma_guess
low_bounds[0] = 0
up_bounds[0] = up_bound_sigma

# lorentzian width
guess[1] = gamma_guess
low_bounds[1] = 0
up_bounds[1] = up_bound_gamma

# centers
peaks = [bin_centers[peak_ind] for peak_ind in peaks_ind]
guess[2::2] = peaks
low_bounds[2::2] = [peak - max_center_shift for peak in peaks]
up_bounds[2::2] = [peak + max_center_shift for peak in peaks]

# amplitudes
amps = [counts[i] for i in peaks_ind]
guess[3::2] = amps
low_bounds[3::2] = [amp - max_amp_shift for amp in amps]
up_bounds[3::2] = [amp + max_amp_shift for amp in amps]

#popt, pcov = curve_fit(multi_voigt, bin_centers, counts, guess, bounds=(low_bounds,up_bounds))
#print(f'Gaussian width: {popt[0]:.3f}', f'Lorentzian width: {popt[1]:.3f}')

x = np.arange(minenergy, maxenergy+binsize, binsize/5)

'''
if plot_theory:
    theory[:,1] = theory[:,1]/np.max(theory[:,1])*1E4
    theory = np.concatenate(([3],[1],theory.flatten()))
    plt.plot(x,multi_voigt(x, *theory), color='g', label='Theory')
'''
#if plot_theory:
#    for i in theory[:,0]:
#        plt.axvline(i, color='g', linestyle='--', zorder=0)

for i in peaks_ind:
    pass
    #plt.axvline(bin_centers[i], color='grey', linestyle='--', zorder=0)
#plt.plot(bin_centers, multi_voigt(bin_centers, *guess), color='b', linestyle=':', zorder=9, label='Before fitting')
#plt.plot(x, multi_voigt(x, *popt), color='r', linestyle='--', zorder=10, label='Fitted')
plt.plot(bin_centers,counts,marker='.', zorder=5, label='Data')

plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel(f'Counts per {binsize} eV bin')
plt.title(f'{date}{day}_{runnum}_{statelist}')
plt.show()