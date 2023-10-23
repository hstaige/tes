import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 16})

#floc = 'C:/Users/ahosi/OneDrive/Desktop/calibratedTES_Dec2022'
floc = '/home/tim/research/tes/calibratedTES_Dec2022'
theory_dat = '/home/tim/research/tes/theory/Pr.dat'
full_theory_dat = '/home/tim/research/tes/theory/Pr_3e12/'
efficiency_curve_file = '/home/tim/research/tes/TES_Efficiency_Dec2022.csv'
#ddest = 'C:/data/Line_ID_Nd'
ddest = '/home/tim/research/tes/line_id'

date = '202212'
day = '19'
runnum = '0000'
statelist = ['R']
minenergy = 785
maxenergy = 810
binsize = 1

use_prev_fit = True
write_csv = False

sigma_guess = 2 # approx gaussian width [eV]
gamma_guess = .001 # approx gaussian width [eV]
up_bound_sigma = 10 # max gaussian width
up_bound_gamma = .001 # max lorentzian width
max_center_shift = 10 # distance the fitted line center can be from the data
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
x = np.arange(minenergy, maxenergy+binsize, binsize/5)

th_spectra = os.listdir(full_theory_dat)

file = f'{floc}/{date}{day}_{runnum}'
data = np.loadtxt(f'{file}_{statelist[0]}photonlist.csv', skiprows=1, delimiter=',')
if len(statelist)>1:
    for state in statelist[1:]:
        data = np.vstack((data,np.loadtxt(f'{file}_{statelist}photonlist.csv', skiprows=1, delimiter=',')))

counts, _ = np.histogram(data, bins=bin_edges)
bin_centers = bin_edges[:-1]+binsize/2

theory = np.genfromtxt(theory_dat, usecols=(0,1))
theory = theory[(theory[:,0]>=minenergy) & (theory[:,0]<=maxenergy)]

det_eff = np.loadtxt(efficiency_curve_file,delimiter=',',skiprows=1)
eff_curve = interp1d(det_eff[:,0],det_eff[:,1],kind='cubic')

guess = np.empty(len(theory)*2+2)
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
peaks = theory[:,0]
guess[2::2] = peaks
low_bounds[2::2] = [peak - max_center_shift for peak in peaks]
up_bounds[2::2] = [peak + max_center_shift for peak in peaks]

# amplitudes
amps = theory[:,1] * 1E3 * eff_curve(theory[:,0])
guess[3::2] = amps
low_bounds[3::2] = [amp - max_amp_shift for amp in amps]
low_bounds[3::2] = [0 if lb <0 else lb for lb in low_bounds[3::2]]
up_bounds[3::2] = [amp + max_amp_shift for amp in amps]

prev_fit = True #initializing
if use_prev_fit:
    if os.path.isfile(f'{ddest}/fit_{date}{day}_{runnum}_{statelist}'):
        popt = np.loadtxt(f'{ddest}/fit_{date}{day}_{runnum}_{statelist}', delimiter=',')
        pcov = np.loadtxt(f'{ddest}/cov_{date}{day}_{runnum}_{statelist}', delimiter=',')
    else:
        prev_fit = False
else:
    prev_fit = False

if prev_fit == False:
    popt, pcov = curve_fit(multi_voigt, bin_centers, counts, guess, bounds=(low_bounds,up_bounds))
    np.savetxt(f'{ddest}/fit_{date}{day}_{runnum}_{statelist}', popt, delimiter=',')
    np.savetxt(f'{ddest}/cov_{date}{day}_{runnum}_{statelist}', pcov, delimiter=',')

print(f'Gaussian width: {popt[0]:.3f}', f'Lorentzian width: {popt[1]:.3f}')

#np.savetxt(f'{ddest}/unc_{date}{day}_{runnum}_{statelist}.csv', pcov, delimiter=',', fmt='%.3e')

if write_csv:
    uncertainties = np.sqrt(np.diag(pcov))
    csv_out = np.zeros((int((len(popt)-2)/2),7))
    csv_out[:,0] = popt[2::2]
    csv_out[:,1] = popt[3::2]
    csv_out[:,2] = uncertainties[2::2]
    csv_out[:,3] = uncertainties[3::2]
    csv_out[:,4] = peaks
    csv_out[:,5] = amps
    csv_out[:,6] = abs(csv_out[:,0] - csv_out[:,4])

    np.savetxt(f'{ddest}/list_{date}{day}_{runnum}_{statelist}.csv', csv_out, delimiter=',', fmt='%.3e', header='center,amplitude,center_unc,amp_unc,theory_center,theory_amp,delta_center')

color = iter(cm.rainbow(np.linspace(0, 1, len(th_spectra))))
for spectrum in th_spectra:
    th = np.loadtxt(full_theory_dat+spectrum)
    th = th[(th[:,0]>=minenergy) & (th[:,0]<=maxenergy)]
    th[:,1] = th[:,1] * eff_curve(th[:,0])
    th[:,1] = th[:,1]/np.max(th[:,1])*np.max(counts)
    c = next(color)
    #plt.plot(th[:,0],th[:,1], label=spectrum, linewidth=1.5, color=c)

#for i in theory[:,0]:
#    plt.axvline(i, color='grey', linestyle='--', zorder=0)

#plt.plot(x, multi_voigt(x, *popt), color='r', linestyle='--', zorder=10, label=f'Fitted ($\sigma=${popt[0]:.2f}, $\gamma=${popt[1]:.2f})')
#plt.plot(x, multi_voigt(x, *popt), color='r', linestyle='--', zorder=10, label=f'Theory')
#plt.plot(x,multi_voigt(x,*guess), zorder=6, label='Theory')
plt.plot(bin_centers,counts,marker='.', zorder=5, label='Data')

#np.savetxt(f'{ddest}/{date}{day}_{runnum}_{statelist}_{binsize}bin.csv', , delimiter=',')

plt.legend()
plt.xlabel('Energy [eV]')
plt.ylabel(f'Counts per {binsize} eV bin')
#plt.title(f'{date}{day}_{runnum}_{statelist}')
plt.title(f'{date}{day}_{runnum}_{statelist} (Pr, 1.92 keV, 32.6 mA)')
plt.show()