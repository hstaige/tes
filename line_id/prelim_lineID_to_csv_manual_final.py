import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from scipy.special import voigt_profile
from scipy.optimize import curve_fit
from uncertainties import ufloat

plt.rcParams.update({'font.size': 12})

data_dir = '/home/tim/research/tes/tdc_included/'
data_files = ['20221219_0000_BCDAC_cal.csv','20221221_0002_AIOAH_cal.csv']
manual_fits = '/home/tim/research/tes/single_fits_params.csv'
ddest = '/home/tim/research/tes/line_id/' # dir to save the various outputs

states = ['20221219_0000_R',
          '20221221_0002_E',
          '20221221_0002_G',
          '20221221_0002_K',
          '20221221_0002_M',
          '20221221_0002_R']

first_col_label = '20221219_0000_B_Energy' # ie the energies

write_csv = False

fwhm_guess = 4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def voigt(x, sigma, gamma, center, A):
     out = voigt_profile(x-center, sigma, gamma)
     return (A*out)/np.max(out)

def multi_voigt(x, *params):
    #fwhm = 0.5346*gamma + (0.2166*(gamma**2)+sigma**2)**0.5 # doi.org/10.1016/0022-4073(77)90161-3
    y = np.zeros_like(x)
    for i in range(0, len(params), 5):
        y = y + voigt(x, params[i], params[i+1], params[i+2], params[i+3]) + params[i+4]
    return y

data = pd.read_csv(data_dir+data_files[0])
if len(data_files)>1:
    for file in data_files[1:]:
        data = pd.concat([data,pd.read_csv(data_dir+file)],axis=1)

org_data = data.rename(columns={first_col_label:'energy'})

energy_ranges = np.loadtxt('/home/tim/research/tes/peak.csv')

for erange in energy_ranges:
    print(erange)
    low_e = erange - fwhm_guess
    high_e = erange + fwhm_guess
    x = np.linspace(low_e,high_e,1000)

    data = org_data.drop(org_data[(org_data.energy < low_e) | (org_data.energy > high_e)].index)

    out = pd.DataFrame(columns=['Line_num','Energy','Energy_unc','Rel_int','Rel_int_unc','FWHM','FWHM_unc','Sigma','Sigma_unc','Gamma','Gamma_unc','State'])
        
    fig, axs = plt.subplots(2,3) # 6 states hardcoded
    axs = axs.ravel()
    for i,ax in enumerate(axs):
        print(states[i])
        counts = data[states[i]+'_Counts']
        
        # sigma, gamma, center, amp, background
        p0 = [3.5,0.0001,erange,np.max(counts),0]
        bounds = [(1,0,erange-2,0,-np.inf),(5,0.001,erange+2,np.inf,np.inf)]
        #bounds = [(1,0,erange-2,0,-0.001),(5,0.001,erange+2,np.inf,0.001)]

        # ax.plot(data.energy,counts,color='b')
        # ax.plot(x, multi_voigt(x,*p0),color='r',zorder=10)
        # plt.show()
        weights = [1/(count**(1/2)) for count in counts]
        try:
            popt, pcov = curve_fit(multi_voigt, data.energy, counts, p0=p0, bounds=bounds, sigma=weights)
            uncertainties = np.sqrt(np.diag(pcov))
        except:
            popt = p0
            uncertainties = [-1,-1,-1,-1,-1]
        ax.plot(data.energy,counts)
        ax.plot(x, multi_voigt(x,*popt),zorder=100)
        j=1
        for line,unc in zip(popt.reshape((-1,5)),uncertainties.reshape((-1,5))):
            ax.plot(data.energy,counts,color='b')
            ax.plot(x, multi_voigt(x,*line),color='k',zorder=10)
            ax.axvline(line[2], color='grey')
            sigma = ufloat(line[0],unc[0])
            gamma = ufloat(line[1],unc[1])
            fwhm = 0.5346*gamma + (0.2166*(gamma**2)+sigma**2)**0.5 # doi.org/10.1016/0022-4073(77)90161-3
            to_out = {'Line_num':j,
                        'Energy':line[2],
                        'Energy_unc':unc[2],
                        'Rel_int':line[3],
                        'Rel_int_unc':unc[3],
                        'Background':line[4],
                        'Background_unc':unc[4],
                        'FWHM':fwhm.n,
                        'FWHM_unc':fwhm.s,
                        'Sigma':sigma.n,
                        'Sigma_unc':sigma.s,
                        'Gamma':gamma.n,
                        'Gamma_unc':gamma.s,
                        'State':states[i]}
            j+=1
            out = out.append(to_out, ignore_index=True)
        ax.set_title(f'{states[i]} {erange}')
    out = out.sort_values(by='Line_num')
    out = out.append([np.NaN], ignore_index=True)
    if write_csv:
        out.to_csv(f'{ddest}/linelist_final.csv', mode='a', header=not os.path.exists(f'{ddest}/linelist_final.csv'))
    plt.show()