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
ddest = '/home/tim/research/tes/lineid/' # dir to save the various outputs

states = ['20221219_0000_R',
          '20221221_0002_E',
          '20221221_0002_G',
          '20221221_0002_K',
          '20221221_0002_M',
          '20221221_0002_R']

# states = ['20221221_0002_V',
#           '20221221_0002_X',
#           '20221221_0002_Z',
#           '20221221_0002_AB',
#           '20221221_0002_AD',
#           '20221221_0002_AF']


first_col_label = '20221219_0000_B_Energy' # ie the energies

energy_ranges = {0:[780,1900],
                 1:[790,805],
                 2:[1110,1120],
                 3:[1140,1155],
                 4:[814,825],
                 5:[879,900],
                 6:[909,941],
                 7:[1092,1105],
                 8:[1178,1188],
                 9:[1292,1305],
                 10:[1065,1225]}

dict_ind = 0

do_wat = 5
# 0: manual
# 1: fit
# 2: big
# 3: summed

write_csv = True

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def voigt(x, sigma, gamma, center, A):
     out = voigt_profile(x-center, sigma, 0.001)
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

data = data.rename(columns={first_col_label:'energy'})
low_e, high_e = energy_ranges[dict_ind]
data.drop(data[(data.energy < low_e) | (data.energy > high_e)].index, inplace=True)

out = pd.DataFrame(columns=['Line_num','Energy','Energy_unc','Rel_int','Rel_int_unc','FWHM','FWHM_unc','Sigma','Sigma_unc','Gamma','Gamma_unc','State'])

#color = iter(cm.rainbow(np.linspace(0, 1, len(states))))
if do_wat==1:
    fig, axs = plt.subplots(2,3) # 6 states hardcoded
    axs = axs.ravel()
    for i,ax in enumerate(axs):
        print(states[i])
        man_params = pd.read_csv(manual_fits, comment='#', skip_blank_lines=True)
        voigts = man_params[(man_params['key']==dict_ind) & (man_params['state']==i)].to_numpy()
        num = len(voigts)
        p0 = voigts[:,2:].flatten()
        bounds = [(0,0,low_e,0,0)*num,(3,3,high_e,np.inf,np.inf)*num]

        weights = [count**(-1/2) for count in data[states[i]+'_Counts']]
        popt, pcov = curve_fit(multi_voigt, data.energy, data[states[i]+'_Counts'], p0=p0, bounds=bounds, sigma=weights, maxfev=10000)
        #popt, pcov = curve_fit(multi_voigt, data.energy, data[states[i]+'_Counts'], p0=p0, bounds=bounds, maxfev=10000)
        uncertainties = np.sqrt(np.diag(pcov))
        ax.plot(data.energy,data[states[i]+'_Counts'])
        x = np.linspace(low_e,high_e,1000)
        ax.plot(x, multi_voigt(x,*popt),zorder=100)
        j=1
        for line,unc in zip(popt.reshape((-1,5)),uncertainties.reshape((-1,5))):
            ax.plot(x, multi_voigt(x,*line),color='k',zorder=10)
            ax.axvline(line[2], color='grey')
            fwhm = 0.5346*line[1] + (0.2166*(line[1]**2)+line[0]**2)**0.5 # doi.org/10.1016/0022-4073(77)90161-3
            to_out = {'Line_num':j,
                      'Energy':line[2],
                      'Energy_unc':unc[2],
                      'Rel_int':line[3],
                      'Rel_int_unc':unc[3],
                      'Background':line[4],
                      'Background_unc':unc[4],
                      'FWHM':fwhm,
                      'FWHM_unc':-1,
                      'Sigma':line[0],
                      'Sigma_unc':unc[0],
                      'Gamma':line[1],
                      'Gamma_unc':unc[1],
                      'State':states[i]}
            j+=1
            out = out.append(to_out, ignore_index=True)
        ax.set_title(states[i])
    out = out.sort_values(by='Line_num')
    if write_csv:
        out.to_csv(f'{ddest}/linelist_{low_e}_{high_e}.csv')
    plt.show()

elif do_wat==0:
    fig, axs = plt.subplots(2,3) # 6 states hardcoded
    axs = axs.ravel()
    plt.ion()
    while True:
        for i,ax in enumerate(axs):
            plt.ion()
            man_params = pd.read_csv(manual_fits, comment='#', skip_blank_lines=True)
            voigts = man_params[(man_params['key']==dict_ind) & (man_params['state']==i)].to_numpy()
            p0 = voigts[:,2:].flatten()
            x = np.linspace(low_e,high_e,1000)
            ax.clear()
            ax.plot(data.energy,data[states[i]+'_Counts'])
            ax.plot(x, multi_voigt(x,*p0))
            ax.set_title(states[i])
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

elif do_wat==2:
    for state in states:
        plt.plot(data.energy,data[state+'_Counts'], label=state)
    plt.legend()
    plt.show()

elif do_wat==3:
    summed = np.zeros_like(data.energy)
    for state in states:
        summed += data[state+'_Counts']
    plt.plot(data.energy,summed)
    plt.legend()
    plt.show()

elif do_wat==4:


    fig, axs = plt.subplots(2,3) # 6 states hardcoded
    axs = axs.ravel()
    for i,ax in enumerate(axs):
        print(states[i])
        man_params = pd.read_csv(manual_fits, comment='#', skip_blank_lines=True)
        voigts = man_params[(man_params['key']==dict_ind) & (man_params['state']==i)].to_numpy()
        num = len(voigts)
        p0 = voigts[:,2:].flatten()
        bounds = [(0,0,low_e,0,0)*num,(3,3,high_e,np.inf,np.inf)*num]

        weights = [count**(-1/2) for count in data[states[i]+'_Counts']]
        popt, pcov = curve_fit(multi_voigt, data.energy, data[states[i]+'_Counts'], p0=p0, bounds=bounds, sigma=weights, maxfev=10000)
        #popt, pcov = curve_fit(multi_voigt, data.energy, data[states[i]+'_Counts'], p0=p0, bounds=bounds, maxfev=10000)
        uncertainties = np.sqrt(np.diag(pcov))
        ax.plot(data.energy,data[states[i]+'_Counts'])
        x = np.linspace(low_e,high_e,1000)
        ax.plot(x, multi_voigt(x,*popt),zorder=100)
        j=1
        for line,unc in zip(popt.reshape((-1,5)),uncertainties.reshape((-1,5))):
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
        ax.set_title(states[i])
    out = out.sort_values(by='Line_num')
    plt.show()


elif do_wat==5:
    full_theory_dat = '/home/tim/research/tes/theory/Pr_3e12/'
    th_spectra = os.listdir(full_theory_dat)


    summed = np.zeros_like(data.energy)
    for state in states:
        summed += data[state+'_Counts']

    color = iter(cm.rainbow(np.linspace(0, 1, len(th_spectra))))
    for spectrum in th_spectra:
        th = np.loadtxt(full_theory_dat+spectrum)
        #th = th[(th[:,0]>=minenergy) & (th[:,0]<=maxenergy)]
        th[:,1] = th[:,1]/np.max(th[:,1])*np.max(summed)
        c = next(color)
        plt.plot(th[:,0],th[:,1], label=spectrum, linewidth=1.5, color=c)

    plt.plot(data.energy,summed, color='k', label='Exp')
    plt.legend()
    plt.show()


