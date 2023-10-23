
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from uncertainties import ufloat
from scipy.special import voigt_profile
from lmfit import create_params, minimize

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
fwhm_guess = 4

erange = 1145

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def gauss_resid(variables, x, data, uncertainty):
    sigma = variables[0]
    gamma = 0.001
    amp = variables[1]
    center = variables[2]

    model = voigt_profile(x-center,sigma,gamma)
    model = amp*model/np.max(model)

    return (data-model) / uncertainty

data = pd.read_csv(data_dir+data_files[0])
if len(data_files)>1:
    for file in data_files[1:]:
        data = pd.concat([data,pd.read_csv(data_dir+file)],axis=1)

org_data = data.rename(columns={first_col_label:'energy'})

print(erange)
low_e = erange - fwhm_guess
high_e = erange + fwhm_guess
x = np.linspace(low_e,high_e,1000)

data = org_data.drop(org_data[(org_data.energy < low_e) | (org_data.energy > high_e)].index)

fig, axs = plt.subplots(2,3) # 6 states hardcoded
axs = axs.ravel()
for i,ax in enumerate(axs):
    print(states[i])
    counts = data[states[i]+'_Counts']
    print(counts.items)
    weights = [(count**(1/2)) for count in counts]
    params = create_params(sigma=3, amp=np.max(counts[:,1]),center=counts[:,0])
    print(minimize(gauss_resid,params,args=(x,counts,weights)))

# energy_ranges = np.loadtxt('/home/tim/research/tes/peak.csv') # manually specfied peaks

# for erange in energy_ranges:
#     print(erange)
#     low_e = erange - fwhm_guess
#     high_e = erange + fwhm_guess
#     x = np.linspace(low_e,high_e,1000)

#     data = org_data.drop(org_data[(org_data.energy < low_e) | (org_data.energy > high_e)].index)
