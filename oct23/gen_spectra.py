import numpy as np
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.signal import find_peaks, savgol_filter, wiener, deconvolve

def voigt(x,center,A,sigma,gamma):
     out = voigt_profile(x-center, sigma, gamma)
     return (A*out)/np.max(out)

def multi_voigt(x, *params):
    params = params[0]
    sigma = params[0]
    gamma = params[1]
    #fwhm = 0.5346*gamma + (0.2166*(gamma**2)+sigma**2)**0.5 # doi.org/10.1016/0022-4073(77)90161-3
    y = np.zeros_like(x)
    for i in range(2,len(params),2):
        y = y + voigt(x,params[i],params[i+1],sigma,gamma)
    return y

def deriv(x,order):
    x_max = np.max(x)
    for _ in range(order):
        x = np.gradient(x,edge_order=2)
    x *= x_max/np.max(x)
    return x

x = np.linspace(0,50,10000)
params = [1,0.000001,15,5,16.4,4]
y = multi_voigt(x,params)
y += np.random.normal(0,.05,10000)

q,deconv = deconvolve(y, voigt(x,15,.1,1,0.000001))
print(q)
plt.plot(x,y,label='d')
# plt.plot(x,deriv(y,1),label='1')
# plt.plot(x,deriv(y,2),label='2')
# plt.plot(x,deriv(y,4),label='4')
plt.plot(x,wiener(y),label='w')
plt.plot(x,deconv,label='d')
plt.ylim([-np.max(y),np.max(y)])
plt.legend()
plt.show()