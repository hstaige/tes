import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from lmfit import Model
from mass import MLEModel
import my_utils as utils

method = 1
# 0: LS
# 1: MLE

l1 = 1.5e-4
l2 = 4e-3

isotopes = [{'lifetime':l1, 'population':30}, {'lifetime':l2, 'population':70}]

def gen_timestamps(isotopes, seed=None):
    rng = np.random.default_rng(seed=seed)
    timestamps = []
    for isotope in isotopes:
        lifetime, population = isotope['lifetime'], isotope['population']
        timestamps.append(rng.exponential(scale=lifetime, size=population))
    
    return np.sort(np.hstack(timestamps))

if method==0:
    for seed, color in enumerate(cm.rainbow(np.linspace(0, 1, 10))):
        timestamps = gen_timestamps(isotopes, seed=seed)

        def exponential(x, a, l):
            return a * np.exp(-x/l)

        Exp1, Exp2 = Model(exponential, prefix='e1_'), Model(exponential, prefix='e2_')

        Exp_tot = Exp1+Exp2
        params_tot = Exp_tot.make_params(e1_a={'value':30,'min':0}, e1_l={'value':1e-4,'min':0}, e2_a={'value':70,'min':0}, e2_l={'value':1e-3,'min':0})

        bins = np.arange(.000001,.0001,.000001)
        error = []
        for bin in bins:
            t_bin_edges = np.arange(0,.025,bin)
            counts,_ = np.histogram(timestamps, bins = t_bin_edges)
            out = Exp_tot.fit(counts, params=params_tot, x=utils.midpoints(t_bin_edges))
            l1_fit = out.params['e1_l']
            l2_fit = out.params['e2_l']
            error.append(((abs((l1-l1_fit)/l1_fit))**2 + (abs((l2-l2_fit)/l2_fit))**2)**.5)
        plt.plot(bins,error,color=color)

    plt.xlabel('bins size [s]')
    plt.ylabel('error')
    plt.ylim([0,1])
    plt.show()


if method==1:
    for seed, color in enumerate(cm.rainbow(np.linspace(0, 1, 10))):
        timestamps = gen_timestamps(isotopes, seed=seed)

        def twoexp(t, A1, tau1, A2, tau2):
            return A1/tau1*np.exp(-t/tau1)+A2/tau2*np.exp(-t/tau2)
        
        model = MLEModel(twoexp)
        params = model.make_params()
        params["A1"].set(30, min=0)
        params["tau1"].set(1e-4, min=0)
        params["A2"].set(70, min=0)
        params["tau2"].set(1e-3, min=0)

        bins = np.arange(.000001,.0001,.000001)
        error = []
        for bin in bins:
            #print(bin)
            t_bin_edges = np.arange(0,.025,bin)
            counts,_ = np.histogram(timestamps, bins = t_bin_edges)

            out = model.fit(counts, t=utils.midpoints(t_bin_edges), params=params)
            l1_fit = out.params['tau1']
            l2_fit = out.params['tau2']
            error.append((((l1-l1_fit)/l1_fit)**2 + ((l2-l2_fit)/l2_fit)**2)**.5)
        plt.plot(bins,error,color=color)

    plt.xlabel('bins size [s]')
    plt.ylabel('error')
    plt.ylim([0,1])
    plt.show()