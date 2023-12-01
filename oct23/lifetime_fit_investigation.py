import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

isotopes = [{'lifetime':0.1e-3, 'population':30}, {'lifetime':10e-3, 'population':70}]

def gen_timestamps(isotopes, seed=None):
    rng = np.random.default_rng(seed=seed)
    timestamps = []
    for isotope in isotopes:
        lifetime, population = isotope['lifetime'], isotope['population']
        timestamps.append(rng.exponential(scale=lifetime, size=population))
    
    return np.sort(np.hstack(timestamps))

timestamps = gen_timestamps(isotopes)

def exponential(x, amplitude, lifetime):
    return amplitude * np.exp(-x/lifetime)

Exp1, Exp2 = Model(exponential, prefix='e1_'), Model(exponential, prefix='e2_')

Exp_tot = Exp1+Exp2
params_tot = Exp_tot.make_params(e1_amplitude=30, e1_lifetime=0.1e-3, e2_amplitude=70, e2_lifetime=10e-3)

inpu = len(timestamps)-np.arange(len(timestamps))-1
print(timestamps)
out = Exp_tot.fit(inpu, params=params_tot, x=timestamps)
print(out.params.pretty_print())