import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import norm

plt.rcParams.update({'font.size': 16})

anode_v = [5,6,7,8,9,10,11,12,13,14,15]


r_std =[2.9050497603344234, 2.962529901669435, 2.8956338444697742, 2.873184090655507, 2.78912540942713, 2.6031113049073196, 2.4888083166651813, 2.4125037281731667, 2.12564015894465, 2.0956235212902943, 2.0557762222751874]
rw_std =[2.6898614424836125, 2.72154956361469, 2.6284333351153455, 2.6822735704559126, 2.4970689626010953, 2.3971627015665895, 2.1609785463258118, 2.18967407634564, 1.879505190860531, 1.8078583958719245, 1.8074838157636566]

r_std = [i*100 for i in r_std]

plt.scatter(anode_v,r_std)
plt.xlabel('Anode Voltage [kV]')
plt.ylabel('Beam Width [$\mu$m]')
plt.show()
'''
Ar paper:
ramp rate norm exp data
details on crm
label fig 6 and 7
more fig text

full cycle figure with cook and cutouts
cnts/ms for fig 2 and fig 3
5ms bins fig 2
inset for slices
fig 6 run crm to 10kev, match axis
look up nature ect figure for inset insperation
fig 7 bigger annotate letters
fig 1 e1b ect, remake diff levels

to discuss:
are klm xsections similar between he and h

Pr/Nd:
read ASD refs

e1 1150
co like 1180
look at pr 1.3 21

fix unc prop
more lit to the overleaf

'''