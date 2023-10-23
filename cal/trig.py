import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import argrelextrema
from mass.off import ChannelGroup, getOffFileListFromOneFile
import ebit_util
import pandas as pd

dir = '/media/tim/HDD/tesdata'
day = "20221221"
run = "0002"

dir = '/home/tim/research/tes/cal_data'
day = "20230803"
run = "0007"

external_trigger_filename =  f'{dir}/{day}/{run}/{day}_run{run}_external_trigger.bin'
etrc = ebit_util.get_external_triggers(external_trigger_filename, good_only=False)

clk_p = 200e-9
lim = 2000

#etrc = etrc[:lim]

d = np.diff(etrc)
plt.figure()
plt.plot(d,".",label="all")
#plt.yscale('log')
plt.show()
quit()
max_t = int(round(np.max(d)*clk_p)/clk_p)

first_real_trig_ind = np.argmax(d>(max_t/2))
first_real_trig = etrc[first_real_trig_ind + 1]
resids = (etrc - first_real_trig) % max_t
real_trigs = argrelextrema(resids, np.less)[0]

plt.figure()
# plt.plot(d,".",label="all")
for i in real_trigs:
    plt.axvline(i)
plt.scatter(range(len(etrc)),(etrc-first_real_trig)*clk_p, marker='.')
#plt.yscale('log')
plt.xlabel('Trigger number')
plt.ylabel('Row Count')
plt.show()