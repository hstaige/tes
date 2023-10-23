import numpy as np 
import pylab as plt 
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
import os 
import ebit_util
import time
from scipy.fft import fft, fftfreq

d = os.path.dirname(os.path.abspath(__file__))
today = "20230728"
rn = "0005"

fltoday = getOffFileListFromOneFile(os.path.join(d, f"{today}", f"{rn}", f"{today}_run{rn}_chan4.off"), maxChans=300)
external_trigger_filename =  os.path.join(d, f"{today}", f"{rn}", f"{today}_run{rn}_external_trigger.bin")
data = ChannelGroup(fltoday)
data.setDefaultBinsize(0.1)
external_trigger_rowcount = ebit_util.get_external_triggers(external_trigger_filename, good_only=True)

for ds in data.values():
    ebit_util.calc_external_trigger_timing(ds, external_trigger_rowcount)
seconds_after_external_triggers = np.hstack([ds.seconds_after_external_trigger[ds.getStatesIndicies()[0]] for ds in data.values()])

state = 'B'
sec = []
for i in data:
    ds = data[i] 
    sec = np.concatenate([sec,ds.seconds_after_external_trigger[tuple(ds.getStatesIndicies(states=[state]))][ds.getAttr("cutNone", [state], "cutNone")]]) #change the cut by swapping out the first "cutNone" with another cut.
seconds_after_external_triggers = sec
binsize = 1e-3
t_bounds = [np.min(seconds_after_external_triggers),np.max(seconds_after_external_triggers)]
bins = np.arange(t_bounds[0],t_bounds[1],binsize)
counts,_ = np.histogram(seconds_after_external_triggers,bins)

N = len(counts)
T = binsize
xf = fftfreq(N,T)[1:N//2]

plt.plot(xf,2/N*np.abs(fft(counts)[1:N//2]))
plt.show()
