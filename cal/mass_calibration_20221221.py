
import numpy as np 
import matplotlib.pyplot as plt 
from mass.off import ChannelGroup, getOffFileListFromOneFile
import ebit_util
import pandas as pd

dir = '/media/tim/HDD/tesdata'
day = "20221221"
run = "0002"
datdest = '/home/tim/research/tes/cal_data'
# '/media/tim/HDD/tesdata/20221221/0002/'
cal_states = ['A','B','I','O','AH']

file = getOffFileListFromOneFile(f'{dir}/{day}/{run}/{day}_run{run}_chan1.off', maxChans=300)

data = ChannelGroup(file)
defbinsize = 0.1
data.setDefaultBinsize(defbinsize)

ds = data[1]
# ds.plotHist( np.arange(0, 60000, 20), "filtValue", coAddStates=False, states=cal_states)   #states=None by default uses all states
# plt.show()
# quit()

ds.learnDriftCorrection(overwriteRecipe=True, states=cal_states)
ds.calibrationPlanInit("filtValueDC")
ds.calibrationPlanAddPoint(8190, "AlKAlpha", states=cal_states)
ds.calibrationPlanAddPoint(9550, "SiKAlpha", states=cal_states)
ds.calibrationPlanAddPoint(14090, "ClKAlpha", states=cal_states)
ds.calibrationPlanAddPoint(17550, "KKAlpha", states=cal_states)
ds.calibrationPlanAddPoint(23270, "TiKAlpha", states=cal_states)
ds.calibrationPlanAddPoint(31720, "FeKAlpha", states=cal_states)
ds.calibrateFollowingPlan("filtValueDC")
data.learnDriftCorrection(overwriteRecipe=True, states=cal_states)
data.alignToReferenceChannel(ds, "filtValueDC", np.arange(0,50000,30))
data.calibrateFollowingPlan("filtValueDC")

external_trigger_filename =  f'{dir}/{day}/{run}/{day}_run{run}_external_trigger.bin'
external_trigger_rowcount = ebit_util.get_external_triggers(external_trigger_filename, good_only=True)
for ds in data.values():
    ebit_util.calc_external_trigger_timing(ds, external_trigger_rowcount)

scistates = ['A']
for state in scistates:
 
    histall = np.array(data.hist(np.arange(0, 14000, defbinsize), "energy", states=state))
    stadat = pd.DataFrame(data=histall)
    stadat.to_csv(f'{datdest}/{day}_{run}_{state}.csv', index=False)

    sec = []
    energy = []
    time = []
    for i in data:
        ds = data[i]    
        sec = np.concatenate([sec,ds.seconds_after_external_trigger[tuple(ds.getStatesIndicies(states=[state]))][ds.getAttr("cutNone", [state], "cutNone")]]) #change the cut by swapping out the first "cutNone" with another cut.
        energy = np.concatenate([energy, ds.getAttr("energy", state, "cutNone")])
        #energy.extend(list(ds.getAttr('energy', state)))
        time.extend(list(ds.getAttr('unixnano', state)))
    plist = np.array([energy, time, sec], dtype=object).T

    photlist = pd.DataFrame(data=plist, columns=['energy', 'time', 'tst'])
    photlist.to_csv(f'{datdest}/{day}_{run}_{state}photonlist.csv', index=False)
