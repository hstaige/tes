import numpy as np 
import pylab as plt 
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks
import os 
import ebit_util

d = os.path.dirname(os.path.abspath(__file__))
today = "20230727"
rn = "0011"
fltoday = getOffFileListFromOneFile(os.path.join(d, f"{today}", f"{rn}", f"{today}_run{rn}_chan4.off"), maxChans=300)
data = ChannelGroup(fltoday)
data.setDefaultBinsize(0.5)



#Extract hist centers and counts and then plot through pyplot
# centers, counts = data.hist(np.arange(0,50000,50), 'filtValue', states=None, cutRecipeName=None)

# plt.plot(centers, counts)
# plt.show()

#or

#Use channelgroup method
print(data[3].unixnano)
data.plotHists(np.arange(0,50000,50), 'filtValue', channums = [3])
plt.show()