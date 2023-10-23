import mass
import numpy as np
import pylab as plt
plt.ion()

pulse_files = ["/home/pcuser/Desktop/Periodicity_Testing/20230630/0001/20230630_run0001_chan1.ljh"]
data = mass.TESGroup(filenames = pulse_files,
        noise_filenames = pulse_files)

data.summarize_data()
ds = data.first_good_dataset

plt.plot(np.diff(ds.p_rowcount[:]),".")