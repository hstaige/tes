import mass
import numpy as np
import pylab as plt
plt.ion()

pulse_files = "/home/pcuser/Desktop/Periodicity_Testing/20230720/0001/20230720_run0001_chan*.ljh"
data = mass.TESGroup(filenames = pulse_files,
        noise_filenames = pulse_files, overwrite_hdf5_file=True, max_chans=2)

data.summarize_data()
ds = data.first_good_dataset
data.calc_external_trigger_timing(from_nearest=True) 

plt.figure()
plt.plot(np.diff(ds.p_rowcount[:-100]),".")
plt.xlabel("pulse index")
plt.ylabel("diff(p_rowcount)")

plt.figure()
plt.plot(np.diff(ds.external_trigger_rowcount[:-100]),".")
plt.xlabel("pulse index")
plt.ylabel("diff(external_trigger_rowcount)")

plt.figure()
plt.plot(ds.rows_from_nearest_external_trigger[:-100],".")
plt.xlabel("pulse index")
plt.ylabel("rows_from_nearest_external_trigger")
