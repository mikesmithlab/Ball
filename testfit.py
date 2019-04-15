from fitting import Fit
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt




basepath='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/xpos_msd/'

#filename = filedialog.askopenfilename(initialdir=basepath,title='Select Data File', filetypes = (('CSVS', '*.csv'),))
filename = basepath + '10mmP120_045_data_finaldata_msdvals_errors.csv'
print(filename)
data_to_be_fitted = np.loadtxt(filename, delimiter=',',skiprows=1)
bins = data_to_be_fitted[:, 2]
prob = data_to_be_fitted[:, 3]


#velocities = np.random.normal(0,0.8,10000000)
#r_omega = np.random.normal(0,5*0.159,10000000)

#combo = -velocities/r_omega

#freq,bin_edges = np.histogram(combo, bins = 31,range=(-5,5))
#bins = (bin_edges[:-1] + bin_edges[1:])/2
#fig = plt.figure()#filename + param + '_histogram')
#fig.suptitle('')
#plt.plot(bins,prob,'r-')
#plt.plot(bins,prob,'bx')
#plt.show()
logic = prob > np.log10(0.002)
logic2 = bins < np.log10(0.4)
logic3 = logic * logic2
print(logic3)

fit_obj = Fit('linear', x=bins, y=prob)
fit_obj.add_filter(logic3)
fit_obj.add_params([10,1],lower=None, upper=None)
params=fit_obj.fit(interpolation_factor=0.1, errors=True)

fit_obj.plot_fit(show=True)
