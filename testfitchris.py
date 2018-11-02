from fitting import Fit
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt




filename='/home/ppzmis/Documents/PythonScripts/forcedata.txt'


data_to_be_fitted = np.loadtxt(filename, delimiter='\t',skiprows=1)
force_per_pixel = data_to_be_fitted[:, 5]
intensity = data_to_be_fitted[:, 7]

logic = intensity > 35


plt.figure()
plt.plot(force_per_pixel[logic],intensity[logic],'x')
plt.show()

fit_obj = Fit('axb', x=force_per_pixel, y=intensity)

fit_obj.add_filter(logic)
fit_obj.add_params([0,0.33],lower=[None, 0], upper=[None,2])
params=fit_obj.fit(errors=False)

fit_obj.plot_fit(show=True)
