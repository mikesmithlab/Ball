import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from OSFile import get_files_directory, write_row_to_csv

def msd(data_frame,fps=500.0,max_t=2.0,show=False):
    lags_original = np.logspace(0,np.log10(max_t*500.0),num=20)

    lags = [int(i) for i in lags_original]
    msd_vals = np.array([])
    for lag in lags:
        print(lag)
        msd_vals = np.append(msd_vals,((data_frame['ballXMM'].diff(periods=int(lag)))**2).mean())
    lags = np.array(lags)/fps
    if show:
        plt.figure('msd')
        plt.plot(lags,msd_vals,'-')

    return lags, msd_vals


if __name__ == '__main__':
    # Load dataframe
    # filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/',title='Select Data File', filetypes = (('DataFrames', '*finaldata.hdf5'),))
    basepath = '/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/*'

    names = ['800_050','400_050','240_050','120_050','80_050','800_077','800_062','800_040','400_077','400_062','400_040','240_077','240_062','240_040','120_077','120_062','120_040','80_077','80_062','80_040']
    #names = ['P80_077']
    pathnames = [basepath + name + '_data_finaldata.hdf5' for name in names]
    # each path get the 3 files for a particular experiment.
    for path in pathnames:

        filenames = get_files_directory(path, full_filenames=True)

        # Read dataframe from file
        msd_av = np.array([])
        filename_op = min(filenames)[:-5]

        for i, filename in enumerate(filenames):
            data = pd.read_hdf(filename)
            lags,msd_val = msd(data, show=True)
            if i==0:
                msd_av = msd_val
            elif i ==1:
                msd_av = (msd_av + msd_val)
            elif i == 2:
                msd_av = (msd_av + msd_val)/3

        log_lags = np.log10(lags)
        log_msd_av = np.log10(msd_av)

        output =np.c_[lags, msd_av, log_lags, log_msd_av]
        np.savetxt(filename_op + '_msdvals.csv', output, fmt='%.3f', delimiter=',',
                   header=" lags, msd_vals. log_lags, log_msd")
    plt.show()