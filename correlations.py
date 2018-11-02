import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import csv
#from itertools import izip
from OSFile import get_files_directory, write_row_to_csv


if __name__ == '__main__':

    # Load dataframe
    # filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/',title='Select Data File', filetypes = (('DataFrames', '*finaldata.hdf5'),))
    basepat h ='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/*'

    # names = ['800_050','400_050','240_050','120_050','80_050','800_077','800_062','800_040','400_077','400_062','400_040','240_077','240_062','240_040','120_077','120_062','120_040','80_077','80_062','80_040']
    names = ['P80_077']
    pathnames = [basepath + name + '_data_finaldata.hdf5' for name in names]
    param = 'xVelMM'
    # each path get the 3 files for a particular experiment.
    for path in pathnames:

        filename s =get_files_directory(path ,full_filenames=True)

        # Read dataframe from file


        filename_op = min(filenames)[:-5]
        if '_040' in filename_op:
            acceleration = 1.9
        elif '_050' in filename_op:
            acceleration = 2.25
        elif '_062' in filename_op:
            acceleration = 2.75
        else:
            acceleration = 3.25

        if 'P80_' in filename_op:
            roughness = 201.0
        elif 'P120_' in filename_op:
            roughness = 124.0
        elif 'P240_' in filename_op:
            roughness = 58.50
        elif 'P400_' in filename_op:
            roughness = 35.0
        else:
            roughness = 21.6


        for i ,filename in enumerate(filenames):


            data = pd.read_hdf(filename)
            diffs, time_bounces = changes(data, filename_op + 'changes', param)

            if param == 'vx_over_r_omegak':
                vals = -data['xVelMM'].value s /( 5 *data['omega_k'].values)
            else:
                vals = data[param].values
            filename_op2 = filename[:-5]

            # histogram(vals,filename_op2,'individual'+ param)

            if i == 0:
                datavals = vals
                output = np.c_[roughness, acceleration ,vals.std() ,vals.mean() ,vals.max() ,vals.min()]
                changevals = diffs
                timevals  = time_bounces
            elif i== 1:
                output = np.vstack(
                    (output, np.c_[roughness, acceleration, vals.std(), vals.mean(), vals.max(), vals.min()]))
                datavals = np.append(datavals, vals)
                changevals = np.append(changevals, diffs)
                timevals = np.append(timevals, time_bounces)
            else:
                output = np.vstack(
                    (output, np.c_[roughness, acceleration, vals.std(), vals.mean(), vals.max(), vals.min()]))
                datavals = np.append(datavals, vals)
                changevals = np.append(changevals, diffs)
                timevals = np.append(timevals, time_bounces)
        print(output)
        print(output[:, 2])
        print(output[:, 2].std())
        output = np.insert(output, 3, output[:, 2].std(), axis=1)
        print(output)
        output2 = [np.mean(output, axis=0)]
        print(output2)
        print(filename_op)
        np.savetxt(filename_op + param + 'calcvals.csv', output, fmt='%.3f', delimiter=',',
                   header=" roughness, acceleration, std,+-, mean,+,-")
        np.savetxt(filename_op + param + 'meancalcvals.csv', output2, fmt='%.3f', delimiter=',',
                   header=" roughness, acceleration, std,+-, mean,+,-")

        print(np.shape(changevals))

        # histogram(datavals,filename_op,param,range_data=(-5,5),symmetric=False)
        histogram(changevals, filename_op + '_changes_', param)

        plt.figure()
        plt.plot(timevals, changevals, 'rx')
        plt.show()







