import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import csv
#from itertools import izip
from OSFile import get_files_directory, write_row_to_csv


def plotWithBounces(ball_track, param, data_length=300):
    
    bounce_indices = np.where((ball_track['bounce'] == True)&(ball_track.index < data_length))#np.where(ball_track[ball_track.index < data_length]['bounce'] == True)
    #bounce_indices=np.append(np.array([[ball_track.index.min()]]),bounce_indices)
    #bounce_indices=np.append(bounce_indices,np.array([[ball_track[ball_track.index < data_length].index.max()]]))
    plt.figure(param)
    plt.plot(ball_track[ball_track.index < data_length].index,ball_track[ball_track.index < data_length][param],'bx')
    for bounce_index in bounce_indices:
        plt.plot([bounce_index,bounce_index],[ball_track[ball_track.index < data_length][param].min(),ball_track[ball_track.index < data_length][param].max()],'g--')
    
def createHistogram(ball,param, filename,save=False):
    values = ball[param]
    
    histogram(values,filename,param)   
    

def changes(ball,filename,param,exclude_wall = True):
    'works out how a parameter changes due to a bounce, Excludes those points near a wall'
    bounce_vals = ball[(ball['bounce']==True)][param]
    #wall_vals = ball[(ball['bounce']==True)&(ball['wall']==True)].index
    #print(np.shape(wall_vals))
    differences = bounce_vals.diff()

    #filtered_differences = differences[wall_vals]
    filtered_differences=differences.fillna(value=0)
    return filtered_differences.values,np.append([0],np.diff(differences.index))

def histogram(values,filename,param,num_bins=15,range_data=False,symmetric=True,save=True):
    print(values.max())
    print(values.min())
    #if symmetric == True:
    #    std=values.std()
    #range_data = (values.min(), values.max())
        
       
    #if (range_data == False) and (symmetric == False):
    #    freq,bin_edges = np.histogram(values, bins = num_bins)
    #else:
    freq,bin_edges = np.histogram(values, bins = num_bins,range=range_data)
    bins = (bin_edges[:-1] + bin_edges[1:])/2
    freq = freq / (np.shape(values)[0] * (bin_edges[2] - bin_edges[1]))
    fig = plt.figure()#filename + param + '_histogram')
    fig.suptitle(np.shape(values))
    plt.plot(bins,freq,'r-')
    plt.plot(bins,freq,'bx')
    

    if save:
        #plt.savefig(filename + param +'.png')
        
        np.savetxt(filename + param + '.csv', np.c_[bins,freq], fmt='%.5f', delimiter=',', header=" bins,freq")
    
def calcBasics(ball,param, filename, save=True, append=False):
    meanVal = ball[param].mean()
    stdVal = ball[param].std()
    minVal = ball[param].min()
    maxVal = ball[param].max()
    medianVal = ball[param].median()
    magnmean = ball[param].abs().mean()
    
    if append == False:
        return np.c_[meanVal,maxVal,minVal]
    else:
        np.savetxt(filename + param + 'calcvals.csv', np.c_[meanVal,stdVal], fmt='%.5f', delimiter=',', header=" mean,+-")  
        
    
    
    
    #print(param + ':(mean,std,min,max,median,magnmean)')
    #print([meanVal,stdVal,minVal,maxVal,medianVal,magnmean])
    
    #Find those bounces that are followed by a wall
    
    
    
    #print(ball[ball['bounce']==True][param])
    values=np.diff(ball[ball['bounce']==True][param])
    indices=ball[ball['bounce']==True][param].index
    #print(values)
    
    if param == 'ballHeightMM':
        return [str(meanVal),str(minVal),str(maxVal)]
    else:
        return [meanVal,stdVal,medianVal,minVal,maxVal,magnmean]
    
    
        
    
    
    
if __name__ == '__main__':

    #Load dataframe
    #filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/',title='Select Data File', filetypes = (('DataFrames', '*finaldata.hdf5'),))    
    basepath='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed12_5mm/*'
    #basepath = '/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed_new10mm/*'
    
    names = ['P120_045']
    #names = ['P120_090']
    pathnames = [basepath + name + '_data_finaldata.hdf5' for name in names]
    param = 'omega_k'
    #each path get the 3 files for a particular experiment.
    for path in pathnames:
        
        filenames=get_files_directory(path,full_filenames=True)
    
        #Read dataframe from file
    
    
        filename_op = min(filenames)[:-5]
        if '_040' in filename_op:
            acceleration = 1.9
        elif '045' in filename_op:
            # 2.05
            acceleration = 2.05
        elif '_050' in filename_op:
            acceleration = 2.25
        elif '054' in filename_op:
            # 2.381g
            acceleration = 2.381
        elif '_062' in filename_op:
            acceleration = 2.75
        elif '_069' in filename_op:
            acceleration = 3.01
        elif '_077' in filename_op:
            acceleration = 3.25
        else:
            #0.9
            acceleration = 3.6
                
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
            
            
        for i,filename in enumerate(filenames):
            
            print(filename)
            data = pd.read_hdf(filename)
            diffs, time_bounces = changes(data, filename_op + 'changes', param)

            if param == 'vx_over_r_omegak':
                vals = -data['xVelMM'].values/(5*data['omega_k'].values)
            else:
                print('here')
                print(data[param].std())
                vals = data[param].values
            filename_op2 = filename[:-5]

            #histogram(vals,filename_op2,'individual'+ param)
            
            if i == 0:
                datavals = vals                
                output = np.c_[roughness, acceleration,vals.std(),vals.mean(),vals.max(),vals.min()]
                changevals = diffs
                timevals  = time_bounces
            elif i==1:
                output = np.vstack((output,np.c_[roughness, acceleration,vals.std(),vals.mean(),vals.max(),vals.min()]))
                datavals = np.append(datavals,vals)
                changevals = np.append(changevals,diffs)
                timevals = np.append(timevals,time_bounces)
            else:
                output = np.vstack((output,np.c_[roughness, acceleration,vals.std(),vals.mean(),vals.max(),vals.min()]))
                datavals=np.append(datavals,vals)
                changevals = np.append(changevals, diffs)
                timevals = np.append(timevals, time_bounces)
        print(output)
        print(output[:,2])
        print(output[:,2].std())
        output=np.insert(output,3,output[:,2].std(),axis=1)
        print(output)
        output2 = [np.mean(output,axis=0)]
        print(output2)
        print(filename_op)
        np.savetxt(filename_op + param + 'calcvals.csv', output, fmt='%.3f', delimiter=',', header=" roughness, acceleration, std,+-, mean,+,-")
        np.savetxt(filename_op + param + 'meancalcvals.csv',output2, fmt='%.3f', delimiter=',', header=" roughness, acceleration, std,+-, mean,+,-")
        
        print(np.shape(datavals))
        
                
        histogram(datavals,filename_op,param,range_data=(-2,2),symmetric=False)
        #histogram(changevals, filename_op + '_changes_', param)
        
        #plt.figure()
        #plt.plot(timevals,changevals,'rx')
    plt.show()
        
    
    
    
    
    
    
    