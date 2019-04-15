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
    print(np.shape(values))

    histogram(values,filename,param)   
    

def changes(ball,filename,param,exclude_wall = True):
    'works out how a parameter changes due to a bounce, Excludes those points near a wall'
    bounce_indices = ball[(ball['bounce']==True)].index
    bounce_indices_filtered = ball[(ball['bounce']==True)&(ball['wall']==False)].index
    ix = np.isin(bounce_indices,bounce_indices_filtered)
    
    indices = np.where(ix)
    differences = np.diff(ball[np.isin(ball.index,bounce_indices[indices])][param])
    
    histogram(differences,filename,param)
    if True:
        
        plt.figure(param)
        plt.plot(bounce_indices[indices][:-1],differences,'bx')
        for bounce_index in bounce_indices:
            plt.plot([bounce_index,bounce_index],[differences.min(),differences.max()],'g--')
        for bounce_index in bounce_indices_filtered:
            plt.plot([bounce_index,bounce_index],[differences.min(),differences.max()],'r-')
        plt.show()
    return differences

def histogram(values,filename,param,num_bins=14,range_data=False,save=True):
    print('Histogram: (num_bins,(min,max))')
    print(num_bins,(values.min(),values.max()))
    if range_data == False:
        freq,bin_edges = np.histogram(values, bins = num_bins)
    else:
        freq,bin_edges = np.histogram(values, bins = num_bins,range=range_data)
    bins = (bin_edges[:-1] + bin_edges[1:])/2
    plt.figure(param + '_histogram')
    plt.plot(bins,freq,'r-')
    plt.plot(bins,freq,'bx')
    if save:
        with open(filename_op + param + '.csv','w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(zip(bins, freq))
    
def calcBasics(ball,param):
    meanVal = ball[param].mean()
    stdVal = ball[param].std()
    minVal = ball[param].min()
    maxVal = ball[param].max()
    medianVal = ball[param].median()
    magnmean = ball[param].abs().mean()
    
    
    
    
    
    #print(param + ':(mean,std,min,max,median,magnmean)')
    #print([meanVal,stdVal,minVal,maxVal,medianVal,magnmean])
    
    #Find those bounces that are followed by a wall
    
    
    
    #print(ball[ball['bounce']==True][param])
    values=np.diff(ball[ball['bounce']==True][param])
    indices=ball[ball['bounce']==True][param].index
    #print(values)
    
    if param == 'ballHeightMM':
        return [str(meanVal),str(minVal),str(maxVal),str(maxVal+minVal),str(meanVal- (maxVal - minVal)/2)]
    else:
        return [meanVal,stdVal,medianVal,minVal,maxVal,magnmean]
    
    
        
    
    
    
if __name__ == '__main__':

    #Load dataframe
    #filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/',title='Select Data File', filetypes = (('DataFrames', '*finaldata.hdf5'),))    
    basepath='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/finalProcessed/*'
    
    names = ['800_077','800_062','800_050','800_040','400_077','400_062','400_050','400_040','240_077','240_062','240_050','240_040','120_077','120_062','120_050','120_040','80_077','80_062','80_050','80_040'] 
    pathnames = [basepath + name + '_data_finaldata.hdf5' for name in names]
    for path in pathnames:
        
        filenames=get_files_directory(path,full_filenames=True)
    
        #Read dataframe from file
    
    
        
        for filename in filenames:
            print(filename)
            data = pd.read_hdf(filename)
    
            filename_op = filename[:-5]
    
    
            c=calcBasics(data[data['bounce'] == True],'ballHeightMM')
            print(c)
            #print(values)
            #write_row_to_csv(filenames[1][:-5]+'height.csv',values,append=True)
            #print(filenames[1][:-5]+'height.csv')
            #print(a + b + c + d + e)
            #param = 'ballHeightMM'
            #plotWithBounces(data,param)
            createHistogram(data,param,filename_op)
            #changes(data,filename_op+'changes',param)
            #plt.show()
    
    
    
    
    
    
    