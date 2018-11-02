import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import spline
import trackpy as tp
from scipy import optimize
import scipy.fftpack

import os


def preProcessData(ball,max_displacement = 20,memory_val=5,minTrajectory=9):
    '''
    This function takes a dataFrame with unsorted data and links the trajectories
    We then filter any trajectories that are too short to be useful and return the 
    updated dataFrame.
    
    Inputs: 
        ball - DataFrame with unsorted particle coordinates
        max_displacement - max distance in pixels a spot could have moved between frames
        memore_val - Number of frames a spot can be missing for
        minTrajectory - Shortest useful trajectory to keep
    
    Outputs:
        ball - dataFrame which now includes a particle number for each unique 
        trajectory and the spurious dots removed. spurious = not found for more than minTrajectory
        frames.
    '''
    
    tp.link_df(ball,max_displacement,memory=memory_val)
        
    #remove those dots that are only measured in a single frame
    ball=tp.filter_stubs(ball,threshold=minTrajectory)
    
    
    
    #ball.drop('frame',axis=1,inplace=True)
   
    #print(ball.head(n=20))
    
    return ball

def removeStaticPoints(ball,deviationY=5,deviationX=5):
    #There were some dots on the cell boundary which were sometimes tracked. These dots don't move in the
    #x direction and have y values which match the surface with some offset. Remove any tracks that move less than deviation during lifetime.
    #particle here means a dot on the ball
    
    particle_ids = ball.particle.unique()
    
    plt.figure(1)
    plt.figure(2)
    for id in particle_ids:
        
        condition1 = np.std(ball[ball['particle']==id]['y'] - ball[ball['particle']==id]['surface']) < deviationY
        
        condition2 = np.std(ball[ball['particle'] == id]['x']) < deviationX
        #print(conditionDrop)
        if condition1 and condition2:
            #Find rows of that trajectory

            indices=ball[ball['particle']==id].index
            plt.figure(2)
            plt.plot(ball[ball['particle']==id].x,ball[ball['particle']==id].y,'-')
            print(np.shape(ball))
            ball = ball[ball['particle'] != id]
            print(np.shape(ball))
        else:
            plt.figure(1)
            plt.plot(ball[ball['particle']==id].x,ball[ball['particle']==id].y,'-')

    #particle_ids = ball.particle.unique()
 
    
    plt.figure(1)
    plt.savefig(os.path.join('/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/dynamic'+'.png'))
    plt.figure(2)
    plt.savefig(os.path.join('/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/static'+'.png'))
    
    return ball


    
    
    return ball
   
if __name__ == '__main__':
    plt.close('all')
    
    #Load dataframe
    filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/RawDataandTracking/',title='Select Data File', filetypes = (('DataFrames', '*data.hdf5'),))    
    new_folder = filename[:-5] + '_analyse'
    try:
        os.mkdir(new_folder) #Make directory to store fit graphs
    except:
        pass
    
    #Read dataframe from file
    ball = pd.read_hdf(filename)
    
    
    
    #process data - link trajectories and remove spurious unlinked particles. 
    #ball is a dataframe of linked dot trajectories
    ball = preProcessData(ball)
    
    
    
    
    dotlist = ball.particle.unique().flatten()   
    #Remove marks on cell which only move with the cell and are nothing to do with the ball.
    #ball=removeStaticPoints(ball)    
    
    newdotlist = ball.particle.unique().flatten()   
    
    #save modifed dataframe to new file
    ball.to_hdf(filename[:-5] + '_initialprocessed.hdf5','data')     
    
    
    removedDots = [dot for dot in dotlist if dot not in newdotlist ]
    
    print(np.shape(dotlist)[0] - np.shape(newdotlist)[0])
    
    print(removedDots)
    print('processing complete')    
    
    

