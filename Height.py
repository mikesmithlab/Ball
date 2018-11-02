from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tkinter import filedialog
import os
from scipy.signal import argrelextrema
from scipy import optimize

def createHistogram(ball,filename,binnum=100):
    plt.figure()
    file = filename[:-5]+'.txt'

    ball['Height'].hist(bins=binnum)
    
    
    surfmax = [ball['surface'].max(),ball['surface'].max()]
    
    count,binedges = np.histogram(ball['Height'],bins=binnum)
    countmax = [0,count.max()]
    bincentres = (binedges[:-1] + binedges[1:])/2
    plt.plot(surfmax,countmax,'r--')
    np.savetxt('myfile.txt', np.c_[bincentres,count])
    plt.savefig(file[:-4] + '.png')



def surfaceScale(ball,FrameRate=500,accelerationmm = 33000):
    #Get values in pixels of surface motion
    minimumSurfVal,meanSurfVal,MaximumSurfVal=plotFitSurface(ball[ball['frame']<500])
    #Need to use the actual amplitude to scale the surface motion.
    Amplitude = accelerationmm/(2*np.pi*50)**2
    
    print('Amplitude pixels')
    print(abs(minimumSurfVal - MaximumSurfVal)/2)
    
    ball['surface'] = -(ball['surface']-meanSurfVal)
    scale_surface = Amplitude/abs(MaximumSurfVal-meanSurfVal)
    
    
    ball['surface']=ball['surface']*scale_surface
    return ball

def ballScale(ball,FrameRate=500,RadBallInMM=5):
    #Define new yball in terms of height above the mean surface position, which is the optical axis
    ball['Height']=-(ball['yball'] - ball['surface'].mean())-ball['radball']   
    ball['ballscale']=RadBallInMM/ball['radball']
    ball['Height']=ball['Height']*ball['ballscale']
    print(ball['Height'].max())
    print(ball['Height'].mean())
    print(ball['Height'].min())
    return ball


def sin_f(x, A,B, C, D): # this is your 'straight line' y=f(x)
    
    return A*np.sin(B*x + C) + D

def plotFitSurface(ball):
    drivingF = 50
    camFPS = 500
    dataLength = np.shape(ball.index.unique())[0]
   
    
    omega = 2*np.pi*(drivingF)/camFPS
    #frames = ball.groupby(by='frame').mean().index.values

    surfacedata = (ball.groupby(by='frame').mean()['surface'])#-ball.groupby(by='frame').mean()['surface'].mean())
    frames = surfacedata.index
    params,SD = optimize.curve_fit(sin_f,frames,surfacedata,bounds=([-np.inf,omega*0.999,-np.inf,0],[np.inf,omega*1.001,np.inf,1000]))
    frame_fine = np.arange(0,dataLength,0.01)
    surface = sin_f(frame_fine,params[0],params[1],params[2],params[3])
                   
    if False:
        plt.figure()
        plt.plot(frames,surfacedata,'bx')
        plt.plot(frame_fine,surface,'r-')
        plt.show()
    
    minimumSurfVal = np.min(surface)
    maximumSurfVal = np.max(surface)
    meanSurfVal = np.mean(surface)
     
    return (minimumSurfVal,meanSurfVal,maximumSurfVal)
    
def plotVar(ball,value,file='',maxVal=10000,show=False,save=True,):
    plt.figure()
    frames = ball.groupby('frame').mean().index
    Variable = ball.groupby('frame').mean()[value]
    plt.plot(frames[frames < maxVal],Variable[frames < maxVal],'rx')
    plt.plot(frames[frames < maxVal],Variable[frames < maxVal],'b-')
    if save:
        plt.savefig(file + value +'.png')
    if show:
        plt.show()

    
if __name__ == "__main__":
    
    filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/Processed Data/',title='Select Data File', filetypes = (('DataFrames', '*.hdf5'),))    
    print(filename)
    ball = pd.read_hdf(filename)
    
    ball = ballScale(ball,FrameRate=500,RadBallInMM=5)
    ball = surfaceScale(ball)
    plotVar(ball,'Height',file=filename,show=False)
    plotVar(ball,'radball',file=filename,show=False)
  
    createHistogram(ball,filename)
    
    
    