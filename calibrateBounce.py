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

from scipy.signal import argrelextrema





def calibrate_values(ball,FrameRate=500,RadBallInMM=0,AccVolt=0):
    
    #calibrate motion of surface.
    if AccVolt == 0.4:
        #1.9g
        Amp = 0.1888
    elif AccVolt == 0.5:
        #2.2g
        Amp = 0.2187
    elif AccVolt == 0.62:
        #2.75g
        Amp = 0.2733
    elif AccVolt == 0.77:
        #3.25g
        Amp = 0.3230
    else:
        print('Acceleration not recognised')
    
    surface = ball.groupby(by='frame').mean()['surface']
    
    #When the surface is in the middle of travel its height is zero.The pixel 0,0 is at top left of images so increasing ball['surface'] is decreasing height
    #Calibrate this motion using the amplitude measured using accelerometer.
    surfaceHeight = -(surface-surface.mean())
    maxSurfaceHeight =surfaceHeight.max()
    surfaceZero = surfaceHeight.mean()
    minSurfaceHeight = surfaceHeight.min()
    
    scaleSurface = 2*Amp/(maxSurfaceHeight - minSurfaceHeight)
    
    calibValues = pd.DataFrame()
    calibValues['surfaceHeightMM'] = surfaceHeight*scaleSurface
    
    ballScale = RadBallInMM/ball.groupby(by='frame').mean()['radball']
    centreIm = 640
    calibValues['ballScale']=ballScale
    print('surfaceZero')
    print(surfaceZero)
    print(ball.groupby(by='frame').mean()['yball'].mean())
    calibValues['ballHeightMM'] = (surface.mean() - ball.groupby(by='frame').mean()['yball'])*ballScale
    calibValues['yvel'] = -ball.groupby(by='frame').mean()['yvelball']*ballScale*FrameRate
    calibValues['ballXMM']= (ball.groupby(by='frame').mean()['xball'] - 640)*ballScale
    calibValues['xvel'] = ball.groupby(by='frame').mean()['xvelball']*ballScale*FrameRate
    calibValues['radballPx'] = ball.groupby(by='frame').mean()['radball']
    calibValues['radballMM'] = RadBallInMM*ball.groupby(by='frame').mean()['radball']/ball.groupby(by='frame').mean()['radball']
    
    
   
    
    #All angles do not require any scaling
    calibValues['thetas']=ball.groupby(by='frame').mean()['thetas']
    calibValues['dthetas']=ball.groupby(by='frame').mean()['dthetas']
    calibValues['phis']=ball.groupby(by='frame').mean()['phis']
    calibValues['dphis']=ball.groupby(by='frame').mean()['dphis']
    calibValues['omega_i']=ball.groupby(by='frame').mean()['omega_i']
    calibValues['omega_j']=ball.groupby(by='frame').mean()['omega_j']
    calibValues['omega_k']=ball.groupby(by='frame').mean()['omega_k']
    calibValues['time']=ball.groupby(by='frame').mean()['time']
    calibValues['frame']=calibValues.index
    
    return calibValues

def addBounceCol(ball):
    #Adds 2 logic columns which identify bounces and peaks
    #A bounce is a minimum

    x = ball['ballHeightMM'].values
    xindices = ball['ballHeightMM'].index
    
    minima = np.zeros(np.shape(x))
    maxima = np.zeros(np.shape(x))
    minimaIndices = argrelextrema(x, np.less)
    maximaIndices = argrelextrema(x, np.greater)

    
    ball['bounce'] = False
    ball['bounce'][xindices[minimaIndices[0]]]= True
    
    ball['peak'] = False
    ball['peak'][xindices[maximaIndices[0]]]= True
    
    return ball

def plotVar(ball,value,file='',minVal=0,maxVal=10000,symbol=False,invertY=False,show=True,save=False,):
    plt.figure(value)
    
    frames = ball.groupby('frame').mean().index
    Variable = ball.groupby('frame').mean()[value]
    xdata = frames[(frames>minVal)&(frames < maxVal)]
    ydata = Variable[(frames>minVal)&(frames < maxVal)]
    if invertY ==True:
        ydata = -ydata
    if symbol == False:
        plt.plot(xdata,ydata,'rx')
        plt.plot(xdata,ydata,'b-')
    else:
        plt.plot(xdata,ydata,symbol)
    if save:
        plt.savefig(file + value +'.png')
    if show:
        if False:
            plt.show()

if __name__ == "__main__":
    #Load dataframe
    filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/Processed Data/',title='Select Data File', filetypes = (('DataFrames', '*_ballSmoothed.hdf5'),))    
    #Read dataframe from file
    ball = pd.read_hdf(filename)
    #Plot variables
    n=1
    #plotVar(ball[ball['particle']==1],'omega_k')
    plotVar(ball[ball['particle']==n],'yball',invertY=True)
    plotVar(ball[ball['particle']==n],'xball')
    #plotVar(ball[ball['particle']==n],'omega_k')
    plotVar(ball[ball['particle']==n],'radball',invertY=True)
    plotVar(ball[ball['particle']==n],'x')
    plotVar(ball[ball['particle']==n],'y')
    
    
    plt.show()
    #Calibrate all lengths and velocities to mm and mms^-1
    
    b_out = calibrate_values(ball,RadBallInMM = 10, AccVolt = 0.77)
    #plotVar(b_out,'ballHeightMM')
    
    
    b_out = addBounceCol(b_out)
    #print(b_out.head())
    #print(np.shape(b_out))
    #plotVar(b_out,'ballHeightMM',maxVal=250)
    #plotVar(b_out[b_out['bounce']],'ballHeightMM',maxVal=250,symbol='go')
    #plotVar(b_out[b_out['peak']],'ballHeightMM',maxVal=250,symbol='yo')
    #plt.show()
    #save modifed dataframe to new file
    op_name = filename[:-22] + '_ballCalibrated.hdf5'
    #print(op_name)
    b_out.to_hdf(op_name,'data')     
   
    print('analysis complete')    
    
    