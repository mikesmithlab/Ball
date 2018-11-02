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
from scipy.signal import argrelextrema
import csv
from os import listdir
from os.path import isfile, join
from OSFile import get_files_directory


import os


def smooth(x_series,window_len=0,window='bartlett',show=False):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an even integer. Set to 0 as default
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    name_val = x_series.name
    x = x_series.values
    x_index = x_series.index
    
    if window_len>3:
        s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')

        y=np.convolve(w/w.sum(),s,mode='valid')

        new_y1 = y[int(window_len/2-1):-int(window_len/2)]
        new_y2 = y[int(window_len/2-1)+1:-int(-1+window_len/2)]
        new_y = 0.5*(new_y1 + new_y2)

        #Pack back into pd.Series
        smoothed_vals = pd.Series(new_y,name=name_val,index=x_index)

        if show: 
            plt.figure(name_val)
            plt.plot(x_series[x_series.index < 200].index,x_series[x_series.index < 200],'rx')
            plt.plot(smoothed_vals[smoothed_vals.index < 200].index,smoothed_vals[smoothed_vals.index < 200],'b-')
        return smoothed_vals
    
    else:
        #If window_len < 3 leave unchanged
        return x_series

def findSurfaceZero(data):
    return data
    
def smoothBallTrajectory(ball,filename,show=False,no_bounces=False):
    '''
    ballTrajectory takes a dataframe 
    It uses smoothing to reduce the noise in the radius value data .
    
    The logic here is that each dot appears for a fraction of whole movie.
    The value of xball and yball for a given frame is the same for each dot.
    However, spline fits are often worse near end so if we fit short section of data
    the fit will be calculated differently. We therefore work out the motion of the ball 
    for the whole movie and then in the second bit of the function piecewise add the 
    smoothed result (smoothed data) back for each dot on the ball in turn.
    '''
    
    if 'P80' in filename:
        surfaceZero = 384.7
    elif 'P120' in filename:
        surfaceZero = 387.7
    elif 'P240' in filename:
        surfaceZero = 412.7
    elif 'P400' in filename:
        surfaceZero = 394.9
    else:
        surfaceZero = 416.3
    
    
    indices = ball.groupby(by='frame').mean().index.copy()
    xdata = ball.groupby(by='frame').mean()['xball'].copy()
    ydata = ball.groupby(by='frame').mean()['yball'].copy()
    rdata = ball.groupby(by='frame').mean()['radball'].copy()
 
    #x and y are not being smoothed
    xvals = smooth(xdata,show=show)
    yvals = smooth(ydata,show=show)
    radvals = smooth(rdata,window_len=12,show=show)
    
    
    surfacedata = ball.groupby(by='frame').mean()['surface'].mean()
    #Detects where the ball bounces and reaches a peak
    b=addBounceCol(ball.groupby(by='frame').mean().copy(),no_bounces=no_bounces)  
    b2=addNearWallCol(ball.groupby(by='frame').mean().copy())
    
    ball['xball']=xvals
    ball['yball']=yvals
    
    
    ball['radball']=radvals
    #ball['surface']=surfacedata
    ball['bounce']=b['bounce']
    ball['peak']=b['peak']
    ball['wall']=b2['wall']
    #surfaceZero = ball[ball['bounce']==True]['yball'].mean()
        
    ball['yball'] = surfaceZero - ball['yball'] # flip values so positive y is up and relative to surface
    #All values are in pixels at this point.

    return ball

def dotCalc(ball_track,folder,no_bounces=False,FrameRate=500.0,RadBallInMM=5.0,show=False):
    '''
    Calculate the relative positions of a single dot to centre
    Sign convention and labels phi and theta follows diagram here: http://dynref.engr.illinois.edu/rvs.html#rvs-ew-d
    Z is defined out of the image towards camera. X is horiztonal and to the right. Y is vertical upwards
    The data is smoothed.
    
    inputs:
        ball_track = pandas dataframe of a single dot and ball centre location  ['x','y'.'xball','yball']
        particle = the id of the dot
        RadBall = radius of ball in pixels
    
    output = dataframe with additional columns ['Relx','Rely','Relz','theta','phi']
    ''' 
    colorval=(np.random.rand(1)[0],np.random.rand(1)[0],np.random.rand(1)[0])
    
    ball_track.loc[:,'x'] = smooth(ball_track.loc[:,'x'],window_len=0,show=False)
    ball_track.loc[:,'y'] = smooth(ball_track.loc[:,'y'],window_len=0,show=False)
    
    #plt.figure('xy')
    #plt.plot(ball_track[ball_track.index < 300]['x'],ball_track[ball_track.index < 300]['y'],color=colorval)
    #plt.plot(ball_track[ball_track.index < 300]['xball'],ball_track[ball_track.index < 300]['yball'],color=colorval)
    
    #Theta and phi are defined as shown here:http://dynref.engr.illinois.edu/rvs.html#rvs-ew-d 
    #Calc relative coordinates   
    ball_track['Relx'] = ball_track.loc[:,'x'] - ball_track.loc[:,'xball']
    ball_track['Rely'] = ball_track.loc[:,'y'] - ball_track.loc[:,'yball']
    
    if show:
        plt.figure('Rel')
        plt.plot(ball_track[ball_track.index < 200]['Relx'],ball_track[ball_track.index < 200]['Rely'],color=colorval)
        ax=plt.gca()
        ax.set_ylim((-160,160))
        ax.set_xlim((-160,160))
    
    #Calculate relz using equation of sphere. z must be positive for dot to be visible
    ball_track['Relz']= np.absolute((ball_track['radball']**2 - ball_track['Relx']**2 - ball_track['Rely']**2)**0.5)
    #Calc phi
    ball_track['phi'] = pd.Series(np.arccos((ball_track['Relz'])  /  ball_track['radball'] ) )
    #Calculate theta values
    ball_track['theta'] = pd.Series(np.arctan2(ball_track['Rely'],ball_track['Relx']))
    #Shift theta to make it a continuous function with no jumps from -pi to pi.
    ball_track['thetashift'] = thetaShift(ball_track['theta'])
    #Smooth thetashift and phi
    ball_track.loc[:,'thetashift'] = smooth(ball_track.loc[:,'thetashift'],window_len=4,show=show_plots)
    ball_track.loc[:,'phis'] = smooth(ball_track.loc[:,'phi'],window_len=4,show=show_plots)

    if show_plots:
        plotWithBounces(ball_track,'thetashift')
        plotWithBounces(ball_track,'phis')
        plotWithBounces(ball_track,'xball')
        plotWithBounces(ball_track,'yball')
        
    #Calibrate xball and yball using the ball radius as scale
    ballScale = RadBallInMM/ball_track['radball']

    
    ball_track['ballHeightMM'] = ball_track['yball']*ballScale
    ball_track['ballXMM']= (ball_track['xball'] - 640)*ballScale
    
    ball_track['radBallMM']=RadBallInMM
    
    
    
    #These values have constant gradients between each bounce
    ball_track=calcGradient(ball_track,'thetashift','dthetas')
    ball_track=calcGradient(ball_track,'phis','dphis')
    ball_track=calcGradient(ball_track,'ballXMM','dxballMM')
    #yball takes the running gradient
    ball_track['dyballMM'] = np.gradient(ball_track['ballHeightMM'])
    
    #ball_track.to_csv('/media/ppzmis/data/BouncingBall_Data/newMovies/RawDataandTracking/Examples/test.csv')    
    
    if show_plots:
        plotWithBounces(ball_track,'Relz')
        plotWithBounces(ball_track,'Relx')
        plotWithBounces(ball_track,'Rely')
        plotWithBounces(ball_track,'dphis')
        plotWithBounces(ball_track,'dxballMM')
        plotWithBounces(ball_track,'dyballMM')
    return ball_track

def plotWithBounces(ball_track, param, data_length=1000):
    
    bounce_indices = ball_track.index[np.where((ball_track['bounce'] == True)&(ball_track.index < data_length))]#np.where(ball_track[ball_track.index < data_length]['bounce'] == True)
    near_wall_indices = ball_track.index[np.where((ball_track['wall'] == True)&(ball_track.index < data_length))]#np.where(ball_track[ball_track.index < data_length]['bounce'] == True)
    plt.figure(param)
    plt.plot(ball_track[ball_track.index < data_length].index,ball_track[ball_track.index < data_length][param],'bx')
    for bounce_index in bounce_indices:
        plt.plot([bounce_index,bounce_index],[ball_track[ball_track.index < data_length][param].min(),ball_track[ball_track.index < data_length][param].max()],'g--')
    for near_wall_index in near_wall_indices:
        plt.plot([near_wall_index, near_wall_index],[ball_track[ball_track.index < data_length][param].min(),ball_track[ball_track.index < data_length][param].max()],'r--')
    

def calcGradient(ball_track,param,grad_param):
    bounce_indices = np.where(ball_track['bounce'] == True)
    
    #Add the 1st and last index to the bounce_indices array
    bounce_idx=np.append(np.array([[ball_track.index.min()]]),ball_track.index[bounce_indices])
    bounce_idx=np.append(bounce_idx,np.array([[ball_track.index.max()]]))
    
    frame_gaps_bounces = np.diff(ball_track.loc[bounce_idx,param].index)
    #print(frame_gaps_bounces)
    grad_vals=np.diff(ball_track.loc[bounce_idx,param])/frame_gaps_bounces

    ball_track[grad_param] = 0
    for i in range(np.shape(bounce_idx)[0]-1):
        ball_track.loc[(ball_track.index >= bounce_idx[i])&(ball_track.index < bounce_idx[i+1]),grad_param] = grad_vals[i]/frame_gaps_bounces[i]
    ball_track.loc[bounce_idx[-1],grad_param] = grad_vals[-1]/frame_gaps_bounces[-1]
    return ball_track
        
def thetaShift(angle):
    '''
    Shifts theta values to make theta continuous. ie no jump from pi to -pi
    inputs:
        angle - angle is a Pandas Series of all the angles of a single spot
        
        
    outputs:
        returns the adjusted angles as a Pandas Series
    '''
    
    angle2 = angle.copy()
    theta_diff = angle2 - angle2.shift(1)
    shift_down = theta_diff[theta_diff >= 1.5*np.pi]
    shift_up = theta_diff[theta_diff < -1.5*np.pi]
    
    for i in range(np.shape(shift_down)[0]):
        angle2[shift_down.index[i]:] = angle2[shift_down.index[i]:] - 2*np.pi

    for i in range(np.shape(shift_up)[0]):
        angle2[shift_up.index[i]:] = angle2[shift_up.index[i]:] + 2*np.pi
    
    return angle2

def addBounceCol(ball,no_bounces=True):
    #Adds 2 logic columns which identify bounces and peaks
    #A bounce is a minimum
    x = ball['yball'].values
    xindices = ball['yball'].index
    #print(xindices[0])
    if no_bounces == True:
        maximIndices = np.arange(xindices.min()+5,xindices.max()-5,10)
        minimaIndices = np.arange(xindices.min(),xindices.max(),10)
    else:
        minimaIndices = argrelextrema(x, np.less)
        maximaIndices = argrelextrema(x, np.greater)

    ball['bounce'] = False
    ball['bounce'][xindices[maximaIndices[0]]]= True

    
    ball['peak'] = False
    ball['peak'][xindices[minimaIndices[0]]]= True
    
    return ball

def addNearWallCol(ball,edge = (200,1050)):
    #Adds logic column which identifies if xpos is near edge
    #True if near wall
    radius = ball['radball'].max()
    
    x = ball['xball'].values
    xindices = ball['xball'].index
    
    wallIndices = np.where((x < edge[0] + radius) | (x > edge[1] - radius))
    
    ball['wall'] = False
    ball['wall'][xindices[wallIndices[0]]]= True
    
    return ball

def calc_aggregate_quantities(b,FrameRate=500.0,maxZ = 10.0):
    '''calculates the aggregated quantities and produces new df with just the 
    calibrated info in it
    '''
    data = pd.DataFrame()
    #print(b.groupby(by='frame').mean().head(n=20))
    #print(b.groupby(by='frame').mean()['ballHeightMM'].head(n=20))
    data['bounce'] = b.groupby(by='frame').mean()['bounce']
    data['peak'] = b.groupby(by='frame').mean()['peak']
    data['wall'] = b.groupby(by='frame').mean()['wall']
    data['radBallMM'] = b.groupby(by='frame').mean()['radBallMM']
    data['surfaceHeightMM'] = b.groupby(by='frame').mean()['surfaceHeightMM']
    data['ballHeightMM'] = b.groupby(by='frame').mean()['ballHeightMM']
    
    #Correct Height for wet and dry.
    data['ballHeightMM'] = data['ballHeightMM'] - (data[data['bounce']==True]['ballHeightMM'] - data[data['bounce']==True]['surfaceHeightMM']).mean()
    
    
    data['ballXMM'] = b.groupby(by='frame').mean()['ballXMM']
    data['yVelMM'] = b.groupby(by='frame').mean()['dyballMM']*FrameRate
    data['xVelMM'] = b.groupby(by='frame').mean()['dxballMM']*FrameRate
    #Use Median value of angular velocities to eliminate effect of 1 dodgy tracked point
    data['omega_i'] = (-b.groupby(by='frame').median()['dphis']*FrameRate*np.sin(b.groupby(by='frame').mean()['thetashift']))
    data['omega_j'] = b.groupby(by='frame').median()['dphis']*FrameRate*np.cos(b.groupby(by='frame').mean()['thetashift'])
    data['omega_k'] = b.groupby(by='frame').median()['dthetas']*FrameRate
    
    maxR = b.groupby(by='frame').mean()['radball'].max()
    minR = b.groupby(by='frame').mean()['radball'].min()

    zScale = (maxZ)/(maxR - minR)
    data['ballZMM'] = zScale*(ball.groupby(by='frame').mean()['radball'] - minR) - maxZ/2 # z=0 is the middle of the cell
    data=calcGradient(data,'ballZMM','zVelMM')
    
    print(data['ballHeightMM'].mean())
    return data

def calibrate_surface(ball,AccVolt=0):
    
    #calibrate motion of surface.
    #These values are obtained from an average of the mean height of 
    if '040' in AccVolt:
        #1.9g
        Amp = 0.1888
    elif '050' in AccVolt:
        #2.2g
        Amp = 0.2187
    elif '062' in AccVolt:
        #2.75g
        Amp = 0.2733
    elif '077' in AccVolt:
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
    #print((maxSurfaceHeight - minSurfaceHeight)/2)
    scaleSurface = 2*Amp/(maxSurfaceHeight - minSurfaceHeight)
    ball['surfaceHeightMM'] = surfaceHeight*scaleSurface
    
    return ball


   
if __name__ == '__main__':
    #Set to True if the ball doesn't bounce. Analysis will assume a mean bounce time of 10 frames.
    no_bouncing = False
    show_plots = False
    #smoothing window length
    window_length = 12
    
    #Load dataframe
    #filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/InitProcessed/',title='Select Data File', filetypes = (('DataFrames', '*040_data_initialprocessed.hdf5'),))    
    path = '/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/InitProcessed/*initialprocessed.hdf5'#*.initialprocessed.hdf5'
    
    file_list = get_files_directory(path)
    for filename in file_list:
        
        #'/media/ppzmis/data/BouncingBall_Data/newMovies/ProcessedData/InitProcessed/P80_062_data_initialprocessed.hdf5'
    
    
        
        new_folder = filename[:-22] + '_analyse'
        try:
            os.mkdir(new_folder) #Make directory to store fit graphs
        except:
            pass
    
        #Read dataframe from file
        ball = pd.read_hdf(filename)

        
        
        #Calibrate surface height
        ball = calibrate_surface(ball,AccVolt=filename)
        
        #smooth Radius ball to reduce noise. Returned vals in pixels
        ball = smoothBallTrajectory(ball.copy(), filename,show=show_plots)
        #smooth tracks and calculate derived quantities such as displacements and angles
        #list of tracks
        dots = ball.particle.unique()
        b=pd.DataFrame()
        #For each dot on the ball
    
        for dot in dots:   
            #Only use trajectories greater than or equal to 9 in length
            if np.shape(ball[ball['particle']==dot][:])[0] >=window_length:
                b = pd.concat([b,dotCalc(ball[ball['particle']==dot][:],new_folder,no_bounces=no_bouncing,show=show_plots)])
        #Calibrated columns all have MM in the name, 'ballHeightMM''ballXMM''surfaceHeightMM', . 
        #All other values are in radians or pixels at this point
    
        #
        #Calculate the linear and angular velocities and return simplified dataframe with only calibrated values
    
        data = calc_aggregate_quantities(b)
        print(filename)
        print(data['ballHeightMM'].mean())
    
        plotWithBounces(data, 'ballHeightMM')
        #plotWithBounces(data,'ballZMM')
        # plotWithBounces(data, 'ballXMM')
        #plotWithBounces(data, 'xVelMM')
        #plotWithBounces(data, 'yVelMM')
        #    plotWithBounces(data, 'zVelMM')
        #plotWithBounces(data, 'omega_i')
        #plotWithBounces(data, 'omega_j')
        #plotWithBounces(data, 'omega_k')
        #plotWithBounces(data, 'surfaceHeightMM')
        plt.show()
    
       
    

    
        #save calibrated and aggergated dataframe to new file
        op_name = filename[:-22] + '_finaldata.hdf5'
        #print(op_name)
        #data.to_hdf(op_name,'data')     
    
        #plt.show()
        print('analysis complete ')    
    
