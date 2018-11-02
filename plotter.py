from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tkinter import filedialog
import os
from scipy.signal import argrelextrema

def plotOutData(df,filename,folder='',FrameRate=500):
    
    
    #Extract aggregated data
    xball = clipExtremeData(df.groupby(df.index).mean()['xball'])
    yball = clipExtremeData(df.groupby(df.index).mean()['yball'])
    zball = clipExtremeData(df.groupby(df.index).mean()['zball'])
    xvelball = clipExtremeData(df.groupby(df.index).mean()['xvelball'])
    yvelball = clipExtremeData(df.groupby(df.index).mean()['yvelball'])
    zvelball = clipExtremeData(df.groupby(df.index).mean()['zvelball'])
    omegaiball = clipExtremeData(df.groupby(df.index).mean()['omega_i'],numstd=3)
    omegajball = clipExtremeData(df.groupby(df.index).mean()['omega_j'],numstd=3)
    omegakball = clipExtremeData(df.groupby(df.index).mean()['omega_k'],numstd=3)
    surface = clipExtremeData(df.groupby(df.index).mean()['surface'])
    RadBall = df.groupby(df.index).mean()['radball'].mean()
    frames = df.groupby(df.index).mean().index 
    time = (frames)/FrameRate
    surfacevel = clipExtremeData(df.groupby(df.index).mean()['surfacevel'])
    rolling_ratio = xvelball/(RadBall*omegakball)
    rolling_ratio = clipExtremeData(rolling_ratio.copy())
  
    #Calculate quantities:
    print(time.min())
    print(surface.values.max())
    print(surface.values.min())
    surfZeroVal = surface.mean() 
    yballZero = np.max(yball)
    heightVal = yballZero - yball#This is the change in y relative to the centre of the ball when sitting on the plate at rest.
    surfaceVal = surfZeroVal - surface#This sets the middle of the oscillation to zero.+ve values are upwards
    
    #timetraces xyz
    fig, ax = plt.subplots(3,1)
    ax=createAxis(ax,0,time,xball,'xball')
    ax=createAxis(ax,1,time,heightVal,'height')
    ax=createAxis(ax,2,time,zball,'zball')
    fig.savefig(folder + '/xyz_v_t.png')                       
    
    #timetraces ang vels                
    fig2, ax2 = plt.subplots(3,1)
    ax2=createAxis(ax2,0,time,omegaiball,'wiball')
    ax2=createAxis(ax2,1,time,omegajball,'wjball')
    ax2=createAxis(ax2,2,time,omegakball,'wkball')
    fig2.savefig(folder + '/angvels_v_t.png')   
    
    #ffts                 
    pltFFT(frames,heightVal,xball,surfaceVal,folder,Fs=FrameRate)
                 
    #histograms
    fig3, ax3 = plt.subplots(3,1)
    ax3 = createHistogram(xball,ax3,0,'xballhist')
    ax3 = createHistogram(heightVal,ax3,1,'heighvalhist')#This is the change in height rather than y coord
    ax3 = createHistogram(zball,ax3,2,'zballhist')
    fig3.savefig(folder + '/histsxyz.png')
    
    #histograms
    fig4, ax4 = plt.subplots(3,1,sharex=True)
    
    ax4 = createHistogram(omegaiball,ax4,0,'omegaiballhist')
    ax4 = createHistogram(omegajball,ax4,1,'omegajballhist')
    ax4 = createHistogram(omegakball,ax4,2,'omegakballhist')
    fig4.savefig(folder + '/histsomega.png')
    
    #histograms
    fig5, ax5 = plt.subplots(3,1)
    ax5 = createHistogram(xvelball,ax5,0,'xvelballhist')
    ax5 = createHistogram(yvelball,ax5,1,'yvelballhist')
    ax5 = createHistogram(zvelball,ax5,2,'zvelballhist')
    fig5.savefig(folder + '/histsvel.png')
    '''             
    #Autocorrelations between signals (w is omega_k)
    autocorrxball = signal.correlate(xball, xball, mode='same')/np.size(xball)
    autocorryball = signal.correlate(yball, yball, mode='same')/np.size(yball)
    autocorrzball = signal.correlate(zball, zball, mode='same')/np.size(zball)
    autocorrvxball = signal.correlate(xvelball, xvelball, mode='same')/np.size(xvelball)
    autocorrvyball = signal.correlate(yvelball, yvelball, mode='same')/np.size(yvelball)
    autocorrvzball = signal.correlate(zvelball, zvelball, mode='same')/np.size(zvelball)
    autocorromegaiball = signal.correlate(omegaiball, omegaiball, mode='same')/np.size(omegaiball)
    autocorromegajball = signal.correlate(omegajball, omegajball, mode='same')/np.size(omegajball)
    autocorromegakball = signal.correlate(omegakball, omegakball, mode='same')/np.size(omegakball)
    
    fig6, ax6 = plt.subplots(3,1)
    length=np.size(autocorrxball)
    ax6=createAxis(ax6,0,time[0:int(length/2)],autocorrxball[int(length/2)+1:],'autocorrx')
    ax6=createAxis(ax6,1,time[0:int(length/2)],autocorryball[int(length/2)+1:],'autocorry')
    ax6=createAxis(ax6,2,time[0:int(length/2)],autocorrzball[int(length/2)+1:],'autocorrz')
    fig6.savefig(folder + '/autocorr_xyz.png')   
    
    fig7, ax7 = plt.subplots(3,1)
    ax7=createAxis(ax7,0,time[0:int(length/2)],autocorromegaiball[int(length/2)+1:],'autocorromegai')
    ax7=createAxis(ax7,1,time[0:int(length/2)],autocorromegajball[int(length/2)+1:],'autocorromegaj')
    ax7=createAxis(ax7,2,time[0:int(length/2)],autocorromegakball[int(length/2)+1:],'autocorromegak')
    fig7.savefig(folder + '/autocorr_omega.png')   
    
    fig8, ax8 = plt.subplots(3,1)
    ax8=createAxis(ax8,0,time[0:int(length/2)],autocorrvxball[int(length/2)+1:],'autocorrvx')
    ax8=createAxis(ax8,1,time[0:int(length/2)],autocorrvyball[int(length/2)+1:],'autocorrvy')
    ax8=createAxis(ax8,2,time[0:int(length/2)],autocorrvzball[int(length/2)+1:],'autocorrvz')
    fig8.savefig(folder + '/autocorr_vel.png')   
    '''
    #Plot rolling ratio against time
    fig9, ax9 = plt.subplots(2,1)
    ax9 = createAxis(ax9,0,time,heightVal,'height',symbol='r-')
    ax9 = createAxis(ax9,1,time,rolling_ratio,'rolling_ratio',symbol='b-')
    fig9.savefig(folder + '/rolling_ratio.png')   
    
    mean_xvel = (np.mean(xvelball**2))**0.5
    print('mean speed in x')
    print(mean_xvel)
    
    #magnitude angular velocity
    omega_mag = (omegaiball**2 + omegajball**2 + omegakball**2)**0.5.mean()
    print(omega_mag)
    
    #This function looks at the interaction with the surface
    contactSurface(heightVal,xvelball,yvelball,omegakball,RadBall,surfacevel,time,frames,folder)
    
def clipExtremeData(param,numstd=4):
    medianVal=np.median(param[~np.isnan(param)])
    stdVal = np.std(param[~np.isnan(param)])
    minVal =medianVal - numstd*stdVal
    maxVal = medianVal + numstd*stdVal

    param[(param > maxVal)] = np.nan
    param[(param < minVal)] = np.nan
    return param
    
def createAxis(ax,num,x,y,name,symbol='r-'):
    ax[num].set_title(name)
    ax[num].plot(x,y,symbol)
    return ax

def createHistogram(param,ax,num,name, num_bins='auto'):
    param = param[~np.isnan(param)]
    freq,bin_edges = np.histogram(param, bins = num_bins)
    bins = (bin_edges[:-1] + bin_edges[1:])/2
    ax=createAxis(ax,num,bins,freq,name,symbol='rx')
    return ax
                 
                 
def pltFFT(frames,height,xval,surface,folder,Fs=500):
    # Fs is sampling rate rr frame rate
    Ts = 1.0/Fs; # sampling interval
    t = frames/Fs

    n = len(frames) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range

    FFTsurface = np.fft.fft(surface)/n # fft computing and normalization
    FFTsurface = FFTsurface[range(int(n/2))]
    FFTheight = np.fft.fft(height)/n # fft computing and normalization
    FFTheight = FFTheight[range(int(n/2))]
    #FFTxval = np.fft.fft(xval)/n # fft computing and normalization
    #FFTxval = FFTxval[range(int(n/2))]

    
    fig, ax = plt.subplots(2, 1)
    fig.canvas.set_window_title('Bounce of ball and plate')
    ax[0].plot(t,surface,'b')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[0].plot(t,height,'r')
    
    ax[1].plot(frq[1:],abs(FFTsurface[1:]),'b') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    ax[1].plot(frq[1:],abs(FFTheight[1:]),'r') # plotting the spectrum
    ax[1].set_xlim([0, 80])
    #max1=np.max(FFTsurface[10:])
    #max2=np.max(FFTheight[10:])
    #maxval = 1.5*np.max([max1,max2])
    #ax[1].set_ylim([0,maxval])
    
    fig.savefig(folder + '/fft_heightandsurface.png')

def contactSurface(heightVal,xvelball,yvelball,omegakball,RadBall,surfacevel,time,frames,new_folder,span=5):
    '''
    assumes that contact is made at the minimum of the fitted trajectories. ie when motion goes from down to up
    Extracts time at which contact occurs
    Extracts V_tangential and normal before and afterwards
    Extracts angular vel. before and after
    estimates normal coeff
    estimates tangential coeff
    
    returns values in new dataframe
    '''
    
    #New dataframe for output
    bounce = pd.DataFrame()
    
    #Orient the yvel so that up is positive
    yvel = -yvelball.copy()
    
    
    fig, ax = plt.subplots(5, 1)
    fig.canvas.set_window_title('surface bounce')

    ax[0].plot(time,heightVal,'r-')
    ax[0].plot(time,heightVal,'b.')
    ax[1].plot(time,yvel,'r-')
    ax[1].plot(time,yvel,'b.')
    ax[2].plot(time,surfacevel,'r-')
    ax[2].plot(time,surfacevel,'b.')
        
    #Find minima in height
    args = argrelextrema(heightVal.values,np.less)
    
    maxval= np.max(heightVal.values)
    minval=np.min(heightVal.values)
    maxvelval= np.max(yvel.values)
    minvelval=np.min(yvel.values)
    for val in args:
        ax[0].plot([time[val],time[val]],[minval,maxval],'g--')
        ax[1].plot([time[val],time[val]],[minvelval,maxvelval],'g--')
        ax[2].plot([time[val],time[val]],[minvelval,maxvelval],'g--')
        ax[3].plot([time[val],time[val]],[minvelval,maxvelval],'g--')
    #The frames[0] is related to the fact the first frame may not be 0
    ax[0].plot(time[args[0]],heightVal[args[0]+frames[0]].values,'gx')
    
    ax[3].plot(time,yvel - surfacevel,'r-')
    ax[3].plot(time,yvel - surfacevel,'b.')
    
    
    #ax[4].plot(time,omegakball,'r-')
    #ax[4].plot(time,omegakball,'b.')
    #ax[5].plot(time,xvelball-RadBall*omegakball,'r-')
    #ax[5].plot(time,xvelball-RadBall*omegakball,'b.')
    
    
    
    #Change in relative vertical velocity leading to e normal
    
    #Find minima and maxima in vel
    #Add end point indices to args    
    args2 = np.append([0],args)
    args2 = np.append(args2,yvel.index.max()-1)
    
    minVelArgs=[]
    maxVelArgs=[]
    
    for indice in range(len(args2)-2):
        if indice == 0:
            minVelArgs.append(np.argmin(yvel[0:args2[indice + 1]+frames[0]]))
        else:
            minVelArgs.append(np.argmin(yvel[maxVelArgs[indice-1]:args2[indice + 1]+frames[0]]))
        maxVelArgs.append(np.argmax(yvel[args2[indice+1]+frames[0]:args2[indice + 2]+frames[0]]))
    
    #Velocity before bounce
    velb4 = yvel[minVelArgs]  
    #Velocity after bounce
    velaft = yvel[maxVelArgs]
    
 
    #Annotate maximum and minimum velocities
    ax[1].plot(time[minVelArgs],yvel[minVelArgs],'yx')
    ax[1].plot(time[maxVelArgs],yvel[maxVelArgs],'rx')
    
    
    #Velocity of surface

    #Change in tangential stuff. leading to e tangential
    
    #save new info to bounce dataframe
    #Time at bounce
    timevals = np.array(time[args[0]].values)
    
    #Currently these are both set to be the same.
    surfacevelb4 = surfacevel[args[0]+frames[0]]#surfacevel[minVelArgs]
    surfacevelaft = surfacevel[args[0]+frames[0]]#surfacevel[maxVelArgs]
    
    #annotate points
    ax[2].plot(time[minVelArgs],surfacevelb4,'yx')
    ax[2].plot(time[maxVelArgs],surfacevelaft,'rx')
    
    #mng = plt.get_current_fig_manager()
    #mng.frame.Maximize(True)
    
    d = {'time': timevals,'minVelArgs':minVelArgs,'maxVelArgs':maxVelArgs,'velb4':velb4.values,'velaft':velaft.values,'surfacevelb4':surfacevelb4.values,'surfacevelaft':surfacevelaft.values}
    
    #Store collision data in new dataframe
    bounce = pd.DataFrame(data=d,index=frames[args[0]])
    bounce['rel_vaft']=(bounce['velaft']-bounce['surfacevelaft'])
    bounce['rel_vb4']=(bounce['velb4']-bounce['surfacevelb4'])
    bounce['e_normal']=-bounce['rel_vaft']/bounce['rel_vb4']
    
    ax[4].plot(bounce['time'],bounce['e_normal'])
    
    manager = plt.get_current_fig_manager()
    manager.resize(*manager.window.maxsize())
    plt.savefig(new_folder + '/bouncingSurface.png')
    print(bounce)
    
    
if __name__ == "__main__":
    filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBallData/HighSpeedBalls/',title='Select Data File', filetypes = (('Datafile', '*analysed.hdf5'),))    
    ball = pd.read_hdf(filename)
    new_folder = filename[:-14] + '_plots'
    print(new_folder)
    try:
        os.mkdir(new_folder) #Make directory to store fit graphs
      
    except:
        pass
    plotOutData(ball,filename,folder=new_folder)