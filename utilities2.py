import cv2
import numpy as np
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt

def extractPortionVid(filename,startVal = 0,stopVal = 0):
    #Load Video
    print(filename)
    original = cv2.VideoCapture(filename)
    width = original.get(3)
    height = original.get(4)
    numbFrames = original.get(7)
    
    #Determine where to save output
    filename_output = filename[:-14] + str(startVal) + '_'+str(stopVal) +'_annotated.avi'
    print(filename_output)
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    op_file = cv2.VideoWriter(filename_output,fourcc,30.0,(int(width),int(height)),True)
    
    
    for n in range(1,int(numbFrames),1):
        
        ret, img = original.read()
        if (n >= startVal)&(n<stopVal):
            op_file.write(img)
        if n == stopVal:
            break
        
    #Clean up the video stuff
    original.release()
    op_file.release()


    
def extractPortionDFrame(filename,startVal = 0,stopVal = 0):
    #Read dataframe from file
    ball = pd.read_hdf(filename)
    output = ball[(ball['frame'] >= startVal) & (ball['frame'] < stopVal)]
    print(output)
    filename_output = filename[:-5] + str(startVal) + '_'+str(stopVal) +'_data.hdf5'
    output.to_hdf(filename_output,'ball')
    
def combinePortionsDFrame(filename3,filename4):
    '''
    
    '''
    #Read dataframe from file
    ball = pd.read_hdf(filename3)
    ball2 = pd.read_hdf(filename4)
    df = ball.append(ball2)
    
    
    filename_output = filename3[:-5] + '_stitched_data.hdf5'
    df.to_hdf(filename_output,'ball')
    

def combinePortionsVid(filename5,filename6):
    #combines 2 videos
    #filenames must be in order
    #Load Video
    print(filename5)
    original = cv2.VideoCapture(filename5)
    width = original.get(3)
    height = original.get(4)
    numbFrames = original.get(7)
    
    #Determine where to save output
    filename_output = filename5[:-14] + 'stitched_annotated.avi'
    print(filename_output)
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    op_file = cv2.VideoWriter(filename_output,fourcc,30.0,(int(width),int(height)),True)
    
    
    for n in range(1,int(numbFrames),1):
        
        ret, img = original.read()
        op_file.write(img)#
    original.release()
            
    print(filename6)
    original = cv2.VideoCapture(filename6)
    numbFrames = original.get(7)
    
    for n in range(1,int(numbFrames),1):
        
        ret, img = original.read()
        op_file.write(img)#    
    original.release()    
    
    
    op_file.release()
    
def plotColumns(filename,xindex,yindex):
    ball = pd.read_hdf(filename)
    print(ball.keys())
    x=ball[xindex]
    y=ball[yindex]
    plt.figure('test')
    plt.plot(x,y,'r-')
    plt.plot(x,y,'g.')
    plt.show()

def check(filename):
    ball = pd.read_hdf(filename)
    frame_list = ball['frame'].unique()
    print(frame_list)
    numb_dots = []
    for frame in frame_list:
        numb_dots.append(np.shape(ball[ball['frame']==frame])[0])
    plt.figure('test2')
    plt.plot(frame_list,numb_dots,'r.')
    plt.show()
    
def viewDF(filename,numbrows = 100):
    ball = pd.read_hdf(filename)
    ball2 = ball[ball.index>999]
    print(ball2.head(n=numbrows))
    print(ball[ball.index > 1600])
    
if __name__ == '__main__':
    #filename2 = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBallData/HighSpeedBalls',title='Select Data File', filetypes = (('Datafile', '*.hdf5'),))    
    #filename = filedialog.askopenfilename(initialdir='/home/ppzmis/Documents/Data/BouncingBallData/',title='Select Data File', filetypes = (('AVI', '*.avi'),))    
    #print(filename)
    #check(filename)
    #plotColumns(filename,'frame','surface')
    #viewDF(filename)
    #Remember that the startVal is from beginning of video so just because it is frame 300 if it is the start of the annotated video it should be 0. With
    #The dataframe however it is picking out the actual frame number so should be 300.861
    #start = 1
    #stop =  (0*60+16)*30 
    #extractPortionVid(filename,startVal = start,stopVal = stop)
    #extractPortionDFrame(filename2,startVal = start,stopVal = stop)
    
    
    
    filename3 = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/HighSpeedBalls/P120_10mm/P120 077/',title='Select Data File', filetypes = (('Datafile', '*.hdf5'),))    
    filename4 = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/HighSpeedBalls/P120_10mm/P120 077/',title='Select Data File', filetypes = (('Datafile', '*.hdf5'),))    
    
    
    
    filename5 = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/HighSpeedBalls/P120_10mm/P120 077/',title='Select Data File', filetypes = (('AVI', '*.avi'),))    
    filename6 = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/HighSpeedBalls/P120_10mm/P120 077/',title='Select Data File', filetypes = (('AVI', '*.avi'),))    
    
    combinePortionsDFrame(filename3,filename4)
    
    combinePortionsVid(filename5,filename6)
    
    
   