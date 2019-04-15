import cv2
import numpy as np
import pandas as pd
from tkinter import filedialog
import matplotlib.pyplot as plt
from os.path import split
import glob
import os

'''show various images'''
def showPic(img,name=''):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extractPortionVid(original,op_file,startVal = 0,stopVal = 0):
   
    for n in range(startVal,stopVal,1):

        ret, img = original.read()
        op_file.write(img)
        
    return (original,op_file)

def writeSingleFrame(original,op_file,imgfile):
    #move the original vid on by one frame
    ret,img2=original.read()

    
    #open and read image from new file
    img = cv2.imread(imgfile)
    #showPic(img)
    op_file.write(img)
    
    return (original,op_file)
    
def extractPortionDFrame(ball,startVal = 0,stopVal = 0):
    #Read dataframe from file
    
    output = ball[(ball['frame'] >= startVal) & (ball['frame'] < stopVal)]
    
    return output

def insertFrameToDF(ball_temp,ball_frame):
    ball_temp = pd.concat([ball_temp, ball_frame])
    
    return ball_temp

def showPic(img,name='')   :
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
if __name__ == '__main__':
    #Select main datafile
    filename2 = filedialog.askopenfilename(initialdir='//media/ppzmis/data/BouncingBall_Data/newMovies/',title='Select Data File', filetypes = (('Datafile', '*.hdf5'),))    
    filename = filename2[:-10] + '_annotated.avi'
    print(filename2)
    print(filename[:-27])
    #list frames to be inserted. These should be saved in same folder and their names should just be the frame number.
    filelist = glob.glob(filename[:-27] +'*.png')

    #Create readVidObject
    original = cv2.VideoCapture(filename)
    width = original.get(3)
    height = original.get(4)
    numbFrames = 2997

    print(filelist)
    #print(int(os.path.basename(filelist[0])[:-4]))
    refitFrames = sorted([int(os.path.basename(x)[:-4]) for x in filelist])# [225,226,227,228,229,230,231,239,240,259,277]
    print(refitFrames)
    
    #Create write VidObject
    #Determine where to save output
    filename_output = filename[:-4] +'updated.avi'
    print(filename)
    print(filename_output)
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    op_file = cv2.VideoWriter(filename_output,fourcc,30.0,(int(width),int(height)),True)
    
    
    ball = pd.read_hdf(filename2)
    print(np.shape(ball))
    path,file=split(filename)
    
    
    
    #Extract initial portions
    print(refitFrames[0])
    original, op_file = extractPortionVid(original,op_file,startVal=1,stopVal = refitFrames[0]-1)

    ball_temp = extractPortionDFrame(ball,startVal = 0,stopVal = refitFrames[0])
    
    for i in range(np.shape(refitFrames)[0]):
        #read frame of data
        print(path +'/'+ str(refitFrames[i]) +'.hdf5')
        ball_frame = pd.read_hdf(path + '/'+ str(refitFrames[i]) +'.hdf5')
        #append to df

        ball_temp=insertFrameToDF(ball_temp,ball_frame)
        #read annotated image
        imgfilename = path +'/' + str(refitFrames[i]) + '.png'
        #append to annotated movie.
        writeSingleFrame(original,op_file,imgfilename)
        if i < np.shape(refitFrames)[0]-1:
            original, op_file = extractPortionVid(original,op_file,startVal=refitFrames[i]+1,stopVal = refitFrames[i+1])
            ball_temp=insertFrameToDF(ball_temp, extractPortionDFrame(ball,startVal = refitFrames[i]+1,stopVal = refitFrames[i+1]))
    
    n=refitFrames[-1]
    if refitFrames[-1] < numbFrames:
        ball_temp=insertFrameToDF(ball_temp, extractPortionDFrame(ball,startVal = refitFrames[-1]+1,stopVal = numbFrames+1))
        ret = True
        while ret == True:
            ret,img=original.read()
            op_file.write(img)
            n=n+1
    
    original.release()
    op_file.release()
    print('test')
    print(np.shape(ball_temp))
    print(ball_temp.frame.max())
    ball_temp.to_hdf(filename2[:-5] + 'updated.hdf5','ball')                              
                                  
   
    
   