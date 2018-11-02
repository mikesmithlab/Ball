

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import trackpy as tp
import pandas as pd
from scipy import optimize
from scipy.ndimage import binary_fill_holes
import os
from timeit import time
import math



'''image processing help functions'''

def subtractBkg(img,foldername,showImg=False):
    print(foldername)
    img = cv2.GaussianBlur(img,(5,5),0)
    surface_index = int(findSurface(img))

    #showSurface(img,findSurface(img))
    
    name = foldername[:-5] +'bkgimgs/' + str(int(surface_index)) +'.png'
    print(name)
    bkg_img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    
    bkg_img = cv2.GaussianBlur(bkg_img,(5,5),0)
    #img2 = cv2.subtract(img[:,:,0],bkg_img)
    img2 = cv2.GaussianBlur(img,(5,5),0)
    if showImg==True:
        showPic(img2)
    
    return img2

def random_color():
    color = (np.random.randint(32,high=255),(np.random.randint(32,high=255)),(np.random.randint(32,high=255)))
    return color




def distance2Pts(pt1,pt2):
    return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5

'''
Fit functions to a series of points
'''
    
def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    #residu   = np.sum((Ri - R)**2)
    return xc, yc, R

def findOutlines(img,surface,circles,threshold = 40,minArea=1000,aspect_max = 150,rad1=0,rad2=0,leftedge=136,rightedge=1120,showImg=False,debug=False,saveImg=False,fname2=''):
    # Detect edges using Canny
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31, 11)

    print('outlines')
    
    
    img=adjust_gamma(img,gamma=1.1)
    showPic(img)

    canny_output = cv2.Canny(img, threshold, 250)
    showPic(canny_output)
    kernel = np.ones((3,3),np.uint8)
    canny_output = cv2.dilate(canny_output,kernel,iterations = 1)
    
    #showPic(canny_output)
    # Find contours
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #for contour in contours:
    
    #join close by contours
    #find begin and end of contours
    
    
    
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    drawing2=drawing.copy()
    for contour in range(len(contours)):
        #color = (255,255,255)
        color=random_color()
        cv2.drawContours(drawing, contours, contour, color,4)
    if debug and showImg:
        showPic(drawing)
    
    hull_list = []
    for i in range(len(contours)):
        # check for connections
        
        hull = cv2.convexHull(contours[i])
        if debug == True:
            color=random_color()
        
            #Remove smaller outlines
            #cv2.drawContours(drawing2,[hull],0,color,4)
    
        
        
        
        if cv2.contourArea(hull) > minArea:
            x,y,w,h = cv2.boundingRect(contours[i])
            aspect_ratio1 = float(w)/float(h)
            aspect_ratio2 = float(h)/float(w)
            if aspect_ratio2 > aspect_ratio1:
                aspect = aspect_ratio2
            else:
                aspect = aspect_ratio1
            
            if aspect < aspect_max:
                M = cv2.moments(contours[i]) 
                cy = int(M['m01']/M['m00'])
                cx = int(M['m10']/M['m00'])
                if cy < (surface-10):
                    if cy > surface - 500:
                        if cx < rightedge:
                            if cx > leftedge:
                                hull_list.append(i)
    
    

    
    
    if len(hull_list) > 1:
        for i in range(len(hull_list)-1):
            if i==0:
                contours2 = np.vstack((contours[hull_list[0]],contours[hull_list[1]]))
            else:
                contours2 = np.vstack((contours2,contours[hull_list[i+1]]))
    else:
        contours2 = contours[hull_list[0]]
        
    xdata = []
    ydata = []
    
    for i in range(np.shape(contours2)[0]):
        xdata.append(contours2[i][0][0])
        ydata.append(contours2[i][0][1])
    
    xc,yc,rc=leastsq_circle(xdata,ydata)
    print(rc)
    circles2 = np.ndarray((1,1,3))

    circles2[0][0][0] = xc
    circles2[0][0][1] = yc
    circles2[0][0][2] = rc



    for i in range(len(contours2)):
        color=(255,255,255)
        cv2.drawContours(drawing2, contours2, i, color,4)
        
    #circles2 = cv2.HoughCircles(drawing2[:,:,0],cv2.HOUGH_GRADIENT,1,160,param1=p1,param2=p2,minRadius=140,maxRadius=180)
    if debug and showImg:
        img2=img.copy()
        cv2.circle(img2,(int(circles2[0][0][0]),int(circles2[0][0][1])), int(circles2[0][0][2]),(255,0,0),4)
        showPic(drawing2)
        showPic(img2)
    
    drawing=drawing[:,:,2]
    
    if showImg==True:
        showPic(drawing,name='canny')
    if saveImg==True:
        #print(fname2)
        cv2.imwrite(fname2, drawing)
    return drawing2[:,:,0],circles2

def applyPatch(img,x,surface,dy,width=20,height=30,showImg=False):
    
    
    x = x - int(width/2)
    y= surface -dy - int(height/2)
    
    
    img[y:y+height,x:x+width] = 255#apply patch to masked image
    
    
    
    if showImg == True:
        cv2.imshow('img with patch',img)#th1)#bin_img.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img

def adjust_gamma(image, gamma=0, cutoff=0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    for i in np.arange(0,cutoff):
        table[i] = 0
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

'''show various images'''
def showPic(img,name='',save=False,fname2 = '/media/ppzmis/data/BouncingBall_Data/newMovies/RawDataandTracking/Examples/test.png')   :
    cv2.imshow(name,img)
    if save:
        cv2.imwrite(fname2,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showSurface(Img,surface):
    #Add surface line
    sz = np.shape(Img)
    cv2.line(Img,(0,surface),(sz[1],surface),(255,0,0),2)
    cv2.imshow('detected circles',Img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def showTracking(Img,circles,dotPts,surface,patch=None,title='',name='',showImg = False,saveImg=False):
    '''
    This function enables one to view the ball outline and tracked dots on an image within the program.
    It is also used to generate the annotated images that are written to file by opencv.
    
    inputs:
        Img = original image to be annotated
        circles = tuple (x,y,rad) created by findTrackedBallOutline
        dotPts = Pandas dataframe containing coordinates of all the dots found on the ball
        displayImg = Show the image for debugging etc.
        
    outputs:
        Img = returns annotated image
    '''   
    #Adds the ball outline and located dots to the image and displays if requested
    cv2.circle(Img,(int(circles[0][0][0]),int(circles[0][0][1])),int(circles[0][0][2]),color=(0,255,0),thickness=2)
    
    #Ring the dots being tracked
    for x,y in zip(dotPts.x.tolist(),dotPts.y.tolist()):
        cv2.circle(Img,(int(x),int(y)),10,color=(0,0,255),thickness = 2)
    
    
    #Add surface line
    sz = np.shape(Img)
    cv2.line(Img,(0,surface),(sz[1],surface),(255,0,0),2)
    
    #ShowPatch
    if patch == None:
        pass
    else:
        x=patch[0]
        dy = patch[1]
        width=10
        height=10
        x = x - int(width/2)
        y= surface -dy - int(height/2)
        
        cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),3)
    
    if saveImg==True:
        print(name)
        cv2.imwrite(name, Img) 
    
    if showImg == True:
        cv2.imshow(title,Img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
       
    return Img

'''---------------------------------------------------------------------------
--------------------------------------------------------------------------------------------
main functions
============================================================================================='''
def bkgImgList(foldername,filename2):
    try:
        os.mkdir(foldername) #Make directory to store fit graphs
    except:
        pass
    print(filename2)
    cap2 = cv2.VideoCapture(filename2)
    
    numframes = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))-1
    print(numframes)
    #width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    #height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    #Create a list of frame numbers with surface at different height
    index = np.zeros(numframes)
    ret, img = cap2.read()
    
    for i in range(0,numframes):
        ret, img = cap2.read()
        #showPic(img)
        index[i] = int(findSurface(img))
        #showSurface(img,findSurface(img))
    #surface_vals are the surface pixel positions
    #img_nums are the index of the frame
    #Avoid frame 0 as this sometimes is a dodgy image.hence index[1:] and img_nums = img_nums+1
    surface_vals,img_nums = np.unique(index[1:],return_index=True)
    img_nums = img_nums + 1

    #bkg_imgs = np.zeros((width,height,num_bkgimgs))
    cap2.release()
    
    #Add in missing values as not all surface values are present
    surf = np.arange(int(surface_vals.min() - 4),int(surface_vals.max() + 5), 1)
    
    #just preallocation
    img_nums2 = np.arange(int(surface_vals.min() - 4),int(surface_vals.max() + 5),1)
    i=0
    j=0
    
    
    for s in surf:
        if s in surface_vals:
            img_nums2[j] = img_nums[i]
            i=i+1
        else:
            #If a surface value doesn't exist set the image number
            #to the closest available image
            img_nums2[j] = img_nums[np.argmin((s-surface_vals)**2)]
        j=j+1

    #Save images to file with surface index as names
    cap2 = cv2.VideoCapture(filename2)
    for frame in range(0,numframes):
        ret, img = cap2.read()
        if frame in img_nums2:
            indices, = np.where(img_nums2 == frame)
 
            for indice in indices:
                name =foldername + '/' + surf[indice].astype(str) + '.png' 
                cv2.imwrite(name, img)
    cap2.release()
  
    return True

def findSurface(image, binthreshold=20,surfthreshold=300000,surfaceShift=0, showImg = False):
    '''
    binthreshold binarises the image
    surfthreshold is the sum of the binarised pixels in the x direction above which we
    have a surface.
    The surface shift is used to manually move the position up to account for the wet and dry sandpaper (we are tracking the aluminium surface).
    +ve values move it up on the image.
    returns from the function are simply the pixel index at which the surface is
    '''
    
    #Apply thresholding and remove small dots through erosion
    blurImg = cv2.blur(image.astype(np.uint8),(21,3));
    ret, bin_img = cv2.threshold(blurImg,binthreshold,255,cv2.THRESH_BINARY)
    #showPic(bin_img)
    sumvals = np.sum(bin_img[:,:,0],axis=1)
    #print(sumvals)
    #showPic(bin_img)
    surf = np.where(sumvals > surfthreshold)[0]
    surface_index = surf[0,] - int(surfaceShift)
    
    
    
    if showImg == True:
   
        sz=np.shape(image)
               
        cv2.line(image,(0,surface_index),(sz[1],surface_index),(255,0,0),2)
        cv2.imshow('masked image',bin_img.astype(np.uint8))#image.astype(np.uint8)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
                      
    return surface_index

def findBall(image,surface,circles,minRad=150,maxRad=165,fname='',showImg=False,saveImg=False,debug2=False):
    
    
    img2,circles=findOutlines(image,surface,circles,rad1=minRad,rad2=maxRad,showImg=showImg,saveImg=False,fname2=fname,debug=debug2)
    
    showImg=False
    if showImg==True:
        ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        #image - outline
        img = cv2.bitwise_and(image,image,mask = mask_inv)

        #convert to colour
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        #convert outline to colour
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        img2[:,:,0]=0
        img2[:,:,2]=0
        img = cv2.add(img,img2)
        showPic(img,name=str(np.shape(img)))
    if saveImg==True:
        ret, mask = cv2.threshold(img2, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        # 
        #image - outline
        img = cv2.bitwise_and(image,image,mask = mask_inv)
        #convert to colour
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        #convert outline to colour
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
        img2[:,:,1]=0
        img2[:,:,2]=0
        img = cv2.add(img,img2)
        cv2.imwrite(fname, img)
    return circles
    
def findDots(image,circles,dotSize = 21,inFromEdge=0.92,showImg=False):
    white = [1,1,1]
    
    
    sz = image.shape       
    mask = np.zeros(sz)
    
    cv2.circle(mask,(int(circles[0,0,0]),int(circles[0,0,1])),int(circles[0,0,2]),255,-1)
    image = cv2.bitwise_and(image.astype(np.uint8),image.astype(np.uint8),mask = mask.astype(np.uint8))
  
    dots = tp.locate(image, dotSize,invert=True)
    #Remove those dots that are close to the edge of the ball.
    dots2 = dots[(dots.x - circles[0][0][0])**2+(dots.y - circles[0][0][1])**2 < (inFromEdge*circles[0][0][2])**2]
   
    return dots2
    
    
def findBallDots(img,circles,showAll = False, framenum=0,foldername='',showImg=False,test=False):
    '''
    This function finds the location of the ball and dots on the ball
    
    inputs:
        img = A grayscale image
        dotSize = an upper estimate of diameter in pixels of the dots to be looked for by trackpy
        inFromEdge = do not store dots whose centres are greater than inFromEdge * Bal radius
        RadBall = Radius of ball in pixels
        
    outputs:
        dots2 = pandas dataframe with row for each dot found. 
                columns ['frame','x','y','mass','size','ecc','signal','xball','yball','radball']
                where 'x','y' are dot positiop1=0,p2=0,n and 'xball', 'yball' are ball centre position
        frame_tracked = annotated image as numpy array 
    
    
    '''
   
    img2 = img.copy()
    #img will be used to measure surface position-returns the y pixel value at which the surface is located (0 is at the top of the image)
    surface=findSurface(img)
   
    #subtract bkg images
    print(foldername)
    img2 = subtractBkg(img2,foldername[:-1]+'/')
    #Find ball position and radius
    
    circles = findBall(img2.copy(),surface,circles,fname=foldername + '/' + str(framenum) + '_edge.png',showImg=showImg,saveImg=False,debug2=test)
    
    #find dots on ball
    img3=applyPatch(img2.copy(),961,surface,591-390)
    dots=findDots(img3,circles,showImg=False)
    
    
    trackedImg = showTracking(img,circles,dots,surface,patch=None,title='Radius = ' + str(circles[0][0][2]),name =foldername + '/' + str(framenum) +'.png',showImg=showImg,saveImg=True)
   
    print(circles)
    
    #Add frame number, position of ball centre columns
    frame_num_vals = np.ones(dots.shape[0])*framenum
    dots.loc[:,'frame'] = frame_num_vals
    dots.loc[:,'xball'] = circles[0][0][0]
    dots.loc[:,'yball'] = circles[0][0][1]
    dots.loc[:,'radball'] = circles [0][0][2]
    dots.loc[:,'surface'] = surface
    
    
    #dots2.loc[:,'dCOM'] = dots2.loc[:,'yball'] - surface
    
  

    
    return dots, trackedImg, circles


if __name__ == "__main__":
    skipNFrames =1
    skipEveryNth = 1#works only in test mode
    stopFrame = False
    
    debug=True
    showdebugging = True
   
    #Load Video
    #filename = filedialog.askopenfilename(initialdir='/media/ppzmis/data/BouncingBall_Data/newMovies/RawDataandTracking/',title='Select Data File', filetypes = (('AVI', '*.avi'),))    
    
    filename = '/media/ppzmis/data/BouncingBall_Data/newMovies/RawDataandTracking/8mm_ball/P120/8mmballbP120_077.avi'
    print(filename)
    filename2 = filename[:-4] + '_bkg.avi'
    
    
    
    #showPic(img)
    #Saves a list of bkg images with the surface index as name to enable background subtraction
    #bkgImgList(filename[:-4] +'bkgimgs',filename2)
    
    #Video read object

    cap = cv2.VideoCapture(filename)
    
        
    
    
    #Make directory to store fit graphs
    new_folder = filename[:-4] + '_fits'
    try:
        os.mkdir(new_folder) 
    except:
        pass
    
    #skip first frame
    ret,img=cap.read()
    print(ret)
    n=0
    while n < skipNFrames:
        n = n+1
        ret, img = cap.read()
    frame_size = np.shape(img)    

    #Determine where to save output
    filename_output = filename[:-4] + '_annotated.avi'
    print(filename_output)
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    op_file = cv2.VideoWriter(filename_output,fourcc,30.0,(frame_size[1],frame_size[0]))
    
    
    
    #Set number of frames to analyse
    if stopFrame == False:
        Numb_Of_Frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        Numb_Of_Frames = stopFrame +1
        
        
    
    #Frame index
    circles = np.ndarray((1,1,3))
    #For each frame
    circles[0][0][0] = 0
    circles[0][0][1] = 0
    circles[0][0][2] = 0
    
    
    while n < Numb_Of_Frames:
        #Find position of ball and its dots and store in dataFrame
        if n==(skipNFrames):#Sometimes the first frame is dodgy so best to check fit on 2nd frame
            ball, trackedFrame,circles = findBallDots(img,circles,framenum=n, foldername = new_folder,showImg=showdebugging,test=debug)
            if debug == True:
                    op = filename[:-12] + str(n) + '.hdf5'
                    ball.to_hdf(op,'w')
            #print(ball)   
        elif n>skipNFrames:
            try:
                ball_temp, trackedFrame,circles, = findBallDots(img,circles,framenum=n, foldername = new_folder,showImg=showdebugging,test=debug)
                if debug == True:
                    op = filename[:-12] + str(n) + '.hdf5'
                    ball_temp.to_hdf(op,'w')
                #Add results to dataFrame
                ball = pd.concat([ball, ball_temp])
                #Write annotated frame to file
                op_file.write(trackedFrame)
                
        
                    

                    
                
            except:#This is intended to save what you have so far should something go wrong
                #Clean up the video stuff
                cap.release()
                op_file.release()
    
                #Save Dataframe to file
                dataFile = filename[:-4] + '_data.hdf5'
                ball.to_hdf(dataFile,'w')
                print('Fitting failed on frame' + str(n))
                raise#Abort the program
        else:
            pass
        
        if (Numb_Of_Frames - n + 1) % 100 == 0:
            print(Numb_Of_Frames - n + 1)
            print(n)
            print(filename)
        #Read next frame
        ret, img = cap.read()            
        n=n+1    
            
           
            
            
        
       
    #Clean up the video stuff
    cap.release()
    op_file.release()
    
    #Save Dataframe to file
    dataFile = filename[:-4] + '_data.hdf5'
    ball.to_hdf(dataFile,'w')
    
    
    print('Processing Finished')
    
    
        
    
    
    
    