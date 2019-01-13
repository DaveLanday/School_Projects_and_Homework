# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:15:23 2018

@author: mgreen13
"""

import numpy as np
import pylab as pl



#image = rgb2gray(pl.imread("gasket.png"))
def box_count(image_file):
    """
    DESCRIPTION: function takes in imgage file. A 2d dimensional histogram is 
                 used to determine if points are in boxes. The function returns
                 the number of boxs needed to cover the object and size of boxes
                 for several trials. 
                 
    INPUT: image_file = IMAGE FILE
    
    OUTPUT: sizes ,edge = box sizes, number of boxs
    
    """
    
    
    # count pixles wiith non zero value
    pix = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j][0]>0:
                pix.append((i,j))
    
    # get dim of image
    x = image.shape[1]
    y = image.shape[0]
    
    #convert pix list to array
    pix = pl.array(pix)
    
    
    #Generate box from .01, 1 in log2 space 
    sizes=np.logspace(0.01, 1, 100, endpoint=False, base=2)
    
    # Use n dimensional numpy histogram to create and count boxs 
    
    # Use scales as bins of histogram for x and y dimension
    # Create fewer, larger boxs for each iteration
    boxs = []
    for size in sizes:
        H,edges = np.histogramdd(pix,bins =(np.arange(0,x,size),np.arange(0,y,size)))
        #return the number of histograms that are non zero 
        # i.e. the number of boxes that contain a portion of the fractal 
        boxs.append(np.sum(H>0))
    return(sizes, boxs)

image =pl.imread(image_file)

sizes, boxs = box_count(image)


# Plot and regress data to generate a fractal dimension estimate
pl.figure(figsize = (15,9))
coeffs = np.polyfit(np.log(sizes),np.log(boxs),1)

pl.plot(np.log(sizes),np.log(boxs), 'o', mfc='none')
pl.plot(np.log(sizes), np.polyval(coeffs,np.log(sizes)),label = "Dim = {}".format(np.round(-coeffs[0],3)))
pl.legend(loc = "bottom left")
pl.title("Box Counting Dimension of DLA Fractal")
pl.xlabel('log Box Size')
pl.ylabel('log Box Number')