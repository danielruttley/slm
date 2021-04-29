#Originally written by Mitch Walker
#Modified by Dan Ruttley
#Note that all kinoforms are returned as 512x512 images which should be padded later if needed.
#Arrays are returned 0-1 with 1 being full 2pi phase modulation.

import numpy as np

def blank(size=(512,512)):
    """Returns a 512x512 hologram of all zeros.
    
    Parameters
    ----------
    size : tuple of int
        size of the hologram (xsize,ysize)
    """
    (xsize,ysize) = size
    return np.zeros((ysize,xsize))

def stepver(v1,v2,split):
    """
    This function generates a 2D array for a step function, with the split in the vertical direction. The two sections are on the left and right.
    
    -v1 is an integer value between 0 and 65535, which determines the phase value of half of the kinoform. This value is on the left.
    -v2 is another integer value between 0 and 65535, which determines the phase value for the other half of the kinoform. This value is on the right.
    -split is an integer between 0 and 512, which determines the pixel value where the step is. 0 is on the left, 512 on the right.
    
    returns: a 2d 512x512 array of phase values between 0 and 65535
    """
    
    #Import modules
    import numpy as np    
    
    #Initialise the array to hold the mask values
    maskval = np.empty(512)
    
    #Fill the mask value array
    for i in range(512):
        if i < split:
            maskval[i] = v1
            
        elif i >= split:
            maskval[i] = v2
            
    #Initialise a vertical matrix to multiply the mask value array by to form a 2D matrix    
    vert = np.ones((512,1))

    #Multiply the two arrays
    twodim = vert*maskval

    return twodim/2/np.pi

def stephor(v1,v2,split):
    """
    This function generates a 2D array for a step function, with the split in the horizontal direction. The two sections are on the top and bottom.
    
    -v1 is an integer value between 0 and 65535, which determines the phase value of half of the kinoform. This value is on the top.
    -v2 is another integer value between 0 and 65535, which determines the phase value for the other half of the kinoform. This value is on the bottom.
    -split is an integer between 0 and 512, which determines the pixel value where the step is. 0 is on the top, 512 on the bottom.
    
    returns: a 2d 512x512 array of phase values between 0 and 65535
    """
    
    #Import modules
    import numpy as np    
    
    #Initialise the array to hold the mask values
    maskval = np.empty((512,1))
    
    #Fill the mask value array
    for i in range(512):
        if i < split:
            maskval[(i,0)] = v1
            
        elif i >= split:
            maskval[(i,0)] = v2
       
    #Initialise a horizontal matrix to multiply the mask value array by to form a 2D matrix    
    hor = np.ones(512)

    #Multiply the two arrays
    twodim = maskval*hor
            
    return twodim/2/np.pi

def lagpol(x,p,l):
    """
    This function returns the Laguerre polynomial value for x, given p and l
    
    -x is a float
    -p is an integer between 1 and 6
    -l is a positive integer
    """

    if p == 0:
        val = 1
    elif p == 1:
        val = 1 - x
    elif p == 2:
        val = (1/2)*(x**2 - 4*x + 2)
    elif p == 3:
        val = (1/6)*(-1*(x**3) + 9*(x**2) - 18*x + 6)
    elif p == 4:
        val = (1/24)*(x**4 - 16*(x**3) + 72*(x**2) - 96*x + 24)
    elif p == 5:
        val = (1/120)*(-1*(x**5) + 25*(x**4) - 200*(x**3) + 600*(x**2) - 600*x + 120)
    elif p == 6:
        val = (1/720)*(x**6 - 36*(x**5) + 450*(x**4) - 2400*(x**3) + 5400*(x**2) - 4320*x + 720)

    return val    

def lag_poly(x,p,l):
    import scipy.special as spec
    import numpy as np
    return spec.assoc_laguerre(x,n=p,k=np.abs(l))

def lgmode(x0,y0,p,l,w0):
    """
    This function generates a 2D array for an LG beam mode of 
    
    -x0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the x direction, with 1 at the left and 512 at the right
    -y0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the y direction, with 1 at the top and 512 at the bottom
    -p is an integer between 0 and 6
    -l is an positive integer
    -w0 is a float, and is the waist of the beam in pixels
    
    """
    
    #Import the necessary modules to run the function
    import numpy as np
    import math
    
    #Work out the coordinates based on the origin (x0,y0)
    cartescoord = np.empty((512,512,2))
    
    for i in range(512):
        for j in range(512):
            cartescoord[i,j,0] = (j) - x0
            
            cartescoord[j,i,1] = (j) - y0
            
    #Convert the coordinates from Cartesian to cylindrical polar, taking z to be constant, z=0
    cylincoord = np.empty((512,512,2))

    for i in range(512):
        for j in range(512):
            cylincoord[i,j,0] = np.sqrt((cartescoord[i,j,0]**2)+(cartescoord[i,j,1]**2))
            
            if j == x0:
                if i > y0:
                    cylincoord[i,j,1] = np.pi/2
                elif i < y0:
                    cylincoord[i,j,1] = (3/2)*np.pi
                elif i == y0:
                    cylincoord[i,j,1] = 0
            elif i == y0:
                if j < x0:
                    cylincoord[i,j,1] = np.pi
                elif j > x0:
                    cylincoord[i,j,1] = 0
                elif j == x0:
                    cylincoord[i,j,1] = 0
            elif j>x0 and i>y0:
                cylincoord[i,j,1] = np.arctan(np.abs(cartescoord[i,j,1]/cartescoord[i,j,0]))
            elif j<x0 and i>y0:
                cylincoord[i,j,1] = (np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j<x0 and i<y0:
                cylincoord[i,j,1] = np.pi + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j>x0 and i<y0:
                cylincoord[i,j,1] = ((2)*np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
                
            if np.isnan(cylincoord[i,j,1]) == True:
                print("Not a number ", i, " ", j)
                cylincoord[i,j,1] = 0
            
    
    #Generate and return the phase values
    phase = np.empty((512,512))
    
    laguerre_vals = lag_poly((2 * (cylincoord[:,:,0]**2))/(w0**2),p,l)
    
                
    for i in range(512):
        for j in range(512):
            if laguerre_vals[i,j] <= 0:
                phase[i,j] = (-l*cylincoord[i,j,1])%(2*np.pi)
            elif laguerre_vals[i,j] > 0:
                phase[i,j] = (((-l*cylincoord[i,j,1]) + np.pi)%(2*np.pi))
    return phase/2/np.pi
    # return phase