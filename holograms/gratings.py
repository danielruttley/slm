"""Creates grating holograms to send light to different diffraction orders."""

import numpy as np

def diag(grad,size=(512,512)):
    """
    This function generates a 2D array for a blazed grating phase pattern with diagonal fronts. The fronts can be perpendicular to either the line y=x or the line y=-x
    
    Parameters
    ----------
    grad : float
        the gradient of the blazed grating; a larger value decreases the size 
        of each grating. The sign of grad determines the orientation; 
        positive for fronts perpendicular to the line y=x, and negative for 
        fronts perpendicular to the line y=-x
    size: tuple of int
        size of the hologram to be returned (xsize,ysize)
    
    Returns
    -------
    array : array of floats
        2d 512x512 array of phase values normalised in the range 0-1
    """
    (xsize,ysize) = size
    array = np.empty((ysize,xsize))
    if grad>0:
        #Fill the array based on the distance of each element from the bottom left element, assuming unit spacing between nearest neighbours
        for i in range(ysize):
            for j in range(xsize):
                array[i,j] = j + (ysize-1-i)
        array = (np.abs(grad)*array)%(2*np.pi)
    else:
        for i in range(ysize):
            for j in range(xsize):
                array[i,j] = i + j
        array = (np.abs(grad)*array)%(2*np.pi)

    array /= 2*np.pi

    return array

def vert(grad):
    """
    This function generates a 2D array for a blazed grating phase pattern with vertical fronts.
    
    -grad is the gradient of the blazed grating; a larger value decreases the size of each grating.
    
    returns: a 2d 512x512 array of phase values normalised such that the maximum value is 2pi
    """    
    #Initialise a 1D array of length 512, with each entry spaced by a unit from the previous entry
    hor = np.arange(0,512,1)
    
    #Convert these values into phase values based on the gradient, and modulo between 0 and 2pi
    hor = (np.abs(grad)*hor)%(2*np.pi)
    
    #Make a vertical n by 1 matrix to conjugate the phase values with
    vert = np.ones((512,1))
    
    #Conjugate the vertical array with a horizontal array of ones to create a 2D matrix
    twodimarray = vert*hor
    
    return twodimarray/2/np.pi
    
def hori(grad):
    """
    This function generates a 2D array for a blazed grating phase pattern with horizontal fronts.
    
    -grad is the gradient of the blazed grating; a larger value decreases the size of each grating.
    
    returns: a 2d 512x512 array of phase values normalised such that the maximum value is 2pi
    """
    
    #Import modules
    import numpy as np
    
    #Create an empty 2D vertical array, length 512
    vert = np.ones((512,1))
    
    #Convert these values into phase values based on the gradient, and modulo between 0 and 2pi
    for i in range(512):
        vert[i,0] = (i*grad)%(2*np.pi)
        
    #Create a horizontal 1 by n matrix to conjugate the phase values with
    hor = np.ones((1,512))
    
    #Conjugate the vertical array with a horizontal array of ones to create a 2D matrix
    twodimarray = vert*hor
    
    return twodimarray/2/np.pi
