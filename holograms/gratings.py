"""Creates grating holograms to send light to different diffraction orders."""

import numpy as np

def diag(period,size=(512,512)):
    """
    This function generates a 2D diagonally striped grating pattern, with 
    stripes parallel to y=x.
    
    Parameters
    ----------
    period : float
        the period of one 2pi phase modulation, in SLM pixels
    shape : tuple of int, optional
        shape of the SLM holograms (x,y)
    
    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    (xsize,ysize) = size
    xx,yy = np.meshgrid(range(xsize),range(ysize))
    blazing = ((xx+yy)/(period*np.sqrt(2)))%1
    return blazing

def diag_old(grad,size=(512,512)):
    """
    Original function for generating a diagonal grating with a gradient rather 
    than a period. Left in so that previously used holograms can be regenerated
    if needed.

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

def hori(period,shape=(512,512)):
    """
    This function generates a 2D horizontally striped grating pattern.
    
    Parameters
    ----------
    period : float
        the period of one 2pi phase modulation, in SLM pixels
    shape : tuple of int, optional
        shape of the SLM holograms (x,y)
    
    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    blazing = vert(period,(shape[1],shape[0]))
    return blazing.T
    
def vert(period,shape=(512,512)):
    """
    This function generates a 2D vertically striped grating pattern.
    
    Parameters
    ----------
    period : float
        the period of one 2pi phase modulation, in SLM pixels
    shape : tuple of int, optional
        shape of the SLM holograms (x,y)
    
    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    ver = np.arange(shape[0])
    ver = (ver/period)%1
    hor = np.ones((shape[1],1))
    blazing = ver*hor
    return blazing
