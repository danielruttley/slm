"""Creates grating holograms to send light to different diffraction orders."""

import numpy as np

def grating(period=7,angle=0,shape=(512,512)):#,origin=None):
    """
    This function generates a 2D striped grating pattern, with 
    phase ramp normal to y=tan(theta)*x.
    
    Parameters
    ----------
    period : float
        the period of one 2pi phase modulation, in SLM pixels
    angle : float
        the angle of the line y=tan(angle)*x that the phase ramp should be 
        perpendicular to
    shape : tuple of int, optional
        shape of the SLM holograms (x,y)
    
    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    origin = (int(shape[0]/2),int(shape[1]/2))
    x = range(-origin[0],shape[0]-origin[0])
    y = range(-origin[1],shape[1]-origin[1])
    xx,yy = np.meshgrid(x,y)
    blazing = ((np.sin(angle)*xx+np.cos(angle)*yy)/period)%1
    return blazing

def hori(period,**kwargs):
    """
    This function generates a 2D horizontally striped grating pattern (such 
    that the phase ramp is in the vertical direction).
    
    Parameters
    ----------
    period : float
        the period of one 2pi phase modulation, in SLM pixels
    
    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    return grating(period,0,**kwargs)
    
def vert(period,**kwargs):
    """
    This function generates a 2D vertically striped grating pattern (such that 
    the phase ramp is in the horizontal direction).
    
    Parameters
    ----------
    period : float
        the period of one 2pi phase modulation, in SLM pixels

    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    return grating(period,np.pi/2,**kwargs)

def diag(period,**kwargs):
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
    return grating(period,np.pi/4,**kwargs)

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
