"""Creates grating holograms to send light to different diffraction orders."""

import numpy as np
from .misc import blank

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

def grating_gradient(gradient=73,angle=0,shape=(512,512)):
    """
    This function generates a 2D striped grating pattern, with 
    phase ramp normal to y=tan(theta)*x. The gradient rather than
    the period is provided. The gradient is the number of ramps on the
    SLM screen.
    
    Parameters
    ----------
    gradient : float
        period = 512/gradient. The period of one 2pi phase modulation, in SLM pixels
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
    if gradient == 0:
        return blank(shape=shape)
    return grating(512/gradient,angle,shape)

def hori(period=7,max_mod_depth=1,shape=(512,512)):
    """
    This function generates a 2D horizontally striped grating pattern (such 
    that the phase ramp is in the vertical direction).
    
    Parameters
    ----------
    period : float
        the period of one 2pi phase modulation, in SLM pixels
    max_mod_depth : float
        the maximum modulation depth to use. This should typically be 1 unless
        deliberately trying to reduce the diffraction efficiency.
    
    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    origin = (int(shape[0]/2),int(shape[1]/2))
    x = range(-origin[0],shape[0]-origin[0])
    y = range(-origin[1],shape[1]-origin[1])
    xx,yy = np.meshgrid(x,y)
    blazing = (yy/period)%1
    return blazing%max_mod_depth

def hori_gradient(gradient=1,max_mod_depth=1,shape=(512,512)):
    """
    Applies a horizontal grating in terms of the gradient of the ramp on the SLM screen.
    A gradient of 1 is 1 ramp over the entire screen, a gradient of 2 is 2 ramps over 
    the screen etc.
    
    Parameters
    ----------
    gradient : float
        the gradient of the 2pi modulation. period = 512/gradient.
    max_mod_depth : float
        the maximum modulation depth to use. This should typically be 1 unless
        deliberately trying to reduce the diffraction efficiency.

    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    if gradient == 0:
        return blank(shape=shape)
    return hori(period=512/gradient,max_mod_depth=max_mod_depth,shape=shape)
    
def vert(period=7,max_mod_depth=1,shape=(512,512)):
    """
    This function generates a 2D vertically striped grating pattern (such that 
    the phase ramp is in the horizontal direction).
    
    Parameters
    ----------
    period : float
        the period of one 2pi phase modulation, in SLM pixels
    max_mod_depth : float
        the maximum modulation depth to use. This should typically be 1 unless
        deliberately trying to reduce the diffraction efficiency.

    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    origin = (int(shape[0]/2),int(shape[1]/2))
    x = range(-origin[0],shape[0]-origin[0])
    y = range(-origin[1],shape[1]-origin[1])
    xx,yy = np.meshgrid(x,y)
    blazing = (xx/period)%1
    return blazing%max_mod_depth

def vert_gradient(gradient=1,max_mod_depth=1,shape=(512,512)):
    """
    Applies a vertical grating in terms of the gradient of the ramp on the SLM screen.
    A gradient of 1 is 1 ramp over the entire screen, a gradient of 2 is 2 ramps over 
    the screen etc.
    
    Parameters
    ----------
    gradient : float
        the gradient of the 2pi modulation. period = 512/gradient.
    max_mod_depth : float
        the maximum modulation depth to use. This should typically be 1 unless
        deliberately trying to reduce the diffraction efficiency.

    Returns
    -------
    array
        blazed grating hologram normalised between 0 - 1
    """
    if gradient == 0:
        return blank(shape=shape)
    return vert(period=512/gradient,max_mod_depth=max_mod_depth,shape=shape)

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
