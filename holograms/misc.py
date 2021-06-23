#Originally written by Mitch Walker
#Modified by Dan Ruttley
#Note that all kinoforms are returned as 512x512 images which should be padded later if needed.
#Arrays are returned 0-1 with 1 being full 2pi phase modulation.

import numpy as np
from PIL import Image

def blank(phase=0,shape=(512,512)):
    """Returns a 512x512 hologram of all zeros.
    
    Parameters
    ----------
    size : tuple of int
        size of the hologram (xsize,ysize)
    """
    (xsize,ysize) = shape
    return np.ones((ysize,xsize))*phase/2/np.pi

def save(hologram,filename):
    hologram = np.uint16(hologram*65535)
    red = np.uint8(hologram % 256)
    green = np.uint8((hologram - red)/256)
    blue = np.uint8(np.zeros(hologram.shape))
    rgb = np.dstack((red,green,blue))
    image = Image.fromarray(rgb,"RGB")
    image.save(filename)

def load(filename):
    image = Image.open(filename)
    array = np.array(image)
    red = array[:,:,0]
    green = array[:,:,1]
    holo = red+green*256
    return holo/65535

def translate(holo,shift):
    """
    Returns a hologram that has been translated on the SLM screen. Empty pixels
    are filled with zero.

    Parameters
    ----------
    holo : array of float
        hologram to translate
    shift : tuple of int
        amount to shift the hologram (xshift,yshift)
    
    Returns
    -------
    array
        translated hologram
    """
    shifted_holo = holo.copy()
    (xshift,yshift) = shift
    if xshift < 0:
        shifted_holo = shifted_holo[:,-xshift:]
        shifted_holo = np.pad(shifted_holo,((0,0),(0,-xshift)))
    elif xshift > 0:
        shifted_holo = shifted_holo[:,:-xshift]
        shifted_holo = np.pad(shifted_holo,((0,0),(xshift,0)))
    if yshift < 0:
        shifted_holo = shifted_holo[-yshift:,:]
        shifted_holo = np.pad(shifted_holo,((0,-yshift),(0,0)))
    elif yshift > 0:
        shifted_holo = shifted_holo[:-yshift,:]
        shifted_holo = np.pad(shifted_holo,((yshift,0),(0,0)))
    return shifted_holo


def stepver(v1,v2,split):
    """
    This function generates a 2D array for a step function, with the split in the vertical direction. The two sections are on the left and right.
    
    -v1 is an integer value between 0 and 65535, which determines the phase value of half of the kinoform. This value is on the left.
    -v2 is another integer value between 0 and 65535, which determines the phase value for the other half of the kinoform. This value is on the right.
    -split is an integer between 0 and 512, which determines the pixel value where the step is. 0 is on the left, 512 on the right.
    
    returns: a 2d 512x512 array of phase values between 0 and 65535
    """    
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