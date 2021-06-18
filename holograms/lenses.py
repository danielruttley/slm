"""Creates lens holograms shift the focal plane after the Fourier lens."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
package_directory = os.path.dirname(os.path.abspath(__file__))

def lens(f,center=None,wavelength=1064e-9,pixel_size=15e-6,shape=(512,512)):
    """Generates a 2D array for a thin lens with focal length f.  
    The pixel pitch of the SLM is 15umx15um, which is why 15e-6 appears.
    
    Parameters
    ----------
    f : float
        the focal length of the lens, in meters. Positive for converging, 
        negative for diverging.
    center : tuple of int
        SLM pixels of the center of the lens in the form (x0,y0). If None, the 
        center of the SLM screen is used.
    wavelength : float
        the wavelength of the laser, in meters.
    pixel_size : float, optional
        the height/width of an SLM pixel, in meters.
    shape : tuple of int
        the resolution of the SLM screen (x,y)

    Returns
    -------
    array
        lens hologram normalised 0-1
    """
    if center is None:
        center = (shape[0]/2,shape[1]/2)
    k = (2*np.pi)/wavelength
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx,yy = np.meshgrid(x,y)
    r = np.sqrt((xx-center[0])**2+(yy-center[1])**2)
    r *= pixel_size
    phase = (k*r**2/2/f)%(2*np.pi)
    phase /= 2*np.pi
    return phase

def focal_plane_shift(shift,**kwargs):
    """Returns the required lens hologram which moves the focal plane to the
    required distance from the Fourier lens.

    Parameters
    ----------
    shift : float
        the distance that the SLM should shift the focal length by in um
    
    Returns
    -------
    array
        lens hologram with the correct focal length to shift the lens
    """
    df = pd.read_csv(os.path.join(package_directory, "f_vs_focal_shift.csv"))
    fs = df['f [m]']
    shifts = df['focus shift [um]']
    
    if shift > 0:
        if shift > np.max(shifts[shifts > 0]):
            raise ValueError('shift is above max. value of {:.2f}'.format(np.max(shifts[shifts > 0])))
        elif shift < np.min(shifts[shifts > 0]):
            raise ValueError('shift is below min. positive value of {:.2f}'.format(np.min(shifts[shifts > 0])))
    else:
        if shift < np.min(shifts[shifts < 0]):
            raise ValueError('shift is below min. value of {:.2f}'.format(np.min(shifts[shifts < 0])))
        elif shift < np.min(shifts[shifts < 0]):
            raise ValueError('shift is above max. negative value of {:.2f}'.format(np.max(shifts[shifts < 0])))        
    f1 = interp1d(shifts,fs, kind='linear')
    f = f1(shift)
    print('shift of {:.2f}um => lens with f = {:.2f}m'.format(shift,f))
    return lens(f,**kwargs)
    
if __name__ == "__main__":
    focal_plane_shift(-3.9)