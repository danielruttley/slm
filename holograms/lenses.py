"""Creates lens holograms shift the focal plane after the Fourier lens."""

import numpy as np
from .misc import blank

def lens(w,f,center,pixel_size=15e-6):
    """Generates a 2D array for a thin lens with focal length f.  
    The pixel pitch of the SLM is 15umx15um, which is why 15e-6 appears.
    
    Parameters
    ----------
    w : float
        the wavelength of the laser, in meters.
    f : float
        the focal length of the lens, in meters. Positive for converging, 
        negative for diverging.
    center : tuple of int
        SLM pixels of the center of the lens in the form (x0,y0)
    pixel_size : float, optional
        the height/width of an SLM pixel, in meters.

    Returns
    -------
    array
        lens hologram normalised 0-1
    """

    k = (2*np.pi)/w
    x = np.arange(512)
    y = np.arange(512)
    xx,yy = np.meshgrid(x,y)
    r = np.sqrt((xx-center[0])**2+(yy-center[1])**2)
    r *= pixel_size
    phase = (k*r**2/2/f)%(2*np.pi)
    phase /= 2*np.pi
    return phase

def focal_plane_shift(focal_plane,center,wavelength=1064e-9,f_lens=0.5,d_lens=0.45):
    """Returns the required lens hologram which moves the focal plane to the
    required distance from the Fourier lens.

    Parameters:
        focal_plane: the distance of the focal plane from the Fourier lens
    focal_plane : float
        the distance that the focal plane should be behind the Fourier lens, in
        meters.
    center : tuple of int
        SLM pixels of the center of the lens in the form (x0,y0)
    wavelength : float
        the wavelength of the laser, in meters.
    f_lens : float
        the focal length of the Fourier lens, in meters.
    d_lens : float
        the distance from the SLM to the Fourier lens, in meters
    """
    if focal_plane == 0.5:
        return blank()
    else:
        S1 = 1/((1/f_lens) - (1/focal_plane))
        f_prime = d_lens - S1
        return lens(wavelength, f_prime, center)