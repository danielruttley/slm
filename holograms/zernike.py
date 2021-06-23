import numpy as np
from .apertures import circ

def R(r,radial,azimuthal):
    R = np.zeros_like(r)
    if (radial-azimuthal)%2 == 0:
        kmax = int((radial-azimuthal)/2)
        for k in range(kmax+1):
            prefactor = (-1)**k*np.math.factorial(radial-k)/(np.math.factorial(k)*np.math.factorial((radial+azimuthal)/2-k)*np.math.factorial((radial-azimuthal)/2-k))
            R += prefactor*r**(radial-2*k)
    return R

def zernike(radial=0,azimuthal=0,amplitude=1,x0=None,y0=None,radius=None,shape=(512,512)):#,wrap_phase=False):
    """
    Generates a Zernike polynomial hologram with variable amplitude centred
    about zero
    
    Parameters
    ----------
    radial : int
        radial number of Zernike polynomial
    azimuthal : int
        azimuthal number of Zernike polynomial
    amplitude : float
        maximum value of the hologram. 1 for -2pi to 2pi phase modulation
    xcenter : float
        xcenter of the Zernike polynomial in SLM pixels. If None, the center of
        the hologram is used.
    ycenter : float
        ycenter of the Zernike polynomial in SLM pixels. If None, the center of
        the hologram is used.
    radius: float
        radius of the circle defining the polynomial. If None, the maximum 
        radius possible with x0 and y0 that fits on the screen is used.
    shape : tuple of int, optional
        resolution of the SLM (x,y)

    Returns
    -------
    array
        Zernike polynomial hologram with 1 being 2pi phase modulation
    """
    if azimuthal > radial:
        raise ValueError("azimuthal must be less than or equal to radial")
    if x0 is None:
        x0 = shape[0]/2
    if y0 is None:
        y0 = shape[1]/2
    if radius is None:
        radius = min([shape[0]-x0,shape[1]-y0,x0,y0])
    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx,yy = np.meshgrid(x,y)
    r = np.sqrt((xx-x0)**2+(yy-y0)**2)
    r /= radius
    phi = np.arctan2(yy-y0,xx-x0)
    if azimuthal >= 0:
        phase = R(r,radial,azimuthal)*np.cos(azimuthal*phi)
    else:
        phase = R(r,radial,-azimuthal)*np.sin(-azimuthal*phi)
    phase *= amplitude #divide by 2 to define polynomial on range -0.5 - 0.5
    phase = circ(phase,x0,y0,radius)
    # if wrap_phase:
    #     phase = phase%1
    print(np.max(phase))
    print(np.min(phase))
    return phase