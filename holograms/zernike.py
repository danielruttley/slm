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

def zernike(radial,azimuthal,amplitude,center,radius,shape=(512,512),wrap_phase=False):
    """Generates a Zernike polynomial hologram with variable amplitude centred
    about zero
    
    Parameters
    ----------
    radial : int
        radial number of Zernike polynomial
    azimuthal : int
        azimuthal number of Zernike polynomial
    amplitude : float
        maximum value of the hologram. 1 for 0-2pi phase modulation
    center : tuple of float
        center of the Zernike polynomial in SLM pixels
    radius: float
        radius of the circle defining the polynomial
    shape : tuple of int, optional
        resolution of the SLM (x,y)

    Returns
    -------
    array
        Zernike polynomial hologram with 1 being 2pi phase modulation
    """

    x = np.arange(shape[0])
    y = np.arange(shape[1])
    xx,yy = np.meshgrid(x,y)
    r = np.sqrt((xx-center[0])**2+(yy-center[1])**2)
    r /= radius
    phi = np.arctan2(yy-center[1],xx-center[0])
    if azimuthal >= 0:
        phase = R(r,radial,azimuthal)*np.cos(azimuthal*phi)
    else:
        phase = R(r,radial,-azimuthal)*np.sin(-azimuthal*phi)
    phase *= amplitude #divide by 2 to define polynomial on range -0.5 - 0.5
    phase = circ(phase,center,radius)
    if wrap_phase:
        phase = phase%1
    return phase