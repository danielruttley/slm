"""Implements an adaptive additive Gerchberg Saxton algorithm as detailed in
https://arxiv.org/abs/1903.09286 to form an array of Gaussian traps.
"""

import numpy as np
from scipy.fft import fft2,ifft2,fftshift,ifftshift



def aags(traps='(256,256),(260,256),(256,260),(260,260)',iterations=20,
         beam_waist=None,beam_center=(256,256),shape=(512,512)):#,circ_aper_center=None,):
    """
    Calculates an adaptive additive Gerchberg Saxton hologram which creates 
    an array of Gaussian traps.

    Parameters
    ----------
    traps: list of tuple of int
        A list containing the (y,x) coordinates of the location of the traps in
        the imaging plane.
    iterations : int, optional
        The number of iterations that the AAGS algorithm should run for.
    input_intensity : array of float, optional
        The intensity of the input beam onto the SLM. This defines the 
        resolution of the SLM. If None, the shape parameter defines the SLM 
        resolution.
    shape : tuple of int
        The resolution of the SLM. Only used if input_intensity == None.
    
    Returns
    -------
    array
        The generated AAGS hologram.
    """
    if type(traps) == str:
        traps.replace('[','')
        traps.replace(']','')
        traps = '['+traps+']'
        traps = eval(traps)

    input_waist = beam_waist
    circ_aper_center = beam_center

    if input_waist is None:
        input_intensity = np.ones(shape)
    else:
        input_intensity = generate_input_intensity(waist=input_waist)
    print(input_intensity)
    if circ_aper_center is not None:
        input_intensity = circ(input_intensity,circ_aper_center[0],circ_aper_center[1])
    phi = (np.random.rand(*shape))*2*np.pi

    N = len(traps)
    print('generating {} traps'.format(N))
    T = np.zeros(shape)
    for trap in traps:
        T[trap] = 1
    prev_g = np.ones(shape)

    for i in range(iterations):
        print(i)
        u_plane = fftshift(fft2(np.sqrt(input_intensity)*np.exp(1j*phi)))
        B = np.abs(u_plane)
        psi = np.angle(u_plane)
        
        if i == N:
            psi_N = psi
            print('i=N')
        elif i > N:
            psi = psi_N
            print('i>N')
        
        B_N = 0
        for trap in traps:
            B_N += B[trap]
            print(trap,B[trap])
        B_N /= N
        
        g = np.zeros(shape)
        for trap in traps:
            g[trap] = B_N/B[trap]*prev_g[trap]
        B = T*g
        prev_g = g
        
        x_plane = ifft2(ifftshift(B * np.exp(1j*psi)))
        A = np.abs(x_plane)
        phi = np.angle(x_plane)

    return (phi%(2*np.pi))/2/np.pi

def generate_input_intensity(waist=None,center=None,shape=(512,512)):
        """Defines the Gaussian intensity incident onto the SLM and stores it 
        as a parameter.

        Parameters
        ----------
        waist : float, optional
            The 1/e^2 waist of the beam, in SLM pixels. if waist is None, then 
            a uniform intensity is assumed.
        center : tuple of float, optional
            The center of the beam, in SLM pixels. if center is None, the beam 
            is centered on the SLM.

        Returns
        -------
        array
        """
        xx,yy = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
        if waist is None:
            I = np.ones(shape)
        else:
            if center is None:
                center = tuple(x/2 for x in shape)
            xx = xx - center[0]
            yy = yy - center[1]
            r = np.sqrt(xx**2+yy**2)
            I = np.exp(-2*r**2/waist**2)
            I = I/np.max(I)
        I = circ(I)
        return I


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    from os import path
    sys.path.append(path.dirname(path.abspath(__file__)))
    from apertures import circ

    traps = [(256,256),(260,260),(256,260),(260,256)]
    holo = aags(traps,input_waist=210)
    plt.pcolor(holo)
    plt.colorbar()
    plt.show()
else:
    from holograms.apertures import circ