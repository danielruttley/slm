"""Implements an adaptive additive Gerchberg Saxton algorithm as detailed in
https://arxiv.org/abs/1903.09286 to form an array of Gaussian traps.
"""

import numpy as np
from scipy.fft import fft2,ifft2,fftshift,ifftshift
import matplotlib.pyplot as plt

from .apertures import circ

def aags(traps,iterations=20,input_intensity=None,circ_aper_center=None,
         shape=(512,512)):
    """Calculates an adaptive additive Gerchberg Saxton hologram which creates 
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
    if input_intensity is None:
        shape = (512,512)
        input_intensity = np.ones(shape)
    else:
        shape = input_intensity.shape
    if circ_aper_center is not None:
        input_intensity = circ(input_intensity,circ_aper_center)
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

# plt.pcolor(phi)
# plt.colorbar()
# plt.show()
# u_plane = fft2(np.sqrt(I) * np.exp(1j*phi))
# u_plane = fftshift(u_plane)
# B = np.abs(u_plane)
# psi = np.angle(u_plane)
# array = np.abs(B)**2

# # plt.pcolor(array)
# # plt.colorbar()
# # plt.show()

# for trap in traps:
#     print(trap,'{:.3e}'.format(array[trap]))
# plt.pcolor(array)
# plt.colorbar()
# plt.show()
# plt.pcolor(psi)
# plt.colorbar()
# plt.show()