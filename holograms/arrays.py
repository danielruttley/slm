"""Implements an adaptive additive Gerchberg Saxton algorithm as detailed in
https://arxiv.org/abs/1903.09286 to form an array of Gaussian traps.
"""

import numpy as np
from scipy.fft import fft2,ifft2,fftshift,ifftshift
from .zernike import remove_low_zernikes
from .apertures import circ
from .mixing import mix
from .gratings import hori_gradient, vert_gradient

def pad(array,padding_scale):
    """Pads an array with zeros around it. The padding_scale defines the output
    array size relative to the input array size."""
    padding_scale = int(padding_scale)
    padded_array = array.copy()
    if padding_scale > 1:
        padding_index = (np.array(array.shape)/2*(padding_scale-1)).astype('int')
        padding_index = [[x]*2 for x in padding_index]
        padded_array = np.pad(padded_array,padding_index,constant_values=0)
    return padded_array

def unpad(array,padding_scale):
    """Reverses a pad applied with the pad command. Parameters are the same."""
    padding_scale = int(padding_scale)
    unpadded_array = array.copy()
    if padding_scale > 1:
        unpad_idx = (np.array(array.shape)*(0.5-1/2/padding_scale)).astype('int')
        unpadded_array = array[unpad_idx[0]:-unpad_idx[0],unpad_idx[1]:-unpad_idx[1]]
    return unpadded_array

def aags(traps='(256,256),(260,256),(256,260),(260,260)',iterations=20,
         padding_scale=1,beam_waist=None,beam_center=(256,256),shape=(512,512),
         remove_low_zernike=False):#,circ_aper_center=None,):
    """
    Calculates an adaptive additive Gerchberg Saxton hologram which creates 
    an array of Gaussian traps.

    Parameters
    ----------
    traps: list of tuple of int
        A list containing the (y,x) coordinates of the location of the traps in
        the imaging plane. Optionally the tuple can include a third term for 
        the intensity (not amplitude) of that trap. The default is 1 if not
        specified.
    iterations : int, optional
        The number of iterations that the AAGS algorithm should run for.
    padding_scale : int, optional
        The amount of padding that should be performed when making the array.
        The spatial resolution at the atom plane is increased by this factor.
        See https://doi.org/10.1364/OE.27.002184 for more details.
        The default is 1 (for no padding). Values smaller than 1 will have no
        effect.
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

    padding_scale = int(padding_scale)

    print(traps)

    input_waist = beam_waist
    circ_aper_center = beam_center

    if input_waist is None:
        input_intensity = np.ones(shape)
    else:
        input_intensity = generate_input_intensity(waist=input_waist)
    print(input_intensity)
    if circ_aper_center is not None:
        input_intensity = circ(input_intensity,circ_aper_center[0],circ_aper_center[1])
        
    np.random.seed(1064)
    phi = (np.random.rand(*shape))*2*np.pi

    N = len(traps)
    print('generating {} traps'.format(N))
    T = np.zeros(shape)
    for trap in traps:
        try:
            T[trap[:2]] = np.sqrt(trap[2]) # sqrt so that intensity is specified rather than amp
            print('trap',trap, f'using {trap[2]}')
        except IndexError: # default amp to 1 if not specified
            T[trap[:2]] = 1
        except ZeroDivisionError: # ignore this trap if there is a zero in the amplitude
            pass
            print('trap',trap,'set to 1')
    prev_g = np.ones(shape)

    for i in range(iterations):
        print(i)
        input_complex_amp = np.sqrt(input_intensity)*np.exp(1j*phi)
        input_complex_amp = pad(input_complex_amp,padding_scale)
        u_plane = fftshift(fft2(input_complex_amp))
        B_padded = np.abs(u_plane)
        psi = np.angle(u_plane)
        
        if i == N:
            psi_N = psi
            print('i=N')
        elif i > N:
            psi = psi_N
            print('i>N')

        print(B_padded.shape)
        B = unpad(B_padded,padding_scale)
        print(B.shape)

        B_N = 0
        for trap in traps:
            try: # B_N is the sum of all the traps so have to scale by the expected amplitude
                B_N += B[trap[:2]]*np.sqrt(trap[2])
            except IndexError:
                B_N += B[trap[:2]]
            except ZeroDivisionError:
                pass # ignore this trap if there is a zero in the amplitude
            print(trap,B[trap[:2]]**2) # square to show intensity rather than amp
        B_N /= N
        
        g = np.zeros(shape)
        for trap in traps:
            try: # if the intensity of the trap is what is expected, the g correction should be B_N
                g[trap[:2]] = B_N/(B[trap[:2]]/np.sqrt(trap[2]))*prev_g[trap[:2]]#*np.sqrt(trap[2])
            except IndexError:
                g[trap[:2]] = B_N/B[trap[:2]]*prev_g[trap[:2]]
        B = T*g
        prev_g = g
        
        B_padded = pad(B,padding_scale)

        trap_complex_amp = B_padded * np.exp(1j*psi)
        x_plane_padded = ifft2(ifftshift(trap_complex_amp))

        x_plane = unpad(x_plane_padded,padding_scale)

        A = np.abs(x_plane)
        phi = np.angle(x_plane)

    phi = circ(phi,beam_center[0],beam_center[1])
    if remove_low_zernike:
        print('removing low Zernike polynomials')
        phi = remove_low_zernikes(phi,center=beam_center)

    return phi/2/np.pi

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
        print('input waist',waist)
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

def mixed_array(traps='(0,0,1),(1,0,1)',shape=(512,512)):
    """Generates an array of traps by randomly mixing the holograms defining
    each trap. Each trap should have defined a horizontal and vertical grating
    and a weight to be used when mixing the traps.

    Parameters
    ----------
    traps : list of tuple of floats, optional
        List of the traps to be used of the form 
        [[horizontal grating, vertical grating, depth],...], by default [0,0,1]
    shape : tuple of int, optional
        The resolution of the SLM, by default (512,512)
    """
    if type(traps) == str:
        traps.replace('[','')
        traps.replace(']','')
        traps = '['+traps+']'
        traps = eval(traps)

    holos = []
    weights = []
    for trap in traps:
        holos.append((hori_gradient(trap[0],shape=shape) + vert_gradient(trap[1],shape=shape))%1)
        weights.append(trap[2])
    holo = mix(holos,weights)
    return holo

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