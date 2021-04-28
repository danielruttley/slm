import numpy as np
from scipy.special import assoc_laguerre
import bisect
from .kinoforms import fresnel_lens

class ComplexAmpMod():
    """Class containing the object required to make a complex amplitude 
    modulation hologram as detailed in https://doi.org/10.1073/pnas.2014017117
    """
    def __init__(self,xsize=512,ysize=512,wavelength=1064e-9):
        x = range(xsize)
        y = range(ysize)
        self.xx,self.yy = np.meshgrid(x,y)
        self.wavelength = wavelength
        self.sinc_dict = {}
        xs = np.linspace(-np.pi,0,100000)
        for x in xs:
            self.sinc_dict[np.sinc(x/np.pi)] = x
        self.sinc_keys = list(self.sinc_dict.keys())

    def inverse_sinc(self,x):
        ind = bisect.bisect_left(self.sinc_keys, x)
        if ind == len(self.sinc_keys):
            ind -= 1
        key = self.sinc_keys[ind]
        return self.sinc_dict[key]

    def tem_field(self,p,l,r,phi,z,w0,wavelength):
        zR = np.pi*w0**2/wavelength
        w = w0*np.sqrt(1+(z/zR)**2)
        xi = r/w
        amp = ((-1)**p
            *np.sqrt(2/np.pi*np.math.factorial(p)/(np.math.factorial(p+np.abs(l))))
            *(np.sqrt(2)*xi)**np.abs(l)/w
            *np.exp(-xi**2)
            *assoc_laguerre(2*xi**2,p,np.abs(l))
            *np.exp(-1j*l*phi)
            *np.exp(-1j*xi**2*z/zR)
            *np.exp(1j*(2*p+np.abs(l)+1)*np.arctan2(z,zR)))
        return amp

    def superposition_holo(self,ps,center,waist,blazing,input_waist=None,
                           focal_plane=None,custom=False):
        """Generates a hologram for a superposition of radial TEMp0 modes.

        Parameters:
            ps: A list of the p values in the superposition. Each entry should 
                be an integer.
            center: tuple containing the center of the hologram on the SLM 
                    screen in pixels in the form (x0,y0)
            waist:  the TEM waist of the hologram in SLM pixels
            blazing:    the blazing hologram which is applied to any hologram
                        to position the first diffraction order on the camera
                        correctly
            input_waist:    the waist of the TEM00 mode being shone onto the 
                            SLM. If None, a top-hat beam is assumed (i.e. 
                            infinite waist)
            focal_plane:    the location to shift the focal plane to. A Fresnel 
                            lens is generated and applied to the total phase
                            before calculating the CAM hologram. None to apply
                            no lens.
        
        Returns:
            holo:   a complex amplitude modulation hologram between 0-1 to be 
                    applied to the SLM screen
        """
        

        r = np.sqrt((self.xx-center[0])**2+(self.yy-center[1])**2)
        phi = np.arctan2(self.yy-center[1],self.xx-center[0])%(2*np.pi)

        field = np.zeros(self.xx.shape,dtype=np.complex128)
        for p in ps:
            field += self.tem_field(p,0,r,phi,0,waist,self.wavelength)
        field /= np.max(field)

        if input_waist != None:
            input_field = self.tem_field(0,0,r,phi,0,input_waist,self.wavelength)
            input_field /= np.max(input_field)
            field = field/input_field
            A = np.abs(field)
            super_threshold_indices = A > 1
            A[super_threshold_indices] = 1
        else:
            A = np.abs(field)
        phase = (np.angle(field))%(2*np.pi)
        print(np.max(phase))
        print(np.min(phase))

        if focal_plane != None:
            lens = fresnel_lens(focal_plane,center,self.wavelength)*2*np.pi
            phase = (phase+lens)%(2*np.pi)

        print(np.max(phase))
        print(np.min(phase))

        print(np.max(A))
        inverse_sinc = np.vectorize(self.inverse_sinc)
        M = 1+inverse_sinc(A)/np.pi

        print(np.max(M))
        print(np.min(M))

        F = phase-np.pi*M
        return M*((F+blazing*2*np.pi)%(2*np.pi))/2/np.pi
    
