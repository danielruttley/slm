"""Calculates the maximum amount of light that could go into the first-order
diffraction pattern as a fraction of the total amount of light reflected from
the SLM based on the hologram being applied.
"""

import numpy as np
from scipy.special import assoc_laguerre
import matplotlib.pyplot as plt
plt.style.use(r"Z:\Tweezer\People\Dan\code\dan.mplstyle")

class GaussianBeam():
    def __init__(self,waist,center):
        """A class defining the incoming Gaussian beam being shone onto the 
        SLM. The beam is assumed to be collimated.
        
        Parameters
        ----------
        waist : float
            the 1/e^2 waist of the beam, in SLM pixels
        center: tuple of float or int
            the center of the beam (x0,y0) on the SLM screen, in SLM pixels
        """
        
        self.waist = waist
        self.center = center

class Superposition():
    def __init__(self,ps,waist,center):
        self.ps = ps
        self.waist = waist
        self.center = center

class SLM():
    def __init__(self,input_beam,output_beam,pixel_size = 15e-6, resolution = (512,512)):
        """Class defining the SLM that light is reflected from.
        
        Parameters
        ----------
        input_beam : GaussianBeam
            input beam onto the SLM
        pixel_size : float
            the size of the SLM pixels [m]
        resolution : tuple of int
            resolution of the SLM screen (xpixels,ypixels)
        """
        
        self.xx,self.yy = np.meshgrid(np.arange(resolution[0]),np.arange(resolution[1]))
        
        input_I = self.input_intensity(input_beam)
        output_I = self.output_intensity(output_beam)
        
        rel_I = output_I/input_I
        super_threshold_indices = rel_I > 1
        rel_I[super_threshold_indices] = 1
        
        output_I = input_I*rel_I
        
        self.input_I = input_I
        self.output_I = output_I
        self.efficiency = np.sum(output_I)/np.sum(input_I)
        
    def input_intensity(self,beam):
        waist = beam.waist
        center = beam.center
        xx = self.xx - center[0]
        yy = self.yy - center[1]
        
        r = np.sqrt(xx**2+yy**2)
        I = np.exp(-2*r**2/waist**2)
        I = I/np.max(I)
        return I
    
    def output_intensity(self,beam):
        waist = beam.waist
        center = beam.center
        ps = beam.ps
        
        xx = self.xx - center[0]
        yy = self.yy - center[1]
        
        r = np.sqrt(xx**2+yy**2)
        
        field = np.zeros(r.shape,dtype=np.complex128)
        for p in ps:
            field += self.temp0_field(p,r,waist)
        I = np.abs(field)**2
        I /= np.max(I)

        self.radius = min(I.shape[1]-center[0],center[0],I.shape[0]-center[1],center[1])
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                if (i-center[1])**2+(j-center[0])**2 > self.radius**2:
                    I[i,j] = 0        
        return I
    
    def temp0_field(self,p,r,w0):
        xi = r/w0
        amp = (np.exp(-xi**2)*assoc_laguerre(2*xi**2,p))
        return amp

input_beam = GaussianBeam(336, (288,277))
output_beam = Superposition([0,2,4], 50, (288,277))

slm = SLM(input_beam,output_beam)

plt.pcolor(slm.input_I,cmap='viridis')
plt.colorbar()
plt.title('input Gaussian, center {}, waist {}'.format(input_beam.center,input_beam.waist))
plt.show()

plt.pcolor(slm.output_I,cmap='viridis')
plt.colorbar()
plt.title('input {} superpos., center {}, waist {}, circ. mask'.format(output_beam.ps,output_beam.center,output_beam.waist))
plt.show()

print(slm.efficiency)

#%%
output_widths = np.linspace(50,224,20)
efficiencies = []

for output_width in output_widths:
    output_beam = Superposition([0,2,4], output_width, (288,277))
    slm = SLM(input_beam,output_beam)
    efficiencies.append(slm.efficiency)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
ax1.scatter(output_widths,efficiencies)
ax1.set_ylabel('efficiency')
ax1.set_xlabel(r'superposition $1/e^2$ waist [pixels]')
fig.suptitle('maximum efficiencies with an input Gaussian, center {}, waist {}'.format(input_beam.center,input_beam.waist))
ax1.set_ylim(bottom=0)
ax1.set_xlim(np.min(output_widths),np.max(output_widths))
ax1.grid()

def tick_function(X):
    V = X/slm.radius
    return ["%.2f" % z for z in V]

ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(ax1.get_xticks())
ax1.set_xticks(ax1.get_xticks())
ax2.set_xticklabels(tick_function(ax1.get_xticks()))
ax2.set_xlabel("SLM filling factor")

plt.show()
        
        
        
        