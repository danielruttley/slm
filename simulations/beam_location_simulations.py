import numpy as np
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre
from scipy.optimize import curve_fit

def gaussian_beam(r,width,peak_intensity):
    return peak_intensity*np.exp(-2*r**2/width**2)

def amp_pl(p,l,r,phi,z,w0,wavelength):
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

x = np.linspace(-100,100,512)
y = np.linspace(-100,100,512)
xx,yy = np.meshgrid(x,y)
r = np.sqrt(xx**2+yy**2)
phi = np.arctan2(yy,xx)%(2*np.pi)

gaussian = gaussian_beam(r, 50, 1)
tem00 = amp_pl(0,0,r,phi,0,20,1064e-3)
tem00 = tem00 / np.max(tem00)
tem00 = np.abs(tem00)**2

plt.pcolor(x,y,gaussian,shading='auto',cmap='viridis')
plt.colorbar()
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$y$ ($\mu$m)')
plt.show()

width = 2
centers = np.arange(512)
integrals = []
test = 0
for center in centers:
    start = round(center-width/2)
    end = round(center+width/2)
    section = gaussian[:,start:end]
    integrals.append(np.sum(section))
integrals /= max(integrals)
centers = (centers/512-0.5)*200
width *= 200/512
plt.plot(centers,integrals,label=str(width))

width = 20
centers = np.arange(512)
integrals = []
for center in centers:
    start = round(center-width/2)
    end = round(center+width/2)
    section = gaussian[:,start:end]
    integrals.append(np.sum(section))
integrals /= max(integrals)
centers = (centers/512-0.5)*200
width *= 200/512
plt.plot(centers,integrals,label=str(width))

width = 100
centers = np.arange(512)
integrals = []
for center in centers:
    start = round(center-width/2)
    end = round(center+width/2)
    section = gaussian[:,start:end]
    integrals.append(np.sum(section))
integrals /= max(integrals)
centers = (centers/512-0.5)*200
width *= 200/512
plt.plot(centers,integrals,label=str(width))

width = 300
centers = np.arange(512)
integrals = []
for center in centers:
    start = round(center-width/2)
    if start < 0:
        start = 0
    end = round(center+width/2)
    if end > 511:
        end = 511
    section = gaussian[:,start:end]
    integrals.append(np.sum(section))
integrals /= max(integrals)
centers = (centers/512-0.5)*200
width *= 200/512
plt.plot(centers,integrals,label=str(width))

plt.xlabel('aperture center')
plt.ylabel('normalised intensity')
plt.title(r'$w_0$ = 20$\mu$m')
plt.legend(title=r'aperture width ($\mu$m)')
plt.show()

norm_widths = []
fitted_widths = []
fitted_widths_err = []
for width in [2,10,20,30,40,50,100,150,200]:
    print(width)
    centers = np.arange(512)
    integrals = []
    for center in centers:
        start = round(center-width/2)
        end = round(center+width/2)
        section = gaussian[:,start:end]
        integrals.append(np.sum(section))
    integrals /= max(integrals)
    centers = (centers/512-0.5)*200
    width *= 200/512
    
    norm_width = width/50
    popt, pcov = curve_fit(gaussian_beam,centers,integrals,p0=[20,1])
    perr = np.sqrt(np.diag(pcov))
    
    norm_widths.append(width/50)
    fitted_widths.append(popt[0]/50)
    fitted_widths_err.append(perr[0]/50)

plt.errorbar(norm_widths,fitted_widths,yerr=fitted_widths_err,fmt='o',alpha=0.8)
plt.xlabel('aperture width ($w_0$)')
plt.ylabel('fitted width ($w_0$)')
plt.show()