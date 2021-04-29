import numpy as np
import matplotlib.pyplot as plt
from scipy.special import assoc_laguerre
import time
import bisect

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

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

def inverse_sinc(x):
    ind = bisect.bisect_left(keys, x)
    if ind == len(keys):
        ind -= 1
    key = keys[ind]
    return sinc_dict[key]

def bolduc_hologram(beam):
    A = np.abs(beam)#/np.abs(beam00)
    print(np.max(A))
    phase = (np.angle(beam)+blazingx)%(2*np.pi)
    inv_sinc = np.vectorize(inverse_sinc)
    M = 1+inv_sinc(A)/np.pi
    F = phase-np.pi*M
    return A,M,F,M*np.mod(F,2*np.pi)/2/np.pi

def bolduc_hologram_no_blazing(beam):
    A = np.abs(beam)#/np.abs(beam00)
    print(np.max(A))
    phase = (np.angle(beam))%(2*np.pi)
    inv_sinc = np.vectorize(inverse_sinc)
    M = 1+inv_sinc(A)/np.pi
    F = phase-np.pi*M
    return A,M,F,M*np.mod(F,2*np.pi)/2/np.pi
    
sinc_dict = {}
xs = np.linspace(-np.pi,0,100000)
for x in xs:
    sinc_dict[np.sinc(x/np.pi)] = x
keys = list(sinc_dict.keys())

#all distances in um
wavelength = 1064e-3
z = 0
w0 = 20
ps = [0,2,4]

k = 2*np.pi/wavelength

x = np.linspace(-100,100,512)
y = np.linspace(-100,100,512)
xx,yy = np.meshgrid(x,y)
r = np.sqrt(xx**2+yy**2)
phi = np.arctan2(yy,xx)%(2*np.pi)

# plt.pcolor(x,y,phi,shading='auto')
# plt.colorbar()
# plt.show()

# plt.pcolor(x,y,r,shading='auto')
# plt.colorbar()
# plt.show()

blazingx,blazingy = np.meshgrid(range(512),range(512))
blazingx = 2*np.pi*blazingx/12

beam = np.zeros(xx.shape,dtype=np.complex128)
title_str = ''

beam00 = amp_pl(0,0,r,phi,z,2*w0,wavelength)
beam00 /= np.max(beam00)
intensity00 = np.abs(beam00)**2

for i,p in enumerate(ps):
    beam += amp_pl(p,0,r,phi,z,w0,wavelength)
    title_str += str(p)
    if i != len(ps)-1:
        title_str += '+'
beam /= np.max(beam)
intensity = np.abs(beam)**2
phase = np.angle(beam)%(2*np.pi)
#phase = (np.angle(amp_pl(0,0,r,phi,z,w0,wavelength))+np.angle(amp_pl(2,0,r,phi,z,w0,wavelength))+np.angle(amp_pl(4,0,r,phi,z,w0,wavelength)))%(2*np.pi)

plt.plot(x/w0,np.real(amp_pl(0,0,x,0,z,w0,wavelength))/max(np.real(amp_pl(0,0,x,0,z,w0,wavelength))),label='TEM00',c='deepskyblue',alpha=0.8)
plt.plot(x/w0,np.real(amp_pl(2,0,x,0,z,w0,wavelength))/max(np.real(amp_pl(2,0,x,0,z,w0,wavelength))),label='TEM20',c='royalblue',alpha=0.8)
plt.plot(x/w0,np.real(amp_pl(4,0,x,0,z,w0,wavelength))/max(np.real(amp_pl(4,0,x,0,z,w0,wavelength))),label='TEM40',c='navy',alpha=0.8)
plt.plot(x/w0,np.real(amp_pl(0,0,x,0,z,w0,wavelength)+amp_pl(2,0,x,0,z,w0,wavelength)+amp_pl(4,0,x,0,z,w0,wavelength))/max(np.real(amp_pl(0,0,x,0,z,w0,wavelength)+amp_pl(2,0,x,0,z,w0,wavelength)+amp_pl(4,0,x,0,z,w0,wavelength))),label='p0+2+4',c='red')
plt.legend()
plt.axhline(0,c='k',linestyle='--',alpha=0.5)
plt.xlabel(r'x ($w_0$)')
plt.ylabel('normalised amplitude')
plt.title
plt.show()

plt.pcolor(x,y,intensity,shading='auto')
plt.colorbar()
plt.title(r'$p$ {} intensity; $w_0$ = {}$\mu$m, $\lambda$ = {}nm'.format(title_str,w0,int(wavelength*1e3)))
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$y$ ($\mu$m)')
plt.show()

plt.pcolor(x,y,phase,cmap='twilight_shifted',vmin=0,vmax=2*np.pi,shading='auto')
plt.colorbar()
plt.title(r'$p$ {} phase; $w_0$ = {}$\mu$m, $\lambda$ = {}nm'.format(title_str,w0,int(wavelength*1e3)))
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$y$ ($\mu$m)')
plt.tight_layout()
plt.show()

plt.plot(x,intensity00[256,:]/np.max(intensity00),label='TEM00')
plt.plot(x,intensity[256,:]/np.max(intensity),label ='p {}'.format(title_str))
plt.ylabel('normalised intensity')
plt.xlabel(r'$x$ ($\mu$m)')
plt.legend()
plt.show()

A,M,F,holo = bolduc_hologram(beam)
plt.pcolor(x,y,holo,cmap='twilight_shifted',vmin=0,vmax=1,shading='auto')
plt.colorbar()
plt.title(r'$p$ {} bolduc_hologram; $w_0$ = {}$\mu$m, $\lambda$ = {}nm'.format(title_str,w0,int(wavelength*1e3)))
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$y$ ($\mu$m)')
plt.tight_layout()
plt.show()

plt.pcolor(x,y,A,cmap='viridis',vmin=0,vmax=1,shading='auto')
plt.colorbar()
plt.title(r'$p$ {} bolduc_A; $w_0$ = {}$\mu$m, $\lambda$ = {}nm'.format(title_str,w0,int(wavelength*1e3)))
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$y$ ($\mu$m)')
plt.tight_layout()
plt.show()

# plt.plot(x,amp_pl(0,0,r,phi,z,w0,wavelength)[256,:]/np.max(intensity00),label='TEM00')
# plt.plot(x,amp_pl(2,0,r,phi,z,w0,wavelength)[256,:]/np.max(intensity00),label='TEM20')
# plt.plot(x,amp_pl(4,0,r,phi,z,w0,wavelength)[256,:]/np.max(intensity00),label='TEM40')
# plt.plot(x,beam[256,:]/np.max(intensity),label ='p {}'.format(title_str))
# plt.axhline(0,c='k',linestyle='--')
# plt.ylabel('normalised field amplitude')
# plt.xlabel(r'$x$ ($\mu$m)')
# plt.legend()
# plt.show()

A,M,F,holo = bolduc_hologram_no_blazing(beam)
plt.pcolor(x,y,holo,cmap='twilight_shifted',vmin=0,vmax=1,shading='auto')
plt.colorbar()
plt.title(r'$p$ {} bolduc_hologram; $w_0$ = {}$\mu$m, $\lambda$ = {}nm'.format(title_str,w0,int(wavelength*1e3)))
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$y$ ($\mu$m)')
plt.tight_layout()
plt.show()

plt.pcolor(x,y,A,cmap='viridis',vmin=0,vmax=1,shading='auto')
plt.colorbar()
plt.title(r'$p$ {} bolduc_A; $w_0$ = {}$\mu$m, $\lambda$ = {}nm'.format(title_str,w0,int(wavelength*1e3)))
plt.xlabel(r'$x$ ($\mu$m)')
plt.ylabel(r'$y$ ($\mu$m)')
plt.tight_layout()
plt.show()

fft = np.fft.fft2(holo)
fft_x = np.fft.fftfreq(phase.shape[0],x[1]-x[0])
fft_x = np.fft.fftshift(fft_x)
fft_y = np.fft.fftfreq(phase.shape[1],y[1]-y[0])
fft_y = np.fft.fftshift(fft_y)
fft_xx,fft_yy = np.meshgrid(fft_x,fft_y)
fft = np.fft.fftshift(fft)
fft_i = np.abs(fft)**2
plt.pcolor(fft_i,shading='auto')
plt.colorbar()
plt.show()

plt.plot(x,fft_i[256,:]/np.max(fft_i))
plt.ylabel('normalised intensity')
plt.xlabel(r'$x$ ($\mu$m)')
plt.show()