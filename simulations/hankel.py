import numpy as np
from scipy.fft import fft2, fftshift, fftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pyhank import HankelTransform
from scipy.special import assoc_laguerre
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def tem_field(p,l,r,phi,z,w0,wavelength):
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

def superposition_field(ps,waist,r,gaussian_width_factor):
    wavelength = 1064e-9
    phi = 0

    field = np.zeros(r.shape,dtype=np.complex128)
    for p in ps:
        field += tem_field(p,0,r,phi,0,waist,wavelength)
    gaussian = tem_field(0,0,r,phi,0,waist*gaussian_width_factor,wavelength)
    gaussian /= np.max(gaussian)
    field /= np.max(field)
    
    return field,gaussian

slm_pixel = 15e-6
camera_pixel = 5.2e-6
n = 512
nr = 1000
wavelength = 1064e-9
f = 500e-3

r = np.linspace(0, 512/2, nr)
H = HankelTransform(order=0, radial_grid=r)

w0 = 10
gaussian_width_factor = 2
Er,gaussian = superposition_field([0,2,4],w0,r,gaussian_width_factor)
ErH = H.to_transform_r(Er)
EkrH = H.qdht(ErH)
EkrH /= np.max(EkrH)

intensity = np.abs(EkrH)**2
intensity_peaks_ind = find_peaks(intensity)[0]
intensity_peaks_loc = r[intensity_peaks_ind]/w0
intensity_peaks = intensity[intensity_peaks_ind]
for loc,val in zip(intensity_peaks_loc,intensity_peaks):
    print('{:.2f} w0 = {:.2f} % of max. intensity'.format(loc,val*100))

f, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2)
ax1.plot(r/w0,np.abs(Er))
ax1.plot(r/w0,np.abs(gaussian),linestyle='--',alpha=0.5)
adjusted = np.abs(Er)/np.abs(gaussian)
super_threshold_indices = adjusted > 1
adjusted[super_threshold_indices] = 1
ax1.plot(r/w0,adjusted,linestyle='--',alpha=0.5)
ax1.set_xlim(0,6)
ax1.set_ylabel(r'$|U|$')
ax1.set_xlabel(r'$r$ ($w_0$)')
ax1.set_title(r'$U(r)$')
ax2.plot(np.abs(EkrH))
ax2.set_xlim(0,100)
ax2.set_ylabel(r'$|\widetilde{U}|$')
ax2.set_xlabel(r'$k_r$ (arb.)')
ax2.set_title(r'$\widetilde{U}(k_r)$')
ax3.plot(r/w0,np.angle(Er))
ax3.set_xlim(0,6)
ax3.set_ylabel(r'$\angle U$')
ax3.set_xlabel(r'$r$ ($w_0$)')
ax4.plot(np.angle(EkrH))
ax4.set_xlim(0,100)
ax4.set_ylabel(r'$\angle \widetilde{U}$')
ax4.set_xlabel(r'$k_r$ (arb.)')
f.suptitle('p 0+2+4 superposition, incoming Gaussian ${}w_0$'.format(str(gaussian_width_factor)))
f.tight_layout()
plt.show()

peakss = []
peakss_loc = []
inputs = []
wholo = 20
Er,gaussianholo = superposition_field([0,2,4],wholo/6,r,6)
adjusted = np.abs(Er)/np.abs(gaussianholo)
super_threshold_indices = adjusted > 1
adjusted[super_threshold_indices] = 1
plt.plot(adjusted)
plt.show()
wins = np.arange(4,50,4)
for win in wins:
    gaussianin = np.exp(-r**2/win**2)
    EafterSLM = adjusted*gaussianin
    ErH = H.to_transform_r(EafterSLM)
    inputs.append(EafterSLM)
    EkrH = H.qdht(ErH)
    EkrH /= np.max(EkrH)
    intensity = np.abs(EkrH)**2
    intensity_peaks_ind = find_peaks(intensity)[0]
    intensity_peaks_loc = H.kr[intensity_peaks_ind]
    intensity_peaks = intensity[intensity_peaks_ind]
    peakss_loc.append(intensity_peaks_ind)
    peakss.append(intensity_peaks)
    for peak in intensity_peaks:
        print('{:.2f} % of max. intensity'.format(peak*100))
    if win == wholo:
        plt.plot(intensity*100,label='{:.1f}'.format(win/wholo),linewidth=2)
    else:
        plt.plot(intensity*100,label='{:.1f}'.format(win/wholo),alpha=0.4)
plt.legend(title=r'$w_{in}/w_{holo}$',bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlim(0,150)
plt.ylim(0,20)
for loc,peaks in zip(peakss_loc,peakss):
    #peak_num = np.arange(start=1,stop=peaks.size+1)
    plt.scatter(loc,peaks*100,marker='x')
plt.ylabel(r'normalised intensity (%)')
plt.xlabel(r'$k_r$ (arb.)')
plt.show()

for _input in inputs:
    plt.plot(_input)
plt.show()

peakss = []
peakss_loc = []
Er,gaussian = superposition_field([0,2,4],w0,r,gaussian_width_factor)
effective_widths = np.linspace(10,100,10)
for effective_width in effective_widths:
    print(effective_width)
    gaussian = np.exp(-r**2/effective_width**2)
    ErH = H.to_transform_r(Er*gaussian)
    EkrH = H.qdht(ErH)
    EkrH /= np.max(EkrH)
    plt.plot(np.real(EkrH),label='{:.1f}'.format(effective_width))
    plt.legend()
plt.xlim(0,100)
for loc,peaks in zip(peakss_loc,peakss):
    #peak_num = np.arange(start=1,stop=peaks.size+1)
    plt.scatter(loc,peaks,marker='x')
plt.legend(title=r'$w_{eff}$')
plt.ylabel('normalised amplitude')
plt.xlabel(r'$k_r$ (arb.)')
plt.show()
    

# x = np.linspace(-512/2*15e-6,512/2*15e-6,n)
# y = np.linspace(-512/2*15e-6,512/2*15e-6,n)
# xx,yy = np.meshgrid(x,y)
# r = np.sqrt(xx**2+yy**2)
# phi = np.arctan2(yy,xx)%(2*np.pi)
# field = np.zeros(xx.shape,dtype=np.complex128)
# for p in [0,2,4]:
#     field += tem_field(p,0,r,phi,0,100*15e-6,1064e-9)
# field /= np.max(np.abs(field))
# tem = tem_field(0, 0, r, phi, 0, 100*15e-6*gaussian_width_factor, 1064e-9)
# tem /= np.max(np.abs(tem))
# A = np.abs(field)/np.abs(tem)
# super_threshold_indices = A > 1
# A[super_threshold_indices] = 1
# plt.pcolor(A,cmap='viridis')
# plt.colorbar()
# plt.show()
# plt.pcolor(np.abs(field)**2,cmap='gray')
# plt.colorbar()
# plt.show()