import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import fft2,ifft2,fftshift,ifftshift

def input_intensity(waist,center):
        xx,yy = np.meshgrid(np.arange(512),np.arange(512))
        xx = xx - center[0]
        yy = yy - center[1]
        
        r = np.sqrt(xx**2+yy**2)
        I = np.exp(-2*r**2/waist**2)
        I = I/np.max(I)
        return I

def gerchbert_saxton(input_intensity,desired_image,iterations):
    shape = desired_image.shape
    slm_amp = np.sqrt(input_intensity)
    slm_phase = np.random.rand(*shape)
    img_amp = np.sqrt(desired_image)
    img_phase = np.zeros(shape)
    
    for i in range(iterations):
        print(i)
        slm = slm_amp * np.exp(1j*slm_phase)
        img = fftshift(fft2(slm))
        img_phase = np.angle(img)
        img = img_amp * np.exp(1j*img_phase)
        slm = ifft2(ifftshift(img))
        slm_phase = np.angle(slm)
    
    return slm_phase

img = Image.open(r"Z:\Tweezer\Code\Python 3.7\slm\durham.png").convert('L')
array = np.asarray(img)
I = input_intensity(336, (288,277))
holo = gerchbert_saxton(I,array,100)

plt.pcolor(img)
plt.colorbar()
plt.show()

plt.pcolor(holo)
plt.colorbar()
plt.show()

cam_img = np.abs(fft2(np.sqrt(I)*np.exp(1j*holo)))**2
cam_img /= np.max(cam_img)
plt.pcolor(cam_img)
plt.colorbar()
plt.show()