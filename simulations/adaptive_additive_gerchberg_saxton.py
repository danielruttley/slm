import numpy as np
from scipy.fft import fft2,ifft2,fftshift,ifftshift
import matplotlib.pyplot as plt


np.random.seed(1000)
I = np.ones((512,512))
phi = (np.random.rand(512,512))*2*np.pi

T = np.zeros((512,512))
traps = []
# for x in range(200,400,100):
#     for y in range(200,400,100):
#         traps.append((x,y))
traps = [(156,156),(156,356),(356,156),(356,356)]
N = len(traps)
for trap in traps:
    T[trap] = 1
gs = [np.ones((512,512))]

def gerchbert_saxton(desired_image,iterations):
    shape = desired_image.shape
    slm_amp = np.ones(shape)
    slm_phase = np.random.rand(shape)
    img_amp = np.sqrt(desired_image)
    img_phase = np.zeros(shape)
    
    for i in range(iterations):
        slm = slm_amp * np.exp(1j*slm_phase)
        img = fft2(slm)
        img_phase = np.angle(img)
        img = img_amp * np.exp(1j*img_phase)
        slm = ifft2(img)
        slm_phase = np.angle(slm)
    
    return slm_phase

for i in range(20):
    print(i)
    u_plane = fft2(np.sqrt(I) * np.exp(1j*phi))
    u_plane = fftshift(u_plane)
    B = np.abs(u_plane)
    psi = np.angle(u_plane)
    
    # if i == N:
    #     psi_N = psi
    #     print('i=N')
    # elif i > N:
    #     psi = psi_N
    #     print('i>N')
    
    B_N = 0
    for trap in traps:
        B_N += B[trap]
    B_N /= N
    
    g = np.zeros((512,512))
    prev_g = gs[-1]
    for trap in traps:
        g[trap] = B_N/B[trap]*prev_g[trap]
        #print(trap,B[trap],B_N/B[trap],g[trap])
    B = T*g
    gs.append(g)
    
    
    u_plane = ifftshift(B * np.exp(1j*psi))
    x_plane = ifft2(u_plane)
    A = np.abs(x_plane)
    phi = np.angle(x_plane)

# plt.pcolor(phi)
# plt.colorbar()
# plt.show()
u_plane = fft2(np.sqrt(I) * np.exp(1j*phi))
u_plane = fftshift(u_plane)
B = np.abs(u_plane)
psi = np.angle(u_plane)%(2*np.pi)
array = np.abs(B)**2

# plt.pcolor(array)
# plt.colorbar()
# plt.show()

for trap in traps:
    print(trap,'{:.3e}'.format(array[trap]))
plt.pcolor(array[150:250,190:210])
plt.colorbar()
plt.show()
plt.pcolor(psi)
plt.colorbar()
plt.show()