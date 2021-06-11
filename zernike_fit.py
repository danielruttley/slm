import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pandas as pd

import holograms as hg

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

center = (288,227)
radius = 210
ones = np.ones((512,512))
ones = hg.apertures.circ(ones,center,radius)
num_pixels = np.count_nonzero(ones)

zernike = hg.zernike(0,0,1,center,radius,wrap_phase=False)

correction = hg.misc.load(r"Z:\Tweezer\Code\Python 3.7\slm\phase_correction.png")
shifted_correction = np.where(correction>0.5,correction-1,correction)

cmap='twilight_shifted'
#cmap='coolwarm'
fig,ax1 = plt.subplots(1,1)
fig.set_dpi(300)
pcm = ax1.imshow(zernike,cmap=cmap,interpolation='nearest',vmin=-1,vmax=1)
ax1.axis('off')
fig.tight_layout()
cbar = fig.colorbar(pcm,ax=ax1,ticks=[0,0.5,1],label='phase')
cbar.ax.set_yticklabels(['0','$\pi$',r'$2\pi$'])
plt.show()

fig,ax1 = plt.subplots(1,1)
fig.set_dpi(300)
pcm = ax1.imshow(correction,cmap=cmap,interpolation='nearest',vmin=0,vmax=1)
ax1.axis('off')
fig.tight_layout()
cbar = fig.colorbar(pcm,ax=ax1,ticks=[0,0.5,1],label='phase')
cbar.ax.set_yticklabels(['0','$\pi$','$2\pi$'])
plt.show()

cmap='twilight_r'
fig,ax1 = plt.subplots(1,1)
fig.set_dpi(300)
pcm = ax1.imshow(shifted_correction,cmap=cmap,interpolation='nearest',vmin=-0.5,vmax=0.5)
ax1.axis('off')
fig.tight_layout()
cbar = fig.colorbar(pcm,ax=ax1,ticks=[-0.5,0,0.5],label='phase')
cbar.ax.set_yticklabels(['$-\pi$','0','$\pi$'])
plt.show()

cmap='coolwarm'
fig,ax1 = plt.subplots(1,1)
fig.set_dpi(300)
pcm = ax1.imshow(zernike*shifted_correction,cmap=cmap,interpolation='nearest')
ax1.axis('off')
fig.tight_layout()
cbar = fig.colorbar(pcm,ax=ax1)
plt.show()

#%%
reconstructed = np.zeros_like(shifted_correction)

#shifted_correction = hg.zernike(2,2,0.25,center,radius,wrap_phase=False)

results_df = pd.DataFrame()

plot_strs = []
coeffs = []
fit_coeffs = [0,0,0,0,0.063,-0.048,0,0,0,-0.02,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
std = []

for radial in range(7):
    for azimuthal in np.arange(-radial,radial+2,2):
        results_row = pd.DataFrame()
        results_row.loc[0,'zernike_radial'] = radial
        results_row.loc[0,'zernike_azimuthal'] = azimuthal
        zernike = hg.zernike(radial,azimuthal,1,center,radius,wrap_phase=False)
        contrib = shifted_correction*zernike
        coeff = np.sum(shifted_correction*zernike)/np.sum(zernike**2)
        reconstructed += hg.zernike(radial,azimuthal,coeff,center,radius,wrap_phase=False)
        print(radial,azimuthal,coeff)
        plot_strs.append('({},{})'.format(radial,azimuthal))
        coeffs.append(coeff)
        std.append(np.std(reconstructed-shifted_correction))
        results_row.loc[0,'fit_amp'] = coeff
        results_df = results_df.append(results_row)
        
results_df.to_csv(r'Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\25\zernike_measurements\phase_image_fit.csv',index=False)
        

cmap='twilight_r'
#reconstructed *= np.max(shifted_correction)/np.max(reconstructed)
fig,axs = plt.subplots(1,3)
(ax0,ax1,ax2) = axs
fig.set_dpi(300)
pcm = ax0.imshow(shifted_correction,cmap=cmap,interpolation='nearest',vmin=-0.5,vmax=0.5)
ax0.axis('off')
ax0.set_title('measured')
pcm = ax1.imshow(reconstructed,cmap=cmap,interpolation='nearest',vmin=-0.5,vmax=0.5)
ax1.axis('off')
pcm = ax2.imshow(reconstructed-shifted_correction,cmap=cmap,interpolation='nearest',vmin=-0.5,vmax=0.5)
ax1.set_title(r'fit $n\leq 10$')
ax2.axis('off')
ax2.set_title('fit $-$ measured')
fig.tight_layout()
cbar = fig.colorbar(pcm,ax=axs,ticks=[-0.5,0,0.5],label='phase',shrink=0.6**2)
cbar.ax.set_yticklabels(['$-\pi$','0','$\pi$'])
fig.suptitle('Fitting phase map with Zernike polynomials',y=0.8)
plt.show()

x_pos = [i for i, _ in enumerate(plot_strs)]
plt.bar(x_pos, coeffs)
plt.xlabel("Zernike polynomial (radial,azimuthal) index")
plt.ylabel("coefficient")
plt.title("Zernike polynomial decomposition")
plt.xticks(x_pos, plot_strs)
plt.xticks(rotation=90)
plt.show()

x = np.arange(len(plot_strs))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, coeffs, width, label='phase map decomposition')
rects2 = ax.bar(x + width/2, fit_coeffs, width, label='Zernike fitting')
plt.xlabel("Zernike polynomial (radial,azimuthal) index")
plt.ylabel("coefficient")
plt.title("Zernike polynomial decomposition")
ax.set_xticks(x)
ax.set_xticklabels(plot_strs)
ax.legend()
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
plt.xticks(rotation=90)
fig.tight_layout()
plt.show()

x_pos = [i for i, _ in enumerate(plot_strs)]
plt.scatter(x_pos, std)
plt.xlabel("most recent Zernike polynomial (radial,azimuthal) index")
plt.ylabel("standard deviation of fit $-$ measured")
plt.title("Zernike polynomial error")
plt.xticks(x_pos, plot_strs)
plt.xticks(rotation=90)
plt.show()

# cmap='twilight_r'
# #reconstructed *= np.max(shifted_correction)/np.max(reconstructed)
# fig,ax1 = plt.subplots(1,1)
# fig.set_dpi(300)
# pcm = ax1.imshow(shifted_correction,cmap=cmap,interpolation='nearest')#,vmin=-0.5,vmax=0.5)
# ax1.axis('off')
# fig.tight_layout()
# cbar = fig.colorbar(pcm,ax=ax1)#,ticks=[-0.5,0,0.5],label='phase')
# plt.show()
