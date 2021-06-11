import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

import holograms as hg
center = (288,227)
waist = 336

complex_amp_mod = hg.ComplexAmpMod()

blazing = hg.gratings.diag_old(-12)
focal_planes = np.linspace(0.4,0.8,201)
#np.random.shuffle(focal_planes)
center = (288,227)


holo = complex_amp_mod.superposition_holo([0,2,4],center,78,blazing,336,0.614)
holo = hg.apertures.circ(holo,center)
    
print("program completed")

cmap='twilight_shifted'
fig,ax1 = plt.subplots(1,1)
fig.set_dpi(300)
pcm = ax1.imshow(holo,cmap=cmap,interpolation='nearest',vmin=0,vmax=1)
ax1.set_xlabel('x [pixels]')
ax1.set_ylabel('y [pixels]')
fig.tight_layout()
plt.axis('off')

cbar = fig.colorbar(pcm,ax=ax1,ticks=[0,0.5,1],label='phase')
cbar.ax.set_yticklabels(['0','$\pi$',r'$2\pi$'])
plt.savefig('superposition_holo.svg')
plt.show()