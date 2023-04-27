import matplotlib.pyplot as plt

from copy import deepcopy

from holograms.arrays import aags
from holograms.zernike import remove_low_zernikes, zernike_decomp, zernike

#%%
holo = aags()
holo2 = aags(padding_scale=2)


#%%
cmap='twilight_shifted'
fig,axs = plt.subplots(1,3)
fig.set_dpi(300)

for ax,holo,title in zip(axs,[holo,holo2],['1','2','difference']):
    pcm = ax.imshow(holo%1,cmap=cmap,interpolation='nearest',vmin=0,vmax=1)
    ax.set_xlabel('x [pixels]')
    ax.set_ylabel('y [pixels]')
    ax.axis('off')
    ax.set_title(title)

cbar = fig.colorbar(pcm,ax=axs,ticks=[0,0.5,1],label='phase',shrink=0.8**2)
cbar.ax.set_yticklabels(['0','$\pi$','$2\pi$'])
# plt.savefig('correction_holo.svg')
# fig.tight_layout()
plt.show()