import matplotlib.pyplot as plt

from copy import deepcopy

from holograms.arrays import aags
from holograms.zernike import remove_low_zernikes, zernike_decomp, zernike

#%%
zernike_params = [[0,0,1],
                  [1,-1,0.3],
                  [2,-2,-0.5],
                  [3,-3,0.4],
                  [3,-1,-2],
                  [4,0,0.7],
                  [4,4,-0.2]]

holo_zernike = (zernike(radial=0,azimuthal=0,amplitude=0))
          
for params in zernike_params:
    holo_zernike += zernike(*params)
df = zernike_decomp(holo_zernike,max_radial=4)
df.index = [df['zernike_radial'].astype(int),df['zernike_azimuthal'].astype(int)]

del df['zernike_radial']
del df['zernike_azimuthal']

df['actual_amp'] = 0
for params in zernike_params:
    df.loc[(params[0],params[1]),'actual_amp'] = params[-1]

df = df.reset_index()
print(df)

#%%
xticks = ['({},{})'.format(r,a) for r,a in zip(df['zernike_radial'],df['zernike_azimuthal'])]

plt.scatter(df.index,df['actual_amp'],label='actual')
plt.scatter(df.index,df['fit_amp'],marker='x',label='fit')
plt.legend()#title='amplitude')
plt.xticks(df.index,xticks,rotation='vertical')
plt.xlabel('Zernike polynomial')
plt.ylabel('amplitude')
plt.show()

#%%
holo_full = aags()
df_full = zernike_decomp(holo_full,max_radial=8)

holo_rem = deepcopy(holo_full)
holo_rem = remove_low_zernikes(holo_rem,max_radial=8)
df_rem = zernike_decomp(holo_rem,max_radial=8)

print('full Zernike decomp')
print(df_full)

print('modified Zernike decomp')
print(df_rem)

#%%
cmap='twilight_shifted'
fig,axs = plt.subplots(1,3)
fig.set_dpi(300)

for ax,holo,title in zip(axs,[holo_full,holo_rem,holo_rem-holo_full],['full hologram','removed low Z','difference']):
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

df = df_full
df['fit_amp_rem'] = df_rem['fit_amp']

xticks = ['({:.0f},{:.0f})'.format(r,a) for r,a in zip(df['zernike_radial'],df['zernike_azimuthal'])]
plt.scatter(df.index,df['fit_amp'],label='original',zorder=10)
plt.scatter(df.index,df['fit_amp_rem'],marker='x',label='modified',zorder=11)
# plt.axvline(0,c='k',alpha=0.5,zorder=-10)
plt.legend()#title='amplitude')
plt.xticks(df.index,xticks,rotation='vertical',fontsize=5)
plt.xlabel('Zernike polynomial')
plt.ylabel('amplitude')
plt.ylim(-0.1,0.1)
plt.grid()
plt.gcf().set_dpi(300)
plt.show()

print(df)