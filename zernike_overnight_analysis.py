import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log10,floor
from scipy.optimize import curve_fit

import holograms as hg

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

def quadratic(x,x0,a,offset):
    return a*(x-x0)**2+offset

def print_fit(popt,pcov):
    perr = np.sqrt(np.diag(pcov))
    for arg,val,err in zip(quadratic.__code__.co_varnames[1:],popt,perr):
        prec = floor(log10(err))
        err = round(err/10**prec)*10**prec
        val = round(val/10**prec)*10**prec
        if prec > 0:
            valerr = '{:.0f}({:.0f})'.format(val,err)
        else:
            valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
        print(arg,'=',valerr)

max_intensity_correction = 1

df = pd.read_csv(r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\25\zernike_measurements\combined.csv")
fit_df = pd.read_csv(r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\25\zernike_measurements\phase_image_fit.csv")
fit_df = fit_df.sort_values(['zernike_radial','zernike_azimuthal'])
fit_df = fit_df[:15]
df = df.sort_values(['zernike_radial','zernike_azimuthal'])
df['polynomial'] = list(zip(df['zernike_radial'],df['zernike_azimuthal']))
polys = df['polynomial'].unique()

results_df = pd.DataFrame()

for poly in polys:
    poly_df = df[df['polynomial']==poly]
    results_row = pd.DataFrame()
    results_row.loc[0,'polynomial'] = ('({:.0f},{:.0f})'.format(poly[0],poly[1]))
    results_row.loc[0,'zernike_radial'] = poly[0]
    results_row.loc[0,'zernike_azimuthal'] = poly[1]
    print(poly_df)
    for i in range(max_intensity_correction+1):
        print('\nIntensity correction iteration',i)
        plt.errorbar(poly_df['zernike_amplitude'],poly_df['I0_'+str(i)+'_mean'],yerr=poly_df['I0_'+str(i)+'_std'],capsize=1.5,fmt='o',label=i,c='C'+str(i))
        popt,pcov = curve_fit(quadratic,poly_df['zernike_amplitude'],poly_df['I0_'+str(i)+'_mean'],sigma=poly_df['I0_'+str(i)+'_std'],absolute_sigma=True,p0=[0,0,180])
        perr = np.sqrt(np.diag(pcov))
        print_fit(popt,pcov)
        xfit = np.linspace(min(poly_df['zernike_amplitude']),max(poly_df['zernike_amplitude']),100)
        yfit = quadratic(xfit,*popt)
        plt.plot(xfit,yfit,c='C'+str(i))
        results_row.loc[0,'I_{:.0f}_opt_amp'.format(i)] = popt[0]
        results_row.loc[0,'I_{:.0f}_opt_amp_err'.format(i)] = perr[0]
    plt.ylabel('calibrated trap intensities after 5 corrections')
    plt.xlabel(results_row.loc[0,'polynomial']+' Zernike amplitude')
    plt.legend(title='$I$ corr. itr.')
    plt.ylim(100,200)
    plt.xlim(-0.21,0.21)
    plt.show()
    
    for i in range(max_intensity_correction+1):
        print('\nIntensity correction iteration',i)
        plt.scatter(poly_df['zernike_amplitude'],poly_df['radial_freq_'+str(i)+'_std']/poly_df['radial_freq_'+str(i)+'_mean'],label=i,c='C'+str(i))
        popt,pcov = curve_fit(quadratic,poly_df['zernike_amplitude'],poly_df['radial_freq_'+str(i)+'_std']/poly_df['radial_freq_'+str(i)+'_mean'],p0=[-0.06,1,15])
        perr = np.sqrt(np.diag(pcov))
        print_fit(popt,pcov)
        xfit = np.linspace(min(poly_df['zernike_amplitude']),max(poly_df['zernike_amplitude']),100)
        yfit = quadratic(xfit,*popt)
        plt.plot(xfit,yfit,c='C'+str(i))
        results_row.loc[0,'radial_freq_{:.0f}_opt_amp'.format(i)] = popt[0]
        results_row.loc[0,'radial_freq_{:.0f}_opt_amp_err'.format(i)] = perr[0]
    plt.xlabel(results_row.loc[0,'polynomial']+' Zernike amplitude')
    plt.ylabel(r'$\sigma_{\omega_r}/\mu_{\omega_r}$')
    plt.title('radial frequency $\omega_r \propto I_0/\sqrt{w_xw_y}$ distrbution')
    plt.legend(title='$I$ corr. itr.')
    plt.ylim(0,0.25)
    plt.xlim(-0.21,0.21)
    plt.show()
    
    results_df = results_df.append(results_row,ignore_index=True)

results_df['fit_amp'] = fit_df['fit_amp']
#results_df = results_df.sort_values(['zernike_radial','zernike_azimuthal'])
x_pos = [i for i, _ in enumerate(results_df['polynomial'])]
plt.bar(x_pos, results_df['I_1_opt_amp'],yerr=results_df['I_1_opt_amp_err'])
plt.xlabel("Zernike polynomial (radial,azimuthal) index")
plt.ylabel("coefficient")
plt.title("Zernike polynomial decomposition")
plt.xticks(x_pos, results_df['polynomial'])
plt.xticks(rotation=90)
plt.show()

#%%
x = np.arange(len(results_df['polynomial']))  # the label locations
width = 0.25  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width, results_df['fit_amp'], width, label='phase map decomposition')
#rects2 = ax.bar(x, results_df['I_0_opt_amp'], width, yerr=results_df['I_0_opt_amp_err'], label='I_0 optimisation',capsize=1.5,color='orangered')
rects2 = ax.bar(x, results_df['I_1_opt_amp'], width, yerr=results_df['I_1_opt_amp_err'], label='I_1 optimisation',capsize=1.5)#,color='lightsalmon')
rects2 = ax.bar(x+width, results_df['radial_freq_1_opt_amp'], width, yerr=results_df['radial_freq_1_opt_amp_err'], label='$\sigma_{\omega_r}/\mu_{\omega_r}$_1 optimisation',capsize=1.5)#,color='palegreen')
plt.xlabel("Zernike polynomial (radial,azimuthal) index")
plt.ylabel("coefficient")
#plt.title("Zernike polynomial decomposition")
ax.set_xticks(x)
ax.set_xticklabels(results_df['polynomial'])
ax.legend()
# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)
plt.xticks(rotation=90)
plt.ylim(-0.2,0.2)
plt.grid(axis='y')
fig.tight_layout()
plt.show()

#%%
center = (288,227)
radius = 210

phase_map = hg.blank()
I_correction = hg.blank()
radial_freq_correction = hg.blank()

for index, row in results_df.iterrows():
    radial = int(row['zernike_radial'])
    azimuthal = int(row['zernike_azimuthal'])
    print(radial,azimuthal)
    amp = row['radial_freq_1_opt_amp']
    phase_map += hg.zernike(radial,azimuthal,row['fit_amp'],center,radius,wrap_phase=False)
    I_correction += hg.zernike(radial,azimuthal,row['I_1_opt_amp'],center,radius,wrap_phase=False)
    radial_freq_correction += hg.zernike(radial,azimuthal,row['radial_freq_1_opt_amp'],center,radius,wrap_phase=False)

phase_map -= phase_map[227,288]
phase_map = phase_map%1
phase_map = hg.apertures.circ(phase_map,center,radius)
I_correction -= I_correction[227,288]
I_correction = I_correction%1
I_correction = hg.apertures.circ(I_correction,center,radius)
radial_freq_correction -= radial_freq_correction[227,288]
radial_freq_correction = radial_freq_correction%1
radial_freq_correction = hg.apertures.circ(radial_freq_correction,center,radius)

#%%
cmap='twilight_shifted'
fig,axs = plt.subplots(1,3)
(ax0,ax1,ax2) = axs
fig.set_dpi(300)
pcm = ax0.imshow(phase_map,cmap=cmap,interpolation='nearest',vmin=0,vmax=1)
ax0.axis('off')
ax0.set_title('phase map fit')
pcm = ax1.imshow(I_correction,cmap=cmap,interpolation='nearest',vmin=0,vmax=1)
ax1.axis('off')
pcm = ax2.imshow(radial_freq_correction,cmap=cmap,interpolation='nearest',vmin=0,vmax=1)
ax1.set_title(r'I_1 opt.')
ax2.axis('off')
ax2.set_title('$\sigma_{\omega_r}/\mu_{\omega_r}$_1 opt.')
fig.tight_layout()
cbar = fig.colorbar(pcm,ax=axs,ticks=[0,0.5,1],label='phase',shrink=0.6**2)
cbar.ax.set_yticklabels(['0','$\pi$','$2\pi$'])
fig.suptitle('Zernike optimisations (shifted to zero at centre)',y=0.8)
plt.show()

hg.misc.save(phase_map,r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\25\zernike_measurements\phase_fit.png")
hg.misc.save(I_correction,r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\25\zernike_measurements\I_opt.png")
hg.misc.save(radial_freq_correction,r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\25\zernike_measurements\radial_freq_opt.png")
    