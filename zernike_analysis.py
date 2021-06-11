import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
from math import log10,ceil,floor

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

df = pd.DataFrame()
max_intensity_correction = 1

measures = np.arange(96,117)
for measure in measures:
    print(measure)
    measure_path = r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\25\Measure {}\\".format(measure)
    with open(measure_path+'params.json', 'r') as fp:
        params = json.load(fp)
    images = pd.read_csv(measure_path+'images.csv',index_col=0)
    trap_df = pd.read_csv(measure_path+'trap_df.csv',index_col=0)
    calibrations = images[images['calibration_pixel_sum'].notna()]
    trap_params = ['I0','x0','y0','wx','wy','w','radial_freq']
    for i in range(max_intensity_correction+1):
        trap_df['w_'+str(i)] = np.sqrt(trap_df['wx_'+str(i)]*trap_df['wy_'+str(i)])
        trap_df['radial_freq_'+str(i)] = trap_df['I0_'+str(i)]/np.sqrt(trap_df['wx_'+str(i)]*trap_df['wy_'+str(i)])
    measure_df = pd.DataFrame()
    measure_df['intensity_correction_iteration'] = sorted(images['intensity_correction_iteration'].unique())
    for index, row in measure_df.iterrows():
        iteration_calibrations = calibrations[calibrations['intensity_correction_iteration'] == row['intensity_correction_iteration']]
        measure_df.loc[index,'calibration'] = iteration_calibrations['calibration_pixel_sum'].mean()
        measure_df.loc[index,'calibration_err'] = iteration_calibrations['calibration_pixel_sum'].std()
        for param in trap_params:
            df.loc[params['zernike_amplitude'],param+'_'+str(int(row['intensity_correction_iteration']))+'_mean'] = trap_df[param+'_'+str(int(row['intensity_correction_iteration']))].mean()
            df.loc[params['zernike_amplitude'],param+'_'+str(int(row['intensity_correction_iteration']))+'_std'] = trap_df[param+'_'+str(int(row['intensity_correction_iteration']))].std()

print('\n===INTENSITY - ABSOLUTE VALUE===')
for i in range(max_intensity_correction+1):
    print('\nIntensity correction iteration',i)
    plt.errorbar(df.index,df['I0_'+str(i)+'_mean'],yerr=df['I0_'+str(i)+'_std'],fmt='o',label=i,c='C'+str(i))
    #plt.axvline(-0.06,c='k',linestyle='--')
    popt,pcov = curve_fit(quadratic,df.index,df['I0_'+str(i)+'_mean'],sigma=df['I0_'+str(i)+'_std'],absolute_sigma=True,p0=[0.075,0,180])
    print_fit(popt,pcov)
    xfit = np.linspace(min(df.index),max(df.index),100)
    yfit = quadratic(xfit,*popt)
    plt.plot(xfit,yfit,c='C'+str(i))
plt.ylabel('calibrated trap intensities after 5 corrections')
plt.xlabel('(2,2) Zernike amplitude')
plt.legend(title='$I$ corr. itr.')
plt.show()

print('\n===WIDTH - ABSOLUTE VALUE===')
for i in range(max_intensity_correction+1):
    print('\nIntensity correction iteration',i)
    plt.errorbar(df.index,df['w_'+str(i)+'_mean'],yerr=df['w_'+str(i)+'_std'],fmt='o',label=i,c='C'+str(i))
    #plt.axvline(-0.06,c='k',linestyle='--')
    #plt.ylim(bottom=0)
    popt,pcov = curve_fit(quadratic,df.index,df['w_'+str(i)+'_mean'],sigma=df['w_'+str(i)+'_std'],absolute_sigma=True,p0=[0.075,0,15])
    print_fit(popt,pcov)
    xfit = np.linspace(min(df.index),max(df.index),100)
    yfit = quadratic(xfit,*popt)
    plt.plot(xfit,yfit,c='C'+str(i))
plt.xlabel('(2,2) Zernike amplitude')
plt.ylabel('trap width $w=\sqrt{w_xw_y}$ distributions')
plt.legend(title='$I$ corr. itr.')
plt.show()


print('\n===WIDTH - RELATIVE ERROR===')
for i in range(max_intensity_correction+1):
    print('\nIntensity correction iteration',i)
    plt.scatter(df.index,df['w_'+str(i)+'_std']/df['w_'+str(i)+'_mean'],label=i,c='C'+str(i))
    #plt.axvline(-0.06,c='k',linestyle='--')
    #plt.ylim(bottom=0)
    popt,pcov = curve_fit(quadratic,df.index,df['w_'+str(i)+'_std']/df['w_'+str(i)+'_mean'],p0=[-0.06,1,15])
    print_fit(popt,pcov)
    xfit = np.linspace(min(df.index),max(df.index),100)
    yfit = quadratic(xfit,*popt)
    plt.plot(xfit,yfit,c='C'+str(i))
plt.xlabel('(2,2) Zernike amplitude')
plt.ylabel(r'$\sigma_w/\mu_w$')
plt.legend(title='$I$ corr. itr.')
plt.show()

print('\n===RADIAL FREQ - RELATIVE ERROR===')
for i in range(max_intensity_correction+1):
    print('\nIntensity correction iteration',i)
    plt.scatter(df.index,df['radial_freq_'+str(i)+'_std']/df['radial_freq_'+str(i)+'_mean'],label=i,c='C'+str(i))
    #plt.axvline(-0.06,c='k',linestyle='--')
    #plt.ylim(bottom=0)
    popt,pcov = curve_fit(quadratic,df.index,df['radial_freq_'+str(i)+'_std']/df['radial_freq_'+str(i)+'_mean'],p0=[-0.06,1,15])
    print_fit(popt,pcov)
    xfit = np.linspace(min(df.index),max(df.index),100)
    yfit = quadratic(xfit,*popt)
    plt.plot(xfit,yfit,c='C'+str(i))
plt.xlabel('(2,2) Zernike amplitude')
plt.ylabel(r'$\sigma_{\omega_r}/\mu_{\omega_r}$')
plt.title('radial frequency $\omega_r \propto I_0/\sqrt{w_xw_y}$ distrbution')
plt.legend(title='$I$ corr. itr.')
plt.show()
