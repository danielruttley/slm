import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import log10,floor,ceil
from shutil import copyfile
plt.style.use(r"Z:\Tweezer\People\Dan\code\dan.mplstyle")

def gaussian_beam(x,width,peak_intensity,center):
    return peak_intensity*np.exp(-2*(x-center)**2/width**2)

def gaussian_axial(z,z0,w0,zR):
    return w0*np.sqrt(1+((z-z0)/zR)**2)

csv_path = r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\June\24\Measure 14\images.csv"
df = pd.read_csv(csv_path)
copyfile('plotter.py',csv_path.split('images.csv')[0]+'plotter.py')

x = 'aper_center'
y = 'pixel_sum'

title = r'.'+csv_path.split('slm')[1]

popt,pcov = curve_fit(gaussian_beam,df[x],df[y],p0=[80,600000,220])
xfit = np.linspace(min(df[x]),max(df[x]),100)
yfit = gaussian_beam(xfit,*popt)
perr = np.sqrt(np.diag(pcov))
plt.scatter(df[x],df[y],alpha=0.8,label='$w_x$')
plt.plot(xfit,yfit)

plt.xlabel('ycentre of aperture [pixels]')
plt.ylabel('pixel_sum')
plt.title(title)
plt.show()

for arg,val,err in zip(gaussian_beam.__code__.co_varnames[1:],popt,perr):
    prec = floor(log10(err))
    err = round(err/10**prec)*10**prec
    val = round(val/10**prec)*10**prec
    if prec > 0:
        valerr = '{:.0f}({:.0f})'.format(val,err)
    else:
        valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
    print(arg,'=',valerr)
