import pandas as pd
import numpy as np
import PIL.Image as PILImage
from scipy.optimize import curve_fit
from camera.camera_class import Image
import matplotlib.pyplot as plt
import os
from scipy.special import assoc_laguerre
from scipy.ndimage import center_of_mass
from shutil import copyfile

def gaussian2D(xy_tuple, amplitude, xo, yo, wx, wy, theta, offset):
    (x,y) = xy_tuple
    xo = float(xo)
    yo = float(yo)
    sigma_x = wx/2
    sigma_y = wy/2 #function defined in terms of sigmax, sigmay
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def TEMp0(p,xy_tuple,I0,x0,y0,w,offset):
    (x,y) = xy_tuple
    data = offset + I0*np.exp(-2*((x-x0)**2+(y-y0)**2)/w**2)*(assoc_laguerre(2*((x-x0)**2+(y-y0)**2)/w**2,p))**2
    return data.ravel()

def TEM10(xy_tuple,I0,x0,y0,w,offset):
    return TEMp0(1,xy_tuple,I0,x0,y0,w,offset)

image_dir = r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\April\20\Measure 9"
analysis_dir = image_dir+'/gauss_fits'
os.makedirs(analysis_dir,exist_ok=True)
copyfile('image_analyser.py', analysis_dir+'/image_analyser.py')
try:
    df = pd.read_csv(analysis_dir+'/images.csv',index_col=0)
except:
    df = pd.read_csv(image_dir+'/images.csv',index_col=0)
images = {}
for index, row in df.iterrows():
    image = Image()
    image.properties = row.to_dict()
    image.array = np.asarray(PILImage.open(image_dir+'/'+str(index)+'.png'))
    try:
        image.bgnd_array = np.asarray(PILImage.open(image_dir+'/bgnds/'+str(index)+'_bgnd.png'))
    except:
        pass
    try:
        image.holo = np.asarray(PILImage.open(image_dir+'/holos/'+str(index)+'_holo.bmp'))
    except:
        pass
    images[index] = image

function = gaussian2D
print(images)
for index in range(11,18):
    image = images[index]
    print(image.get_properties())
    print(image.array)
    print(image.bgnd_array)
    array = image.get_bgnd_corrected_array()
    max_val = np.max(array)
    cent_ind = center_of_mass(array)
    cent_ind = (int(cent_ind[0]),int(cent_ind[1]))
    cent_ind = np.unravel_index(np.argmax(array, axis=None), array.shape)
    print(max_val)
    print(cent_ind)
    df.loc[index,'max_pixel'] = max_val
    x, y = np.meshgrid(range(array.shape[1]), range(array.shape[0]))
    popt, pcov = curve_fit(function, (x, y), array.ravel(), p0=[max_val,cent_ind[1],cent_ind[0],50,50,0,0])
    # popt, pcov = curve_fit(TEM10, (x, y), array.ravel(), p0=[max_val,cent_ind[1],cent_ind[0],50,0])
    perr = np.sqrt(np.diag(pcov))
    print(popt)
    for arg,val,err in zip(function.__code__.co_varnames[1:],popt,perr):
        df.loc[index,'gauss_'+arg] = val
        df.loc[index,'gauss_'+arg+'_err'] = err
    data_fitted = function((x, y),*popt).reshape(array.shape[0],array.shape[1])
    fig, ax = plt.subplots(1, 1)
    ax.imshow(array, cmap=plt.cm.gray)
    ax.imshow(data_fitted,cmap=plt.cm.viridis,alpha=0.4)
    fig.tight_layout()
    plt.savefig(analysis_dir+'/'+str(index)+'_gauss.png')
    df.to_csv(analysis_dir+'/images.csv')