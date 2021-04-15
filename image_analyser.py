import pandas as pd
import numpy as np
import PIL.Image as PILImage
from scipy.optimize import curve_fit
from camera.camera_class import Image
import matplotlib.pyplot as plt
import os

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

csv = r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\April\14\Measure 7\images.csv"
df = pd.read_csv(csv,index_col=0)
image_dir = csv.split(r'images.csv')[0]
images = {}
for index, row in df.iterrows():
    image = Image()
    image.properties = row.to_dict()
    image.array = np.asarray(PILImage.open(image_dir+str(index)+'.png'))
    try:
        image.bgnd_array = np.asarray(PILImage.open(image_dir+'/bgnds/'+str(index)+'_bgnd.png'))
    except:
        pass
    try:
        image.holo = np.asarray(PILImage.open(image_dir+'/holos/'+str(index)+'_holo.bmp'))
    except:
        pass
    images[index] = image

analysis_dir = image_dir+'/gauss_fits/'
os.makedirs(analysis_dir,exist_ok=True)

for index in images:
    image = images[index]
    print(image.get_properties())
    print(image.array)
    print(image.bgnd_array)
    array = image.get_bgnd_corrected_array()
    print(np.max(array))
    x, y = np.meshgrid(range(array.shape[1]), range(array.shape[0]))
    popt, pcov = curve_fit(gaussian2D, (x, y), array.ravel(), p0=[200,713,579,20,20,0,0])
    perr = np.sqrt(np.diag(pcov))
    df.loc[index,'gauss_amp'] = popt[0]
    df.loc[index,'gauss_x0'] = popt[1]
    df.loc[index,'gauss_y0'] = popt[2]
    df.loc[index,'gauss_wx'] = popt[3]
    df.loc[index,'gauss_wy'] = popt[4]
    df.loc[index,'gauss_theta'] = popt[5]
    df.loc[index,'gauss_offset'] = popt[6]
    df.loc[index,'gauss_amp_err'] = perr[0]
    df.loc[index,'gauss_x0_err'] = perr[1]
    df.loc[index,'gauss_y0_err'] = perr[2]
    df.loc[index,'gauss_wx_err'] = perr[3]
    df.loc[index,'gauss_wy_err'] = perr[4]
    df.loc[index,'gauss_theta_err'] = perr[5]
    df.loc[index,'gauss_offset_err'] = perr[6]
    data_fitted = gaussian2D((x, y),*popt).reshape(array.shape[0],array.shape[1])
    fig, ax = plt.subplots(1, 1)
    ax.imshow(array, cmap=plt.cm.jet,origin='upper',extent=(x.min(), x.max(), y.min(), y.max()))
    ax.contour(data_fitted,8,origin='upper',colors='w')
    plt.savefig(analysis_dir+str(index)+'_gauss.png')
df.to_csv(analysis_dir+'images.csv')
    #TODO add in 2d gaussian fitting from https://stackoverflow.com/questions/52148141/2d-gaussian-fit-using-lmfit

