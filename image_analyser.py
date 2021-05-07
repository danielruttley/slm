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
from math import log10,floor,ceil

image_dir = r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\April\29\Measure 6"
copyfile('image_analyser.py', image_dir+'/image_analyser.py')
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

max_vals = []
verts = []
horis = []

for index in images:
    image = images[index]
    properties = image.get_properties()
    array = image.get_bgnd_corrected_array()
    max_val = np.max(array)
    max_vals.append(max_val)
    verts.append(properties['vert_grating'])
    horis.append(properties['vert_grating'])
horis = np.asarray(horis).reshape((11,11)).transpose()
verts = np.asarray(verts).reshape((11,11))
max_vals = np.asarray(max_vals).reshape((11,11))
print(horis)
print(verts)
print(max_vals)
plt.pcolor(verts,horis,max_vals,shading='auto')
plt.xlabel('vertical grating [pixels]')
plt.ylabel('horizontal grating [pixels]')
plt.colorbar(label='maximum intensity')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.gcf().tight_layout()
plt.savefig(image_dir+'/max_vals.png',dpi=300)
plt.close()

print(np.min(max_vals)/np.max(max_vals))