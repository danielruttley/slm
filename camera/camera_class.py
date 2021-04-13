import os
import time
import numpy as np
import pandas as pd
import PIL.Image as PILImage

from .uc480 import uc480

class Camera():
    """Object which handles the taking and processing of images from the 
    ThorLabs DCC1545M-GL camera.
    """
    def __init__(self, exposure=1, blacklevel=0, roi=None):
        self.exposure = exposure
        self.blacklevel = blacklevel
        self.set_roi(roi)
        self.cam = uc480()
        self.cam.connect()

        self.update_exposure(self.exposure)
        self.update_blacklevel(self.blacklevel)
    
    def __del__(self):
        self.cam.disconnect()
    
    def update_exposure(self,exposure=None):
        """Sets and gets exposure time in ms."""
        if exposure != None:
            self.cam.set_exposure(exposure)
        self.exposure = self.cam.get_exposure()
        return self.exposure

    def update_blacklevel(self,blacklevel):
        """Set blacklevel compensation on or off."""
        self.blacklevel = blacklevel
        self.cam.set_blacklevel(self.blacklevel)
        return self.blacklevel

    def aquire(self):
        """Aquires a single array from the camera with the current settings."""
        array = self.cam.acquire() #acquire an image
        if self.roi != None:
            array = array[self.roi[1]:self.roi[3],self.roi[0]:self.roi[2]]
        if (array == 255).sum() > 0:
            print('Warning: image saturated')
        array[array > 255] = 255
        return np.uint8(array)

    def take_image(self):
        """Gets an image from the camera and returns it in an object containing
        the current camera settings.
        """
        array = self.aquire()
        return Image(array,self.exposure,self.blacklevel,self.roi)

    def set_roi(self,roi):
        """Sets the roi applied to images taken by the camera.

        Parameters:
            roi: None for no roi or [xmin,ymin,xmax,ymax]
        """
        self.roi = roi

class Image():
    """Custom image object containing the array as well as a dictionary 
    containing the camera settings when the image was taken. Custom properties 
    can be added, which will be saved when the image is saved.
    """
    def __init__(self,array,exposure,blacklevel,roi):
        self.array = array
        self.properties = {'exposure':exposure,
                           'blacklevel':blacklevel,
                           'roi':roi
                          }
    
    def add_property(name,value):
        self.properties[name] = value
    
    def get_properties(self):
        return self.properties
    
    def get_array(self):
        return self.array

class ImageHandler():
    """Deals with the saving and loading of images from the ThorLabs camera"""
    def __init__(self,image_dir=None):
        if image_dir == None:
            time_dir = time.strftime("%Y/%m/%d/%H.%M.%S", time.localtime())
            print(time_dir)
            image_dir = './images/'+time_dir
        self.image_dir = image_dir
        os.makedirs(self.image_dir,exist_ok=True)
        self.df = pd.DataFrame()

    def show_image(self,image):
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.show()
    
    def save(self,image):
        """Save custom image object as a .png file and append the image 
        properties to the csv.
        """
        array = image.get_array()
        properties = image.get_properties()
        print(properties)
        self.df = self.df.append(properties,ignore_index=True)
        filepath = self.image_dir+'/'+str(self.df.index[-1])+'.png'
        PILImage.fromarray(array,"L").save(filepath)
        self.df.to_csv(self.image_dir+'/images.csv')