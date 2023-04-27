import os
import sys
import time
import json
import numpy as np
import pandas as pd
import PIL.Image as PILImage
from shutil import copyfile
import matplotlib.pyplot as plt

from .windows_setup import configure_path
configure_path()
from .source import TLCameraSDK

class Camera():
    """Object which handles the taking and processing of images from the new
    ThorLabs scientific camera API.
    """
    def __init__(self, exposure=1, blacklevel=0, gain=0, roi=None):
        self.exposure = exposure
        self.blacklevel = blacklevel
        self.gain = gain
        self.set_roi(roi)
    
    def update_exposure(self,exposure=None):
        """Sets and gets exposure time in ms."""
        if exposure != None:
            self.exposure = exposure
        return self.exposure

    def update_blacklevel(self,blacklevel=None):
        """Set blacklevel compensation on or off."""
        if blacklevel != None:
            self.blacklevel = blacklevel
        return self.blacklevel
    
    def update_gain(self,gain=None):
        """Set and gets the gain level of the camera.
        
        Parameters:
            gain: gain of the camera. Between 0 - 100.
        """
        if gain != None:
            self.gain = gain
        return self.gain

    # def aquire(self):
    #     """Aquires a single array from the camera with the current settings."""
    #     array = self.cam.acquire() #acquire an image
    #     if self.roi != None:
    #         array = array[self.roi[1]:self.roi[3],self.roi[0]:self.roi[2]]
    #     if (array == 255).sum() > 0:
    #         print('Warning: image saturated')
    #     print(array)
    #     print(np.max(array))
    #     array[array > 255] = 255
    #     return np.uint8(array)

    # def take_image(self):
    #     """Gets an image from the camera and returns it in an object containing
    #     the current camera settings.

    #     Returns:
    #         Image object containing the array from the camera and current 
    #         camera parameters.
    #     """
    #     array = self.aquire()
    #     return Image(array,self.exposure,self.blacklevel,self.roi,self.gain)

    def set_roi(self,roi):
        """Sets the roi applied to images taken by the camera.

        Parameters:
            roi: None for no roi or [xmin,ymin,xmax,ymax]
        """
        self.roi = roi

    def get_roi(self):
        return self.roi

    def auto_gain_exposure(self):
        exposure = self.update_exposure()
        gain = self.update_gain()
        while True:
            image = self.take_image()
            max_pixel = image.get_max_pixel(correct_bgnd=False)
            print(exposure,gain,max_pixel)
            if max_pixel < 2**self.bit_depth*0.2:
                print(f'Max pixel ({max_pixel}) below min value ({2**self.bit_depth*0.2})')
                if exposure < 0.1:
                    exposure += 0.1
                elif exposure < 85:
                    exposure *= 1.1
                else:
                    gain += 1
            elif max_pixel > 2**self.bit_depth*0.99:
                print(f'Max pixel ({max_pixel}) above max value ({2**self.bit_depth*0.99})')
                if gain > 1:
                    gain -= 1
                elif exposure < 1:
                    exposure -= 0.1
                else:                
                    exposure *= 0.9
            else:
                break
            if exposure < 0.05:
                exposure = 0.07
            exposure = self.update_exposure(exposure)
            gain = self.update_gain(gain)
        
        return exposure, gain

    def take_image(self):
        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            if len(available_cameras) < 1:
                print("no cameras detected")

            with sdk.open_camera(available_cameras[0]) as camera:
                print('connected to camera {}'.format(camera.name))
                
                self.bit_depth = camera.bit_depth
                print(f'bit rate = {self.bit_depth}')

                camera.exposure_time_us = int(self.exposure*1e3) # set exposure to 0.5 ms
                camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
                camera.image_poll_timeout_ms = 10000  # 10 second polling timeout
                old_roi = camera.roi  # store the current roi
                
                print(self.roi)
                if self.roi != None:
                    camera.roi = self.roi  # set roi to be at origin point (100, 100) with a width & height of 500
                
                """
                uncomment the lines below to set the gain of the camera and read it back in decibels
                """
                #if camera.gain_range.max > 0:
                #    db_gain = 6.0
                #    gain_index = camera.convert_decibels_to_gain(db_gain)
                #    camera.gain = gain_index
                #    print(f"Set camera gain to {camera.convert_gain_to_decibels(camera.gain)}")

                camera.arm(2)
                camera.issue_software_trigger()

                frame = camera.get_pending_frame_or_null()
                if frame is not None:
                    print("frame #{} received!".format(frame.frame_count))
                    frame.image_buffer  # .../ perform operations using the data from image_buffer

                    #  NOTE: frame.image_buffer is a temporary memory buffer that may be overwritten during the next call
                    #        to get_pending_frame_or_null. The following line makes a deep copy of the image data:
                    image_buffer_copy = np.copy(frame.image_buffer)
                else:
                    print("timeout reached during polling, program exiting...")
                    image_buffer_copy = None
                    
                camera.disarm()
                camera.roi = old_roi  # reset the roi back to the original roi
                array = (image_buffer_copy).astype('uint16')
                return Image(array,self.exposure,self.blacklevel,self.roi,self.gain)
            
    def get_bit_depth(self):
        """Returns the bit depth of the camera. If the bit_depth attribute 
        doesn't yet exist, a discarded image is taken first during which 
        the bit depth is stored."""
        try:
            return self.bit_depth
        except AttributeError:
            self.take_image()
            return self.bit_depth

class Image():
    """Custom image object containing the array as well as a dictionary 
    containing the camera settings when the image was taken. Custom properties 
    can be added, which will be saved when the image is saved.
    """
    def __init__(self,array=None,exposure=None,blacklevel=None,roi=None,gain=None):
        self.array = array
        if roi == None:
            if not (array is None):
                xmin,ymin,xmax,ymax = [0,0,self.array.shape[1],self.array.shape[0]]
            else:
                xmin,ymin,xmax,ymax = None,None,None,None
        else:
            xmin,ymin,xmax,ymax = roi
        self.properties = {'exposure':exposure,
                           'blacklevel':blacklevel,
                           'roi_xmin':xmin,
                           'roi_ymin':ymin,
                           'roi_xmax':xmax,
                           'roi_ymax':ymax,
                           'gain':gain
                          }
        self.bgnd_array = None
        self.hologram = None
    
    def add_property(self,name,value):
        """Adds a property to the image properties dictonary."""
        self.properties[name] = value
    
    def get_properties(self):
        return self.properties
    
    def get_array(self):
        return self.array

    def add_background(self,bgnd_image):
        """Extracts an array from a background image"""
        if self.properties != bgnd_image.get_properties():
            print('Warning: background properties do not match image properties')
        self.bgnd_array = bgnd_image.get_array().copy()
    
    def get_background(self):
        return self.bgnd_array

    def add_hologram(self, hologram):
        self.hologram = hologram
    
    def get_hologram(self):
        return self.hologram

    def get_array_float(self):
        return np.float32(self.array)

    def get_bgnd_corrected_array(self):
        return np.float32(self.array) - np.float32(self.bgnd_array)
    
    def get_pixel_count(self,correct_bgnd=True):
        if correct_bgnd:
            array = np.float32(self.array) - np.float32(self.bgnd_array)
        else:
            array = np.float32(self.array)
        sum = np.int(np.sum(array))
        return sum
    
    def get_max_pixel(self,correct_bgnd=True):
        if correct_bgnd:
            array = np.float32(self.array) - np.float32(self.bgnd_array)
        else:
            array = np.float32(self.array)
        return np.max(array)

    def apply_calibration(self,calibration):
        self.array = np.float32(self.array)/calibration
        if self.bgnd_array is not None:
            self.bgnd_array = np.float32(self.bgnd_array)/calibration

class ImageHandler():
    """Deals with the saving and loading of images from the ThorLabs camera"""
    def __init__(self,measure=None,measure_params=None,measure_folder=None,
                 bit_depth=None):
        """Creates the directory to save images in.

        Parameters
        ----------
        measure: int or None or str
            the measure number to assign. -1 to append to the last 
            measure, and None to create a new measure. If a string is
            passed, this will be used as the subfolder name (without 
            Measure prefixed)
        measure_params : dict or None
            other parameters to save about the measure in a json called 
            params.json in the measure folder
        measure_folder : str or None
            The folder to save the results in. Normally should point to the 
            Experimental Results folder. 
        bit_depth : The bit depth of the camera used to take the image. The 
            images will be normalised by this bit depth to be mapped onto 
            8-bits for image saving.

        
        Returns
        -------
        None
        """
        self.created_dirs = False
        self.measure = measure
        self.measure_params = measure_params
        self.measure_folder = measure_folder
        self.bit_depth = bit_depth

    def create_dirs(self,measure=None):
        if measure is None:
            measure = self.measure
        if self.measure_folder is None:
            date_dir = './images/'+time.strftime('%Y/%B/%d', time.localtime())+'/SLM measures'
        else:
            date_dir = self.measure_folder+'//'+time.strftime('%Y/%B/%d', time.localtime())+'/SLM measures'
        os.makedirs(date_dir,exist_ok=True)

        if type(measure) == str:
            self.image_dir = date_dir+'/'+measure
        else:
            subfolders = [f.name for f in os.scandir(date_dir) if f.is_dir()]
            prev_measures = [f for f in subfolders if 'Measure' in f]
            prev_measures = [int(s.split('Measure ',1)[1]) for s in prev_measures]
            if prev_measures:
                if measure == -1:
                    measure = max(prev_measures)
                elif measure == None:
                    measure = max(prev_measures)+1
            else:
                measure = 0
            self.image_dir = date_dir+'/Measure {}'.format(measure)
        self.measure = measure
        print(self.image_dir)
        os.makedirs(self.image_dir,exist_ok=True)
        copyfile(sys.argv[0], self.image_dir+'\\'+sys.argv[0].split('\\')[-1])
        os.makedirs(self.image_dir+'/bgnds',exist_ok=True)
        os.makedirs(self.image_dir+'/holos',exist_ok=True)
        self.created_dirs = True
        try:
            self.df = pd.read_csv(self.image_dir+'/images.csv',index_col=0)
        except:
            self.df = pd.DataFrame()
        if self.measure_params is not None:
            with open(self.image_dir+'/params.json', 'w') as f:
                json.dump(self.measure_params, f, sort_keys=True, indent=4)

    def show_image(self,image):
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.show()
    
    def save(self,image):
        """Save custom image object as a .png file and append the image 
        properties to the csv.
        """
        if not self.created_dirs:
            self.create_dirs(self.measure)
        array = image.get_array()
        if self.bit_depth is not None:
            array = (array*(2**8/2**self.bit_depth)).astype(np.uint8)
            array = (array*((2**8-1)/np.max(array))).astype(np.uint8)
        properties = image.get_properties()
        print(properties)
        self.df = self.df.append(properties,ignore_index=True)
        filepath = self.image_dir+'/'+str(self.df.index[-1])+'.png'
        bgnd_filepath = self.image_dir+'/bgnds/'+str(self.df.index[-1])+'_bgnd.png'
        holo_filepath = self.image_dir+'/holos/'+str(self.df.index[-1])+'_holo.bmp'

        PILImage.fromarray(array,"L").save(filepath)
        self.df.to_csv(self.image_dir+'/images.csv')
        background = image.get_background()
        if not (background is None):
            if self.bit_depth is not None:
                # background = (background*(2**8/2**self.bit_depth)).astype(np.uint8)
                background = (background*((2**8-1)/np.max(array))).astype(np.uint8)
            PILImage.fromarray(background,"L").save(bgnd_filepath)
        holo = image.get_hologram()
        if not (holo is None):
            holo = np.uint16(holo*65535)
            red = np.uint8(holo % 256)
            green = np.uint8((holo - red)/256)
            blue = np.uint8(np.zeros(holo.shape))
            rgb = np.dstack((red,green,blue))
            PILImage.fromarray(rgb,"RGB").save(holo_filepath)
    
    def get_dir(self):
        return self.image_dir

    def get_last_index(self):
        return self.df.index[-1]

if __name__ == '__main__':
    camera = Camera()