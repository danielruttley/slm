"""
Defines the SLM wrapper class that handles updates to the SLM.
"""

import numpy as np

from .slmpy import SLMdisplay

class SLM():

    def __init__(self,monitor=1,lut=r"./ScaledLutModel.txt",image_lock=True):
        """
        Parameters:
            monitor: int
                the monitor to display the holograms on
            lut: str
                the filename of the SLM lookup table. Each row of this file 
                should have the corresponding phase and voltage values
            image_lock: bool
                should the program lock until the SLM image is updated
        """
        self.slm = SLMdisplay(monitor=monitor,isImageLock=image_lock)
        self.x_size, self.y_size = self.slm.getSize()
        if lut == None:
            self.lut = None
        else:
            self.lut = {}
            with open(lut) as f:
                for line in f:
                    (key, val) = line.split()
                    self.lut[int(key)] = int(val)

    # def __del__(self):
    #     self.slm.close()
    
    def apply_hologram(self,hologram):
        """Applies a hologram to the SLM display after applying the lookup 
        table and padding the image.

        Parameter:
        hologram: a 2D array of values from 0-1 to be applied in the top-left
                  of the SLM screen, representing phase modulation from 0-2pi
        """
        hologram = np.uint16(hologram*65535)
        if self.lut != None:
            hologram = self.apply_lut(hologram)
        image = self.get_slm_image(hologram)
        image = self.pad_image(image)
        self.slm.updateArray(image)
        

    def apply_lut(self,hologram):
        """Uses the lookup table to convert the hologram from the 16bit phase 
        values to the corresponding 16bit voltage that should be applied to 
        each pixel on the SLM display.
        """
        return np.vectorize(self.lut.get)(hologram)

    def get_slm_image(self,hologram):
        """
        This function takes the 16bit SLM voltage array and returns the 16-bit 
        RGB image which should be applied to the SLM (with zero bits in the 
        blue channel, which is ignored by the SLM). Originally written by 
        Mitch Walker.

        Parameter:
        hologram: a 2D array of 16bit voltage values to be applied to the SLM 
                  screen
        """          
        red = np.uint8(hologram % 256)
        green = np.uint8((hologram - red)/256)
        blue = np.uint8(np.zeros(hologram.shape))
        
        return np.dstack((red,green,blue))

    def pad_image(self,image):
        """Pads the image with zeros to fill the remainder of the SLM screen
        that is not specified by the hologram. NOTE: This is not the same as 
        zero phase modulation due to the lookup table. This should be used for
        SLM pixels which are not physically on the display.
        """
        x,y,z = image.shape
        return np.pad(image,((0,self.y_size-y),(0,self.x_size-x),(0,0)))