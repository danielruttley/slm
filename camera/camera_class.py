# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:40:29 2019

@author: danielruttley

Class to handle the taking and processing of images from the Thorcam camera.
"""

from .uc480 import uc480

class Camera():
    def __init__(self, exposure=1, blacklevel=0):
        self.exposure = exposure
        self.blacklevel = blacklevel
        self.cam = uc480()
        self.cam.connect()

        self.update_exposure(self.exposure)
        self.update_blacklevel(self.blacklevel)
    
    def __del__(self):
        self.cam.disconnect()
    
    def update_exposure(self,exposure=None):
        """Sets and gets exposure time in ms."""
        if exposure != None:
            self.cam.set_exposure(self.exposure)
        self.exposure = self.cam.get_exposure()
        return self.exposure

    def update_blacklevel(self,blacklevel=None):
        """Set blacklevel compensation on or off."""
        if blacklevel != None:
            self.cam.set_blacklevel(self.blacklevel)
        self.blacklevel = self.cam.get_blacklevel()
        return self.blacklevel

    def single(self):
        img = self.cam.acquire() #acquire an image
        if (img == 255).sum() > 0:
            print('Warning: image saturated')
        return img