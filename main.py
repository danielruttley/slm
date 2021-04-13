import os
import time
import sys
import numpy as np

from slm import SLM
from camera import Camera, ImageHandler
from boss_class import Boss
import holograms as hg

def bgnd_image(self,hologram,slm,camera,exposure=1,blacklevel=0):
    """Takes a background corrected image by first taking a background image 
    by clearing the SLM screen before applying the hologram and subtracting 
    the background from the resultant image.
    """
    # NOTE slm.apply_hologram(blank)
    slm.apply_hologram(hologram)

slm = SLM(monitor=1)
cam = Camera(exposure=100,roi=[614,597,696,677])
imager = ImageHandler()
boss = Boss(slm,None)
# test = boss.test()
# print(test)

blazing = hg.blazediag(-12)
#lg_holo = (lgmode(219,274,0,0,50)+lgmode(219,274,2,0,50)+lgmode(219,274,4,0,50))%1

for center in np.linspace(200,300,5):
    # lg_holo = lgmode(219,274,1,0,width)
    # array = np.mod(blazing+lg_holo,1)
    slm.apply_hologram(blazing)
    time.sleep(1)
    image = cam.take_image()
    imager.save(image)
    time.sleep(1)
    # take_image(width)
#input("Press Enter to continue...")
print("program completed")
