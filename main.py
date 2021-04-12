"""
Polling Example

This example shows how to open a camera, adjust some settings, and poll for images. It also shows how 'with' statements
can be used to automatically clean up camera and SDK resources.

"""

# try:
#     # if on Windows, use the provided setup script to add the DLLs folder to the PATH
#     from windows_setup import configure_path
#     configure_path()
# except ImportError:
#     configure_path = None

import numpy as np
import os
import time
import sys

from slm import SLM
from PIL import Image

from kinoforms.kinoforms import blazediag,lgmode
from kinoforms.arrtorgb import twodimrgb,padarray

from camera import Camera

slm = SLM(monitor=0)
# cam = Camera()

blazing = blazediag(-12)
#lg_holo = (lgmode(219,274,0,0,50)+lgmode(219,274,2,0,50)+lgmode(219,274,4,0,50))%1

for width in np.linspace(10,300,2):
    # lg_holo = lgmode(219,274,1,0,width)
    # array = np.mod(blazing+lg_holo,1)
    slm.apply_hologram(blazing)
    time.sleep(1)
    # take_image(width)
#input("Press Enter to continue...")
print("program completed")
