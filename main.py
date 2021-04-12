import os
import time
import sys
import numpy as np

from slm import SLM
from camera import Camera
import holograms as hg

slm = SLM(monitor=0)
# cam = Camera()

blazing = hg.blazediag(-12)
#lg_holo = (lgmode(219,274,0,0,50)+lgmode(219,274,2,0,50)+lgmode(219,274,4,0,50))%1

for center in np.linspace(200,300,10):
    # lg_holo = lgmode(219,274,1,0,width)
    # array = np.mod(blazing+lg_holo,1)
    slm.apply_hologram(hg.tools.vert_aper(blazing,center,50))
    time.sleep(1)
    # take_image(width)
#input("Press Enter to continue...")
print("program completed")
