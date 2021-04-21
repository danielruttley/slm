import os
import time
import sys
import numpy as np

from slm import SLM
from camera import Camera, ImageHandler
import holograms as hg

slm = SLM(monitor=1)
cam = Camera(exposure=80)#,roi=[614,597,696,677])
imager = ImageHandler(9)

complex_amp_mod = hg.ComplexAmpMod()

center = (214,265)

blazing = hg.blazediag(-12)
holo = complex_amp_mod.superposition_holo((0,2,4),center,41,blazing,82)
lens = hg.fresnel_lens(0.501,center,1064e-9)
holo = (holo+lens)%1
holo = hg.tools.circ_aper(holo,center)
slm.apply_hologram(holo)
time.sleep(0.5)
image = cam.take_image()
slm.apply_hologram(hg.blank())
time.sleep(0.5)
bgnd = cam.take_image()
image.add_background(bgnd)
image.add_hologram(holo)
max_pixel = image.get_max_pixel()
print(max_pixel)
image.add_property('camera_position',sys.argv[1])
image.add_property('max_pixel',max_pixel)
imager.save(image)

print("program completed")
