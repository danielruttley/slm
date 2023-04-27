import os
import time
import sys
import numpy as np

from slm import SLM
from camera import Camera, ImageHandler
import holograms as hg
from normaliser import load_holos_from_params

slm = SLM(monitor=1)

print('loading holo')
blazing = load_holos_from_params(r"Z:\Tweezer\Experimental Results\2023\March\27\slm_position_base.txt")
print(blazing)
slm.apply_hologram(blazing)
print('loaded holo')

cam = Camera(exposure=30,roi=[1166,160,1200,250])
imager = ImageHandler(measure_folder=r"Z:\Tweezer\Experimental Results",
                      bit_depth=cam.get_bit_depth())

centers = np.linspace(25,490,101)
np.random.shuffle(centers)
for center in centers:
    holo,center = hg.apertures.vert(blazing,center,20,return_center=True)
    slm.apply_hologram(holo)
    time.sleep(0.5)
    image = cam.take_image()
    slm.apply_hologram(hg.blank())
    time.sleep(0.5)
    bgnd = cam.take_image()
    image.add_background(bgnd)
    image.add_hologram(holo)
    pixel_sum = image.get_pixel_count()
    print(pixel_sum)
    image.add_property('vert_aper_center',center)
    image.add_property('pixel_sum',pixel_sum)
    imager.save(image)

print("program completed")
