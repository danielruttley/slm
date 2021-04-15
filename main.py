import os
import time
import sys
import numpy as np

from slm import SLM
from camera import Camera, ImageHandler
import holograms as hg

slm = SLM(monitor=1)
cam = Camera(exposure=2)#,roi=[614,597,696,677])
imager = ImageHandler()

blazing = hg.blazediag(-12)
widths = np.linspace(0.25,0.75,11)
np.random.shuffle(widths)
lg_holo = hg.lgmode(214,265,0,0,81)
for focal_plane in widths:
    lens = hg.fresnel_lens(focal_plane,214,265,1064e-9)
    holo = (blazing+lg_holo+lens)%1
    holo = hg.tools.circ_aper(holo,(214,265),214)
    slm.apply_hologram(holo)
    time.sleep(0.5)
    image = cam.take_image()
    slm.apply_hologram(hg.blank())
    time.sleep(0.5)
    bgnd = cam.take_image()
    image.add_background(bgnd)
    image.add_hologram(holo)
    image.add_property('focal_plane',focal_plane)
    image.add_property('lgholo_w',81)
    image.add_property('aper_x0',214)
    image.add_property('aper_y0',265)
    image.add_property('aper_r',214)
    imager.save(image)

# width = 20
# centers = np.linspace(220,320,51)
# np.random.shuffle(centers)
# for c in centers:
#     holo, center = hg.tools.hori_aper(blazing, c, width)
#     # lg_holo = lgmode(219,274,1,0,width)
#     # array = np.mod(blazing+lg_holo,1)
#     slm.apply_hologram(holo)
#     time.sleep(1)
#     image = cam.take_image()
#     slm.apply_hologram(hg.blank())
#     time.sleep(1)
#     bgnd = cam.take_image()
#     image.add_background(bgnd)
#     image.add_hologram(holo)
#     image.add_property('center',center)
#     image.add_property('width',width)
#     image.add_property('pixel count',image.get_pixel_count())
#     imager.save(image)

print("program completed")
