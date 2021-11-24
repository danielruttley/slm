import os
import time
import sys
import numpy as np

from slm import SLM
from camera import Camera, ImageHandler
import holograms as hg
from arrays import ArrayGenerator

slm = SLM(monitor=1)
cam = Camera(exposure=95,gain=40,roi=[604,671,742,727])
center = (233,251)
waist = 215
imager = ImageHandler()

exposure = 95
gain = 40
exposure = cam.update_exposure(exposure)
gain = cam.update_gain(gain)

holo = (hg.misc.load('zernike_phase_correction.png')+
        hg.gratings.hori(period=7)+
        hg.gratings.hori_gradient(gradient=-7.3)+
        hg.lenses.focal_plane_shift(shift=-3,x0=233,y0=251)+
        hg.gratings.vert_gradient(gradient=5.29))%1
holo = hg.apertures.circ(holo,x0=233,y0=251,radius=233)

traps = [(256,256),(260,256),(252,256)]

array_gen = ArrayGenerator(slm,cam,imager,holo,circ_aper_center=center,circ_aper_radius=233)
array_gen.generate_input_intensity(waist,center)
array_gen.traps = traps
array_gen.get_single_trap_coords()
#array_gen.load_trap_df(r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\November\23\Measure 3\trap_df.csv")
array_gen.generate_initial_hologram(iterations=50)
array_gen.get_trap_depths(reps=1,plot=True)
array_gen.save_trap_df()
for i in range(5):
    array_gen.generate_corrected_hologram(iterations=50)
    array_gen.get_trap_depths(reps=1,plot=True)
array_gen.save_trap_df()
slm.apply_hologram(array_gen.array_holo)

print("program completed")
