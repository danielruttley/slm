import os
import time
import sys
import numpy as np
import time

from slm import SLM
from camera import Camera, ImageHandler
import holograms as hg
from image_processing import ArrayGenerator

import logging
logging.basicConfig(filename=r'Z:\Tweezer\Code\Python 3.7\slm\run2.log',level=logging.WARNING,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

slm = SLM(monitor=1)
cam = Camera(exposure=95,gain=40,roi=[300,331,1121,1022])
center = (288,227)
waist = 336
complex_amp_mod = hg.ComplexAmpMod()

exposure = 95
gain = 40
exposure = cam.update_exposure(exposure)
gain = cam.update_gain(gain)

phase_correction = hg.misc.load("Z:/Tweezer/Code/Python 3.7/slm/images/2021/May/25/zernike_measurements/I_opt.png")
blazing = hg.gratings.diag(-12.2)
holo = (blazing+hg.lenses.focal_plane_shift(0.614,center)+phase_correction)%1
holo = hg.apertures.circ(holo,center,210)

gratings = np.linspace(-12.3,-11.7,7)
np.random.shuffle(gratings)

for i,grating in enumerate(gratings):
    try:
        blazing = hg.gratings.diag(grating)
        holo = (blazing+hg.lenses.focal_plane_shift(0.614,center)+phase_correction)%1
        holo = hg.apertures.circ(holo,center,210)
        imager = ImageHandler(measure_params={'grating':grating})
        array_gen = ArrayGenerator(slm,cam,imager,holo,circ_aper_center=center,circ_aper_radius=210,calibration_blazing=hg.gratings.diag(-12)*0.5)
        array_gen.generate_input_intensity(waist,center)
        array_gen.load_traps_from_image(r"Z:\Tweezer\Code\Python 3.7\slm\image_processing\smiley.png",4)
        #array_gen.load_prev_trap_locs(r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\26\Measure 96\trap_df.csv")
        array_gen.get_single_trap_coords()
        array_gen.generate_initial_hologram(iterations=110)
        array_gen.get_trap_depths(reps=1,plot=False)
        array_gen.generate_corrected_hologram(iterations=110)
        array_gen.get_trap_depths(reps=1,plot=False)
        array_gen.save_trap_df()
    except Exception as e:
       logging.error("Exception occured. Relative measure {}, aperture {}".format(i,grating),exc_info=True)
print("program completed")
