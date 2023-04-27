import os
import time
import sys
import numpy as np

from slm import SLM
from camera import Camera, ImageHandler
import holograms as hg
from arrays import ArrayGenerator
from gui.holo_container import get_holo_container
from gui.main import get_holo_type_function

def load_holos_from_params(filename):
    """
    Set holograms from a SLMparams file.
    """
    with open(filename, 'r') as f:
        msg = f.read()
    msg = eval(msg)
    slm_settings = msg[0]
    holo_list = msg[1]

    global_holo_params = {}
    global_holo_params['beam_center'] = (slm_settings['beam x0'],slm_settings['beam y0'])
    global_holo_params['beam_waist'] = slm_settings['beam waist (pixels)']
    global_holo_params['pixel_size'] = slm_settings['pixel size (m)']
    global_holo_params['shape'] = (slm_settings['x size'],slm_settings['y size'])
    global_holo_params['wavelength'] = slm_settings['wavelength']

    holos = []
    for i,(name,args) in enumerate(holo_list):
        try:
            holo_params = {'name':name}
            holo_params['type'],holo_params['function'] = get_holo_type_function(name)
            holo_params = {**holo_params,**args}
            holo = get_holo_container(holo_params,global_holo_params)
            holos.append(holo)
        except Exception as e:
            print('Error when creating Hologram {}. The hologram has been skipped.\n'.format(i),e)

    total_holo = hg.blank(phase=0,shape=global_holo_params['shape'])
    for holo in holos:
        if holo.get_type() == 'aperture':
            total_holo = holo.apply_aperture(total_holo)
        elif holo.get_type() == 'cam':
            total_holo = holo.get_cam_holo(total_holo)
        else:
            total_holo += holo.get_holo()
    return total_holo%1

if __name__ == '__main__':
    slm = SLM(monitor=1)
    cam = Camera(exposure=0.1,gain=0,roi=[1179,197,1297,228])

    measures = []
    radius_scales = np.linspace(0.5,1,26)
    np.random.shuffle(radius_scales)
    for radius_scale in radius_scales:
        center = (252,272)
        waist = 215
        # radius_scale = 0.8 # semi minor to major ratio for ellipse

        imager = ImageHandler(measure_folder=r"Z:\Tweezer\Experimental Results",
                            bit_depth=cam.get_bit_depth(),measure_params={'radius_scale':radius_scale})
        
        correction_factor = 0.2

        exposure = 1
        gain = 0
        exposure = cam.update_exposure(exposure)
        gain = cam.update_gain(gain)

        # holo = (hg.gratings.hori(period=7)+
        #         hg.gratings.hori_gradient(gradient=-10.2)+
        #         hg.lenses.focal_plane_shift(shift=-15.68,x0=265,y0=251)+
        #         hg.gratings.vert_gradient(gradient=9.6)+
        #         hg.zernike(4,0,-0.129,265,251,247)+
        #         hg.zernike(2,2,-0.2,265,251,247)+
        #         hg.zernike(2,-2,-0.076,265,251,247)+
        #         hg.zernike(4,4,0.08,265,251,247))%1
        # holo = hg.apertures.circ(holo,x0=265,y0=251,radius=247)

        holo = load_holos_from_params(r"Z:\Tweezer\Experimental Results\2023\April\26\normaliser_base_no_ZK.txt")

        traps = [(368, 256, 1.166851), (352, 256, 1.113433), (336, 256, 1.020398), (320, 256, 0.940435), (304, 256, 0.810454), (288, 256, 0.997138), (272, 256, 1.011309), (256, 256, 1.0375530000000002), (240, 256, 1.0392629999999998)]
        padding = 4

        array_gen = ArrayGenerator(slm,cam,imager,holo,
                                circ_aper_center=center,circ_aper_radius=247,
                                input_waist=waist,input_center=center,
                                correction_factor=correction_factor,
                                min_distance_between_traps=5,
                                remove_low_zernike=True,padding=padding)
        array_gen.traps = traps
        # array_gen.get_single_trap_coords()
        array_gen.load_trap_df(r"Z:\Tweezer\Experimental Results\2023\April\26\SLM measures\Measure 10\trap_df.csv")
        array_gen.traps = traps

        exposure = cam.update_exposure(exposure)
        gain = cam.update_gain(gain)

        array_gen.generate_initial_hologram(iterations=50)
        array_gen.array_holo = hg.apertures.ellipse(array_gen.array_holo,x0=center[0],y0=center[1],radius=240,radius_scale=radius_scale)
        array_gen.get_trap_depths(reps=1,plot=True)
        array_gen.save_trap_df()
        # for i in range(50):
        #     array_gen.generate_corrected_hologram(iterations=50)
        #     array_gen.array_holo = hg.apertures.ellipse(array_gen.array_holo,x0=center[0],y0=center[1],radius=240,radius_scale=radius_scale)
        #     array_gen.get_trap_depths(reps=1,plot=True)
        #     print(array_gen.traps)
        array_gen.save_trap_df()
        slm.apply_hologram(array_gen.array_holo)

        print("program completed")
        print("final trap coordinates:")
        print(array_gen.traps)

        measures.append(imager.measure)
        for measure, radius_scale in zip(measures,radius_scales):
            print(f'Measure {measure} = radius_scale {radius_scale}')