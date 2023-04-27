import logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO)

import time
import numpy as np
import PIL.Image as PILImage

from windows_setup import configure_path
configure_path()

from source import TLCameraSDK

def take_image(exposure_ms,roi=None):
    with TLCameraSDK() as sdk:
        available_cameras = sdk.discover_available_cameras()
        if len(available_cameras) < 1:
            print("no cameras detected")

        with sdk.open_camera(available_cameras[0]) as camera:
            print('connected to camera {}'.format(camera.name))
            print(f'bit rate = {camera.bit_depth}')

            camera.exposure_time_us = int(exposure_ms*1e3) # set exposure to 0.5 ms
            camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
            camera.image_poll_timeout_ms = 10000  # 10 second polling timeout
            old_roi = camera.roi  # store the current roi
            
            print(roi)
            if roi != None:
                camera.roi = roi  # set roi to be at origin point (100, 100) with a width & height of 500

            """
            uncomment the lines below to set the gain of the camera and read it back in decibels
            """
            #if camera.gain_range.max > 0:
            #    db_gain = 6.0
            #    gain_index = camera.convert_decibels_to_gain(db_gain)
            #    camera.gain = gain_index
            #    print(f"Set camera gain to {camera.convert_gain_to_decibels(camera.gain)}")

            camera.arm(2)

            camera.issue_software_trigger()

            frame = camera.get_pending_frame_or_null()
            if frame is not None:
                print("frame #{} received!".format(frame.frame_count))
                frame.image_buffer  # .../ perform operations using the data from image_buffer

                #  NOTE: frame.image_buffer is a temporary memory buffer that may be overwritten during the next call
                #        to get_pending_frame_or_null. The following line makes a deep copy of the image data:
                image_buffer_copy = np.copy(frame.image_buffer)
            else:
                print("timeout reached during polling, program exiting...")
                image_buffer_copy = None
                
            camera.disarm()
            camera.roi = old_roi  # reset the roi back to the original roi
            return (image_buffer_copy*int(2**16/2**camera.bit_depth)).astype('uint16')

save_directory = r".\albert_overnight"
while True:
    filepath = save_directory+"\\"+str(time.time())+'.png'
    array = take_image(13)
    PILImage.fromarray(array).save(filepath)
    print('saved image as',filepath)
    time.sleep(50)
    

