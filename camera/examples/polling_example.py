import matplotlib.pyplot as plt

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None

import numpy as np
from source import TLCameraSDK

NUM_FRAMES = 1  # adjust to the desired number of frames

with TLCameraSDK() as sdk:
    available_cameras = sdk.discover_available_cameras()
    if len(available_cameras) < 1:
        print("no cameras detected")

    with sdk.open_camera(available_cameras[0]) as camera:
        print('connected to camera {}'.format(camera.name))

        camera.exposure_time_us = 500  # set exposure to 0.5 ms
        camera.frames_per_trigger_zero_for_unlimited = 0  # start camera in continuous mode
        camera.image_poll_timeout_ms = 1000  # 1 second polling timeout
        old_roi = camera.roi  # store the current roi
        """
        uncomment the line below to set a region of interest (ROI) on the camera
        """
        camera.roi = (1177, 227, 1239, 289)  # set roi to be at origin point (100, 100) with a width & height of 500

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

        for i in range(NUM_FRAMES):
            frame = camera.get_pending_frame_or_null()
            if frame is not None:
                print("frame #{} received!".format(frame.frame_count))

                frame.image_buffer  # .../ perform operations using the data from image_buffer

                #  NOTE: frame.image_buffer is a temporary memory buffer that may be overwritten during the next call
                #        to get_pending_frame_or_null. The following line makes a deep copy of the image data:
                image_buffer_copy = np.copy(frame.image_buffer)
            else:
                print("timeout reached during polling, program exiting...")
                break
        camera.disarm()
        camera.roi = old_roi  # reset the roi back to the original roi

#  Because we are using the 'with' statement context-manager, disposal has been taken care of.

print("program completed")

plt.imshow(image_buffer_copy, vmin=0, vmax=4095, cmap='gray')
