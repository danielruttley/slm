import holograms as hg
from slm import SLM
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

slm = SLM(monitor=0)
blazing = hg.gratings.grating(7,0)
lens = hg.lenses.focal_plane_shift(-3.9)
traps = [(256,256),(260,260),(256,260),(260,256)]
array = hg.arrays.aags(traps,input_waist=210)
print(array)
holo = hg.apertures.circ((blazing+lens+array)%1)

#blazing = hg.gratings.grating(2,0)
#holo,center = hg.apertures.vert(blazing,256,50,return_center=True)

#print(np.max(holo))
slm.apply_hologram(holo)
input('press ENTER to continue')