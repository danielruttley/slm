import holograms as hg
from slm import SLM
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

complex_amp_mod = hg.ComplexAmpMod()

center = (288,227)
waist = 336
phase_correction = hg.misc.load(r"Z:\Tweezer\Code\Python 3.7\slm\phase_correction.png")
slm = SLM(monitor=1)

blazing = hg.gratings.diag(float(sys.argv[1]))
holo = (blazing+hg.lenses.focal_plane_shift(0.614,center))%1
holo = hg.apertures.circ(holo,center)
slm.apply_hologram(blazing)
input('press ENTER to continue')