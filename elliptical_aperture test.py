from holograms.apertures import circ
import matplotlib.pyplot as plt
import numpy as np

ones = np.ones((512,512))
holo = circ(ones,252,272,252)

plt.imshow(holo)