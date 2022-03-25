import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import holograms as hg

"""
NOTE THAT TO GET THIS TO RUN PROPERLY I HAD TO CHANGE THE LENS HOLOGRAM CODE
SO THAT IT DIDN'T MODULO THE HOLOGRAM BEFORE RETURNING IT. I'VE NOW REVERTED
THIS CHANGE.
"""

def linear(x,x0,m):
    return (x-x0)*m

amps = np.linspace(-2,2,11)
diff_sums = []

l = hg.lenses.focal_plane_shift(4,265,251)
l = hg.apertures.circ(l,265,251,247)
for amp in amps:
    z = hg.zernike(2,0,amp,265,251,247)
    z = z - z[256,256]
    z = hg.apertures.circ(z,265,251,247)
    diff = l-z
    diff_sums.append(diff.sum())
    
popt,pcov = curve_fit(linear,amps,diff_sums)
    
z = hg.zernike(2,0,popt[0],265,251,247)
z = z - z[256,256]
z = hg.apertures.circ(z,265,251,247)

plt.imshow(l%1,vmin=0,vmax=1)
plt.show()
plt.imshow(z%1,vmin=0,vmax=1)  
plt.show()  
    
plt.scatter(amps,diff_sums)
plt.plot(amps,linear(amps,*popt),c='C1')
plt.xlabel('Zernike amplitude')
plt.ylabel('sum(lens-Zernike)')

print(popt[0])
