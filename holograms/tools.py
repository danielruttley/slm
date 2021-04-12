"""
Defines some tools that manipulate holograms.
"""

#import numpy as np

def hori_aper(hologram,center,width):
    """Applies a horizontal aperture onto the hologram and returns a copy.
    
    Parameters:
        center: vertical center of the aperture (starting from the top)
        width: width of the aperture
    """
    masked_holo = hologram.copy()
    start = int(round(center-width/2))
    end = int(round(center+width/2))
    if start < 0:
        start = 0
    masked_holo[:start,:] = 0
    masked_holo[end+1:,:] = 0
    return masked_holo

def vert_aper(hologram,center,width):
    """Applies a vertical aperature onto the hologram and returns a copy.

    Parameters:
        center: horizontal center of the aperture (starting from the left)
        width: width of the aperture
    """
    masked_holo = hologram.copy()
    start = int(round(center-width/2))
    end = int(round(center+width/2))
    if start < 0:
        start = 0
    masked_holo[:,:start] = 0
    masked_holo[:,end+1:] = 0
    return masked_holo