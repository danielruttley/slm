"""
Defines some tools that manipulate holograms.
"""

#import numpy as np

def hori_aper(hologram,center,width):
    """Applies a horizontal aperture onto the hologram and returns a copy. The
    center of the hologram will be moved away from the edges of the hologram so
    that the full width is always shown.
    
    Parameters:
        center: vertical center of the aperture (starting from the top)
        width: width of the aperture
    
    Returns:
        masked_holo: hologram masked with the virtual aperture
        center: horizontal center of the aperture (in case this has changed)
    """
    masked_holo = hologram.copy()
    if center < width/2:
        center = width/2
    if center > hologram.shape[0]-width/2:
        center = hologram.shape[0]-width/2
    start = int(round(center-width/2))
    end = int(round(center+width/2))
    if start < 0:
        start = 0
    masked_holo[:start,:] = 0
    masked_holo[end+1:,:] = 0
    return masked_holo, center

def vert_aper(hologram,center,width):
    """Applies a vertical aperature onto the hologram and returns a copy.

    Parameters:
        center: horizontal center of the aperture (starting from the left)
        width: width of the aperture
    
    Returns:
        masked_holo: hologram masked with the virtual aperture
        center: horizontal center of the aperture (in case this has changed)
    """
    masked_holo = hologram.copy()
    if center < width/2:
        center = width/2
    if center > hologram.shape[1]-width/2:
        center = hologram.shape[1]-width/2
    start = int(round(center-width/2))
    end = int(round(center+width/2))
    if start < 0:
        start = 0
    masked_holo[:,:start] = 0
    masked_holo[:,end+1:] = 0
    return masked_holo, center