"""
Defines apertures that can be applied to holograms.
"""

def hori(hologram,y0=256,width=50,return_center=False):
    """
    Applies an aperature onto the hologram and returns a copy. If y0
    is too small or large so that the entire aperture cannot be displayed, x0
    will be modified to allow this to happen.
    
    Parameters
    ----------
    y0 : float
        center of the aperture on the SLM
    width : float 
        width of the aperture in SLM pixels
    
    Returns
    -------
    masked_holo: hologram masked with the virtual aperture
    """
    masked_holo = hologram.copy()
    center = y0
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
    if return_center:
        return masked_holo, center
    return masked_holo
    # else:


def vert(hologram,x0=256,width=50,return_center=False):
    """
    Applies an aperature onto the hologram and returns a copy. If x0
    is too small or large so that the entire aperture cannot be displayed, x0
    will be modified to allow this to happen.

    Parameters
    ----------
    x0 : float
        center of the aperture on the SLM
    width : float 
        width of the aperture in SLM pixels
    
    Returns
    -------
    masked_holo: hologram masked with the virtual aperture
    """
    masked_holo = hologram.copy()
    center = x0
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
    if return_center:
        return masked_holo, center
    return masked_holo
    # else:

def ellipse(hologram,x0=None,y0=None,radius=None,radius_scale=1):
    """
    Applies an elliptical aperature onto the hologram and returns a copy.
    Currently scale factor is hardcoded to a certain factor but want to add 
    this as an arg in future.

    Parameters
    ----------
    x0,y0 : int 
        center of the aperture on the SLM display. 
        None to use the center of the hologram
    radius : float
        radius of the aperture. None for the largest circular aperture
        that will fit on the display
    
    Returns
    -------
        masked_holo: hologram masked with the virtual aperture
    """
    # radius_scale = (11.5/(38/2)) # scale to apply to y

    masked_holo = hologram.copy()
    if x0 is None:
        x0 = hologram.shape[0]/2
    if y0 is None:
        y0 = hologram.shape[1]/2
    center = (x0,y0)
    if radius is None:
        radius = min((hologram.shape[1]-center[0]),center[0],
                     (hologram.shape[0]-center[1])/radius_scale,center[1]/radius_scale)
    
    xradius = radius
    yradius = radius_scale*radius
    print('radius',radius)

    for i in range(hologram.shape[0]):
        for j in range(hologram.shape[1]):
            if (i-center[1])**2/yradius**2+(j-center[0])**2/xradius**2 > 1:
                masked_holo[i,j] = 0

    # masked_holo = circ_old(masked_holo,x0,y0,radius)
    return masked_holo

def circ(hologram,x0=None,y0=None,radius=None):
    """
    Applies a circular aperature onto the hologram and returns a copy.

    Parameters
    ----------
    x0,y0 : int 
        center of the aperture on the SLM display. 
        None to use the center of the hologram
    radius : float
        radius of the aperture. None for the largest circular aperture
        that will fit on the display
    
    Returns
    -------
        masked_holo: hologram masked with the virtual aperture
    """
    masked_holo = hologram.copy()
    if x0 is None:
        x0 = hologram.shape[0]/2
    if y0 is None:
        y0 = hologram.shape[1]/2
    center = (x0,y0)
    if radius is None:
        radius = min(hologram.shape[1]-center[0],center[0],
                     hologram.shape[0]-center[1],center[1])
    for i in range(hologram.shape[0]):
        for j in range(hologram.shape[1]):
            if (i-center[1])**2+(j-center[0])**2 > radius**2:
                masked_holo[i,j] = 0
    return masked_holo