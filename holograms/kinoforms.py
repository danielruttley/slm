#Originally written by Mitch Walker
#Modified by Dan Ruttley
#Note that all kinoforms are returned as 512x512 images which should be padded later if needed.

def blank():
    """Returns a 512x512 hologram of all zeros."""
    import numpy as np
    return np.zeros((512,512))

def blazedg(grad,angle):
    """
    This function takes a gradient and an angle, and produces a blazed grating phase pattern, with fronts perpendicular to the direction of the line which makes the angle given with the positive x axis, in an anticlockwise direction
    
    -grad is the gradient of the blazed grating; a larger value decreases the size of each grating
    -angle determines the orientation of the blazed grating. This value should be in degrees, between 0 and 180. 0 degrees puts the grating vertical; the angle moves anticlockwise with 90 degrees along the positive y axis (horizontal)
    
    returns: a 2D 512x512 array, normalised so that the maximum value is 1, of phase values for the desired blazed grating
    
    PLEASE NOTE: This function is incomplete and as such does not currently work as intended
    """
    
    #Import modules
    import numpy as np
    
    #Convert the angle from degrees into radians
    rad = angle*(np.pi/180)
    
    #Initialise a 2D array which will store the phase value at each blazed grating point
    r = np.empty((512,512))
    
    #Determine the distance of each point in the array from the bottom-left corner (the origin), assuming each entry in the array is spaced 1 unit from its nearest neighbours
    for i in range(512):
        for j in range(512):
            r[i,j] = np.sqrt((j**2)+((511-i)**2))
    
    print(r)
    
def blazediag(grad):
    """
    This function generates a 2D array for a blazed grating phase pattern with diagonal fronts. The fronts can be perpendicular to either the line y=x or the line y=-x
    
    -grad is the gradient of the blazed grating; a larger value decreases the size of each grating. The sign of grad determines the orientation; positive for fronts perpendicular to the line y=x, and negative for fronts perpendicular to the line y=-x
    
    returns: a 2d 512x512 array of phase values normalised such that the maximum value is 2pi
    """
    
    #Import modules
    import numpy as np
    
    #Initialise array to contain data
    array = np.empty((512,512))
    
    #Grating with fronts perpendicular to line x=y
    if grad>0:
        #Fill the array based on the distance of each element from the bottom left element, assuming unit spacing between nearest neighbours
        for i in range(512):
            for j in range(512):
                array[i,j] = j + (511-i)

        #Convert these values into phase values based on the gradient, and modulo between 0 and 2pi
        array = (np.abs(grad)*array)%(2*np.pi)
 
    #Grating with fronts perpendicular to line x=-y
    else:
        #Fill the array based on the distance of each element from the top left element, assuming unit spacing between nearest neighbours
        for i in range(512):
            for j in range(512):
                array[i,j] = i + j
        
        #Convert these values into phase values based on the gradient, and modulo between 0 and 2pi
        array = (np.abs(grad)*array)%(2*np.pi)

    return array

def blazever(grad):
    """
    This function generates a 2D array for a blazed grating phase pattern with vertical fronts.
    
    -grad is the gradient of the blazed grating; a larger value decreases the size of each grating.
    
    returns: a 2d 512x512 array of phase values normalised such that the maximum value is 2pi
    """
    
    #Import modules
    import numpy as np
    
    #Initialise a 1D array of length 512, with each entry spaced by a unit from the previous entry
    hor = np.arange(0,512,1)
    
    #Convert these values into phase values based on the gradient, and modulo between 0 and 2pi
    hor = (np.abs(grad)*hor)%(2*np.pi)
    
    #Make a vertical n by 1 matrix to conjugate the phase values with
    vert = np.ones((512,1))
    
    #Conjugate the vertical array with a horizontal array of ones to create a 2D matrix
    twodimarray = vert*hor
    
    return twodimarray
    
def blazehor(grad):
    """
    This function generates a 2D array for a blazed grating phase pattern with horizontal fronts.
    
    -grad is the gradient of the blazed grating; a larger value decreases the size of each grating.
    
    returns: a 2d 512x512 array of phase values normalised such that the maximum value is 2pi
    """
    
    #Import modules
    import numpy as np
    
    #Create an empty 2D vertical array, length 512
    vert = np.ones((512,1))
    
    #Convert these values into phase values based on the gradient, and modulo between 0 and 2pi
    for i in range(512):
        vert[i,0] = (i*grad)%(2*np.pi)
        
    #Create a horizontal 1 by n matrix to conjugate the phase values with
    hor = np.ones((1,512))
    
    #Conjugate the vertical array with a horizontal array of ones to create a 2D matrix
    twodimarray = vert*hor
    
    return twodimarray

def stepver(v1,v2,split):
    """
    This function generates a 2D array for a step function, with the split in the vertical direction. The two sections are on the left and right.
    
    -v1 is an integer value between 0 and 65535, which determines the phase value of half of the kinoform. This value is on the left.
    -v2 is another integer value between 0 and 65535, which determines the phase value for the other half of the kinoform. This value is on the right.
    -split is an integer between 0 and 512, which determines the pixel value where the step is. 0 is on the left, 512 on the right.
    
    returns: a 2d 512x512 array of phase values between 0 and 65535
    """
    
    #Import modules
    import numpy as np    
    
    #Initialise the array to hold the mask values
    maskval = np.empty(512)
    
    #Fill the mask value array
    for i in range(512):
        if i < split:
            maskval[i] = v1
            
        elif i >= split:
            maskval[i] = v2
            
    #Initialise a vertical matrix to multiply the mask value array by to form a 2D matrix    
    vert = np.ones((512,1))

    #Multiply the two arrays
    twodim = vert*maskval

    return twodim

def stephor(v1,v2,split):
    """
    This function generates a 2D array for a step function, with the split in the horizontal direction. The two sections are on the top and bottom.
    
    -v1 is an integer value between 0 and 65535, which determines the phase value of half of the kinoform. This value is on the top.
    -v2 is another integer value between 0 and 65535, which determines the phase value for the other half of the kinoform. This value is on the bottom.
    -split is an integer between 0 and 512, which determines the pixel value where the step is. 0 is on the top, 512 on the bottom.
    
    returns: a 2d 512x512 array of phase values between 0 and 65535
    """
    
    #Import modules
    import numpy as np    
    
    #Initialise the array to hold the mask values
    maskval = np.empty((512,1))
    
    #Fill the mask value array
    for i in range(512):
        if i < split:
            maskval[(i,0)] = v1
            
        elif i >= split:
            maskval[(i,0)] = v2
       
    #Initialise a horizontal matrix to multiply the mask value array by to form a 2D matrix    
    hor = np.ones(512)

    #Multiply the two arrays
    twodim = maskval*hor
            
    return twodim

def lagpol(x,p,l):
    """
    This function returns the Laguerre polynomial value for x, given p and l
    
    -x is a float
    -p is an integer between 1 and 6
    -l is a positive integer
    """

    if p == 0:
        val = 1
    elif p == 1:
        val = 1 - x
    elif p == 2:
        val = (1/2)*(x**2 - 4*x + 2)
    elif p == 3:
        val = (1/6)*(-1*(x**3) + 9*(x**2) - 18*x + 6)
    elif p == 4:
        val = (1/24)*(x**4 - 16*(x**3) + 72*(x**2) - 96*x + 24)
    elif p == 5:
        val = (1/120)*(-1*(x**5) + 25*(x**4) - 200*(x**3) + 600*(x**2) - 600*x + 120)
    elif p == 6:
        val = (1/720)*(x**6 - 36*(x**5) + 450*(x**4) - 2400*(x**3) + 5400*(x**2) - 4320*x + 720)

    return val    

def lag_poly(x,p,l):
    import scipy.special as spec
    import numpy as np
    return spec.assoc_laguerre(x,n=p,k=np.abs(l))

def lgmode(x0,y0,p,l,w0):
    """
    This function generates a 2D array for an LG beam mode of 
    
    -x0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the x direction, with 1 at the left and 512 at the right
    -y0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the y direction, with 1 at the top and 512 at the bottom
    -p is an integer between 0 and 6
    -l is an positive integer
    -w0 is a float, and is the waist of the beam in pixels
    
    """
    
    #Import the necessary modules to run the function
    import numpy as np
    import math
    
    #Work out the coordinates based on the origin (x0,y0)
    cartescoord = np.empty((512,512,2))
    
    for i in range(512):
        for j in range(512):
            cartescoord[i,j,0] = (j) - x0
            
            cartescoord[j,i,1] = (j) - y0
            
    #Convert the coordinates from Cartesian to cylindrical polar, taking z to be constant, z=0
    cylincoord = np.empty((512,512,2))

    for i in range(512):
        for j in range(512):
            cylincoord[i,j,0] = np.sqrt((cartescoord[i,j,0]**2)+(cartescoord[i,j,1]**2))
            
            if j == x0:
                if i > y0:
                    cylincoord[i,j,1] = np.pi/2
                elif i < y0:
                    cylincoord[i,j,1] = (3/2)*np.pi
                elif i == y0:
                    cylincoord[i,j,1] = 0
            elif i == y0:
                if j < x0:
                    cylincoord[i,j,1] = np.pi
                elif j > x0:
                    cylincoord[i,j,1] = 0
                elif j == x0:
                    cylincoord[i,j,1] = 0
            elif j>x0 and i>y0:
                cylincoord[i,j,1] = np.arctan(np.abs(cartescoord[i,j,1]/cartescoord[i,j,0]))
            elif j<x0 and i>y0:
                cylincoord[i,j,1] = (np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j<x0 and i<y0:
                cylincoord[i,j,1] = np.pi + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j>x0 and i<y0:
                cylincoord[i,j,1] = ((2)*np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
                
            if np.isnan(cylincoord[i,j,1]) == True:
                print("Not a number ", i, " ", j)
                cylincoord[i,j,1] = 0
            
    
    #Generate and return the phase values
    phase = np.empty((512,512))
    
    laguerre_vals = lag_poly((2 * (cylincoord[:,:,0]**2))/(w0**2),p,l)
    
                
    for i in range(512):
        for j in range(512):
            if laguerre_vals[i,j] <= 0:
                phase[i,j] = (-l*cylincoord[i,j,1])%(2*np.pi)
            elif laguerre_vals[i,j] > 0:
                phase[i,j] = (((-l*cylincoord[i,j,1]) + np.pi)%(2*np.pi))
            
    return phase

def gaussian(x0,y0,w0,wlen=1024*1e-9):
    """
    This function returns a 2D array of phase that would generate a Gaussian beam from a flat sheet of light with constant phase
    
    -x0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the x direction, with 1 at the left and 512 at the right
    -y0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the y direction, with 1 at the top and 512 at the bottom
    -w0 is a float, and is the waist of the beam in pixels
    """
    
    #Import the necessary modules to run the function
    import numpy as np
    import math
    
    #Work out some important values
    k = (2*np.pi)/wlen
    
    #Work out the coordinates based on the origin (x0,y0)
    cartescoord = np.empty((512,512,2))
    
    for i in range(512):
        for j in range(512):
            cartescoord[i,j,0] = (j) - x0
            
            cartescoord[j,i,1] = (j) - y0
            
    #Convert the coordinates from Cartesian to cylindrical polar, taking z to be constant, z=0
    cylincoord = np.empty((512,512,2))

    for i in range(512):
        for j in range(512):
            cylincoord[i,j,0] = np.sqrt((cartescoord[i,j,0]**2)+(cartescoord[i,j,1]**2))
            
            if j == x0:
                if i > y0:
                    cylincoord[i,j,1] = np.pi/2
                elif i < y0:
                    cylincoord[i,j,1] = (3/2)*np.pi
                elif i == y0:
                    cylincoord[i,j,1] = 0
            elif i == y0:
                if j < x0:
                    cylincoord[i,j,1] = np.pi
                elif j > x0:
                    cylincoord[i,j,1] = 0
                elif j == x0:
                    cylincoord[i,j,1] = 0
            elif j>x0 and i>y0:
                cylincoord[i,j,1] = np.arctan(np.abs(cartescoord[i,j,1]/cartescoord[i,j,0]))
            elif j<x0 and i>y0:
                cylincoord[i,j,1] = (np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j<x0 and i<y0:
                cylincoord[i,j,1] = np.pi + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j>x0 and i<y0:
                cylincoord[i,j,1] = ((2)*np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
                
            if np.isnan(cylincoord[i,j,1]) == True:
                print("Not a number ", i, " ", j)
                cylincoord[i,j,1] = 0
                
    phase = np.empty((512,512))
    
    return phase

def converginglens(w,f,x0,y0):
    """
    This function generates a 2D array for a thin converging lens with focal length f
    
    -w is a positive float, and is the wavelength of the laser, in m
    -f is a positive float, and is the focal length of the lens the kinoform is modelling, in m
    -x0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the x direction, with 1 at the left and 512 at the right
    -y0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the y direction, with 1 at the top and 512 at the bottom
    """
    
    #Import the necessary modules to run the function
    import numpy as np
    import math
    
    #Define the wavenumber k
    k = (2*np.pi)/w
    
    #Work out the coordinates based on the origin (x0,y0)
    cartescoord = np.empty((512,512,2))
    
    for i in range(512):
        for j in range(512):
            cartescoord[i,j,0] = ((j) - x0) * 15*(10**-6)
            
            cartescoord[j,i,1] = ((j) - y0) * 15*(10**-6)
            
    #Convert the coordinates from Cartesian to cylindrical polar, taking z to be constant, z=0
    cylincoord = np.empty((512,512,2))

    for i in range(512):
        for j in range(512):
            cylincoord[i,j,0] = np.sqrt((cartescoord[i,j,0]**2)+(cartescoord[i,j,1]**2))
            
            if j == x0:
                if i > y0:
                    cylincoord[i,j,1] = np.pi/2
                elif i < y0:
                    cylincoord[i,j,1] = (3/2)*np.pi
                elif i == y0:
                    cylincoord[i,j,1] = 0
            elif i == y0:
                if j < x0:
                    cylincoord[i,j,1] = np.pi
                elif j > x0:
                    cylincoord[i,j,1] = 0
                elif j == x0:
                    cylincoord[i,j,1] = 0
            elif j>x0 and i>y0:
                cylincoord[i,j,1] = np.arctan(np.abs(cartescoord[i,j,1]/cartescoord[i,j,0]))
            elif j<x0 and i>y0:
                cylincoord[i,j,1] = (np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j<x0 and i<y0:
                cylincoord[i,j,1] = np.pi + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j>x0 and i<y0:
                cylincoord[i,j,1] = ((2)*np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
                
            if np.isnan(cylincoord[i,j,1]) == True:
                print("Not a number ", i, " ", j)
                cylincoord[i,j,1] = 0

    #Generate and return the phase values
    phase = np.empty((512,512))
    
    for i in range(512):
        for j in range(512):
            phase[i,j] = (k * (cylincoord[i,j,0]**2)) / (2*f)
            
    phase = phase%(2*np.pi)
    
    return phase

def diverginglens(w,f,x0,y0):
    """
    This function generates a 2D array for a thin diverging lens with focal length f
    
    -w is a positive float, and is the wavelength of the laser, in m
    -f is a positive float, and is the focal length of the lens the kinoform is modelling, in m
    -x0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the x direction, with 1 at the left and 512 at the right
    -y0 is an integer between 1 and 512, and is the position of the pixel to be taken as the origin in the y direction, with 1 at the top and 512 at the bottom
    """
    
    #Import the necessary modules to run the function
    import numpy as np
    import math
    
    #Define the wavenumber k
    k = (2*np.pi)/w
    
    #Work out the coordinates based on the origin (x0,y0)
    cartescoord = np.empty((512,512,2))
    
    for i in range(512):
        for j in range(512):
            cartescoord[i,j,0] = ((j) - x0) * 15*(10**-6)
            
            cartescoord[j,i,1] = ((j) - y0) * 15*(10**-6)
            
    #Convert the coordinates from Cartesian to cylindrical polar, taking z to be constant, z=0
    cylincoord = np.empty((512,512,2))

    for i in range(512):
        for j in range(512):
            cylincoord[i,j,0] = np.sqrt((cartescoord[i,j,0]**2)+(cartescoord[i,j,1]**2))
            
            if j == x0:
                if i > y0:
                    cylincoord[i,j,1] = np.pi/2
                elif i < y0:
                    cylincoord[i,j,1] = (3/2)*np.pi
                elif i == y0:
                    cylincoord[i,j,1] = 0
            elif i == y0:
                if j < x0:
                    cylincoord[i,j,1] = np.pi
                elif j > x0:
                    cylincoord[i,j,1] = 0
                elif j == x0:
                    cylincoord[i,j,1] = 0
            elif j>x0 and i>y0:
                cylincoord[i,j,1] = np.arctan(np.abs(cartescoord[i,j,1]/cartescoord[i,j,0]))
            elif j<x0 and i>y0:
                cylincoord[i,j,1] = (np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j<x0 and i<y0:
                cylincoord[i,j,1] = np.pi + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
            elif j>x0 and i<y0:
                cylincoord[i,j,1] = ((2)*np.pi) + np.arctan(cartescoord[i,j,1]/cartescoord[i,j,0])
                
            if np.isnan(cylincoord[i,j,1]) == True:
                print("Not a number ", i, " ", j)
                cylincoord[i,j,1] = 0

    #Generate and return the phase values
    phase = np.empty((512,512))
    
    for i in range(512):
        for j in range(512):
            phase[i,j] = (k * (cylincoord[i,j,0]**2)) / (2*f)
            
    phase = phase%(2*np.pi)
    
    phase = (2*np.pi) - phase
    
    return phase

def focallength(s_two):
    """
    A function which takes the distance from the lens that the focal plane should be, and outputs the focal length of the lens that will correctly position the focal plane
    
    -s_two is a positive non-zero float, and is the distance from the Fourier lens to the focal plane in m. The SLM is positioned 0.45m from the Fourier lens"""
    
    f = 0.5
    d = 0.45
    
    s_one = 1/((1/f) - (1/s_two))
    
    f_prime = s_one - d
    
    return f_prime