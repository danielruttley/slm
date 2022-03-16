"""
14/06/2021
Calculate the expected axial shift of tweezer with SLM lens hologram focal length.
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from os.path import dirname as up
fig_dir = up(up(up(up(__file__))))
#plt.style.use(fig_dir+r'\vinstyle.mplstyle')
plt.close('all')


def dotter(*args):
    """
    Do dot product of many arrays. First optic first!
    So when you do ABCD matrics, typically:   M = M4 M3 M2 M1
    but here you supply the arguments (dotter(M1, M2, M3, M4)) - 
    more human-readable when plugging in an optics system. 
    """
    M = np.array([[1,0],
                  [0,1]])
    for arg in args:
        M = np.dot(arg, M)
    return M

def Mx(x):
    """
    return ABCD matric for free-space propagation
    """
    return np.array([[1, x],
                      [0, 1]])

def Mf(f):
    """return ABCD matrix for a thin lens"""
    return np.array([[1, 0],
                      [-1/f, 1]])


f_obj = 0.035 # metres
x_obj = np.linspace(34.3,35.3, 100000)/1000  # axial distance at atoms over which to calculate beam profile


def ABCDsolution(M, w0, R0, wavelength):
    """
    Solution for w1 and R1 after solving on paper (photo in onenote>QSUMshared>optical trapping>dichroic flatness)
        complex beam paramter is q=-i(wavelength/pi*w0^2)+1/R0
        complex beam parameter after related to before by ABCD matrix:
        q' = (A*q+B)/(C*q+D)
        function rearranges this to solve for real and im parts of q' to get w1, R1
    Supply the ABCD matrix to solve and the waist and radius of curvature of the ingoing beam.
    """
    A,B,C,D = M[0,0], M[0,1], M[1,0], M[1,1]
    a = wavelength / (np.pi * w0 **2)
    b = 1/R0
    w = B * (a**2+b**2) + A*b
    x = a*A
    y = D * (a**2+b**2) + C*b
    z = a*C

    u = (y*w + x*z) / (w**2 + x**2)
    v = (z*w-x*y) / (w**2+x**2)

    w1 = np.sqrt(np.abs(wavelength / (np.pi*v)))
    R1 = 1/u
    
    return w1, R1


fmax = 1000.  # max range of SLM focal length to plot
fs = np.linspace(-fmax,fmax,10000)
foci = np.zeros(len(fs))
waists = np.zeros(len(fs))


colors = plt.cm.viridis(np.linspace(0, 1,len(fs)))


def focusPlotter():
    """ 
        Function to easily vary one of the parameters in the ABCD matrix to be solved
        and return the waist / focus position as a function of one of those parameters.
    """
    for i in range(len(fs)):
        if i%100 == 0:
            print(i)
        #    Optics path for 1064nm setup using SLM as a lens:
        #    beam -> 35mm lens -> 285mm -> 250mm lens -> SLM (as lens)-> 125mm lens -> 525mm -> 400mm lens -> 0.5m to objective -> 35mm to atoms .
        M_tweezer = dotter(Mf(0.035), Mx(0.285), Mf(0.250), Mx(0.25), Mf(fs[i]), Mx(0.275), Mf(0.075), Mx(0.475), Mf(0.400), Mx(0.500), Mf(f_obj),  Mx(x_obj))
        wi, Ri = ABCDsolution(M_tweezer, w0 = 0.0006, R0 = 1000, wavelength = 1064e-9)
        
       # plt.plot((x_obj-0.035)*1e6, wi*1e6, label = str(fs[i]), color = colors[i]) 
        
        foci[i] = x_obj[np.argmin(wi)]
        waists[i] = np.min(wi)
    return foci-f_obj


#plt.figure()#(figsize=(4,4))
#plt.subplots_adjust(bottom=0.15)


#plt.plot(fs, (focusPlotter())*1e6, label = 'f$_\mathrm{cyl}$ = ' + str(0.2) + 'm')
#plt.xlabel('$f_\mathrm{SLM}$ (m)')
#plt.ylabel('1064nm Tweezer Focus Shift ($\mu$m)')
##plt.xlim(min(fs), max(fs))
##plt.ylim(-10,10)
#plt.title('1064nm tweezer axial position with SLM focal length')
#plt.show()

df = pd.DataFrame()
df['f [m]'] = fs
df['focus shift [um]'] = (focusPlotter())*1e6
df = df.drop(df[np.abs(df['focus shift [um]']) > 200].index)
df.to_csv('f_vs_focal_shift.csv',index=False)



os.chdir(os.path.dirname(os.path.abspath(__file__)))
root = os.getcwd()
os.chdir(root)

# plt.savefig('ABCDresults.svg', dpi=130)
# plt.savefig('ABCDresults.pdf', dpi=130)
# 
# import pandas as pd
# measured = pd.DataFrame({'z_cyl (mm)':ds*1e3, 'f=0.2m': (focusPlotter(0.2))*1e6, 'f-0.5m':(focusPlotter(0.5))*1e6, 'f=1.0m':(focusPlotter(0.2))*1e6})
# data2save = pd.concat([measured], axis=1).reset_index(drop=True)
# print(data2save)
# data2save.to_csv(root + '\\'+'plottedData.csv')
# 
# 
# 
