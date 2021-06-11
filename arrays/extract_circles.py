import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import log10,floor,ceil

def gaussian2D(xy_tuple, amplitude, x0, y0, wx, wy, theta):
    (x,y) = xy_tuple
    sigma_x = wx/2
    sigma_y = wy/2 #function defined in terms of sigmax, sigmay
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                            + c*((y-y0)**2)))
    return g.ravel()

min_distance_between_traps = 30

img = cv2.imread(r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\04\Measure 50\17.png",0)
bgnd = cv2.imread(r"Z:\Tweezer\Code\Python 3.7\slm\images\2021\May\04\Measure 50\bgnds\17_bgnd.png",0)
#bgnd = np.zeros_like(img)
array = np.float32(img) - np.float32(bgnd)
blurred_img = cv2.GaussianBlur(img,(3,3),1)
#img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

(thresh, bwimg) = cv2.threshold(blurred_img, 25, 255, cv2.THRESH_BINARY)
#cimg = cv2.cvtColor(bwimg,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(bwimg,cv2.HOUGH_GRADIENT,1,min_distance_between_traps,
                            param1=200,param2=8,minRadius=10,maxRadius=20)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

# cv2.imshow('detected circles',cimg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

I0s = []
popts = []

for i,circle in enumerate(circles[0]):
    print(i)
    x0,y0,r = circle
    print(x0,y0)
    xmin = max(round(x0-min_distance_between_traps*0.75),0)
    xmax = min(round(x0+min_distance_between_traps*0.75),img.shape[1])
    ymin = max(round(y0-min_distance_between_traps*0.75),0)
    ymax = min(round(y0+min_distance_between_traps*0.75),img.shape[0])
    print(xmin,ymin,xmax,ymax)
    roi = array[ymin:ymax,xmin:xmax]
    max_val = np.max(array)
    x,y = np.meshgrid(np.arange(xmin,xmax),np.arange(ymin,ymax))
    popt, pcov = curve_fit(gaussian2D, (x,y), roi.ravel(), p0=[max_val,x0,y0,r,r,0])#,0])
    popts.append(popt)
    I0s.append(popt[0])
    print(popt)
    perr = np.sqrt(np.diag(pcov))
    data_fitted = gaussian2D((x,y),*popt).reshape(roi.shape[0],roi.shape[1])
    if i == 0:
        fig, (ax1,ax2) = plt.subplots(1, 2)
        fig.set_size_inches(9, 5)
        fig.set_dpi(300)
        c1 = ax1.pcolormesh(x,y,roi,cmap=plt.cm.gray,shading='auto')
        ax1.invert_yaxis()
        ax1.set_title('camera ROI')
        fig.colorbar(c1,ax=ax1,label='pixel count')
        c2 = ax2.pcolormesh(x,y,data_fitted,cmap=plt.cm.viridis,shading='auto')
        ax2.invert_yaxis()
        ax2.set_title('fitted Gaussian')
        fig.colorbar(c2,ax=ax2,label='intensity (arb.)')
        fig.suptitle('({},{})'.format(x0,y0))
        fig.tight_layout()
        plt.show()
    for arg,val,err in zip(gaussian2D.__code__.co_varnames[1:],popt,perr):
        prec = floor(log10(err))
        err = round(err/10**prec)*10**prec
        val = round(val/10**prec)*10**prec
        if prec > 0:
            valerr = '{:.0f}({:.0f})'.format(val,err)
        else:
            valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
        print(arg,'=',valerr)

    print('\n')

fitted_img = np.zeros_like(array)
x, y = np.meshgrid(np.arange(array.shape[1]), range(array.shape[0]))
for popt in popts:
    fitted_img += gaussian2D((x,y),*popt).reshape(array.shape[0],array.shape[1])

fig, (ax1,ax2) = plt.subplots(1, 2)
fig.set_size_inches(9, 5)
fig.set_dpi(300)
c1 = ax1.pcolormesh(x,y,array,cmap=plt.cm.gray,shading='auto')
ax1.invert_yaxis()
ax1.set_title('camera image')
fig.colorbar(c1,ax=ax1,label='pixel count')
c2 = ax2.pcolormesh(x,y,fitted_img,cmap=plt.cm.viridis,shading='auto')
ax2.invert_yaxis()
ax2.set_title('fitted array')
fig.colorbar(c2,ax=ax2,label='intensity (arb.)')
fig.tight_layout()
plt.show()

binwidth=5
fig, ax = plt.subplots(1,1)
fig.set_dpi(300)
ax.hist(I0s, density=False, bins=range(floor(min(I0s)),ceil(max(I0s)+binwidth),binwidth))  # density=False would make counts
ax.set_xlabel(r'fitted Gaussian $I_0$')
ax.set_ylabel('counts')
plt.show()