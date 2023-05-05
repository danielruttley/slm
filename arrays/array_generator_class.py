"""A class to generate holograms for arrays of multiple Gaussian beams. This 
class manages the SLM and camera together to iteratively improve the hologram.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft2,ifft2,fftshift,ifftshift
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
from math import log10,floor,ceil

import holograms as hg
from .image_loading import load_traps_from_image

import sys
sys.path.append('..')
from holograms.arrays import aags

class ArrayGenerator():
    def __init__(self,slm,camera,image_handler,extra_holos=None,
                 shape=(512,512),circ_aper_center=None,circ_aper_radius=None,
                 calibration_blazing=None, input_waist=None, 
                 input_center=(256,256),correction_factor=0.5,
                 min_distance_between_traps=3,
                 remove_low_zernike=True,padding=1):
        """Generates and iteratively improves Gerchberg Saxton holograms for 
        arrays of Gaussian traps.

        Parameters
        ----------
        slm : SLM
            Initialised SLM class to display holograms on.
        camera : Camera
            ThorLabs DCC1545M-GL camera class to capture images with.
        extra_holos : array, optional
            Extra holograms in the range 0-1 to apply to any generated 
            holograms. This will normally contain a blazed grating and perhaps 
            a Fresnel lens.
        shape : tuple of int
            The resolution of the SLM in pixels (x,y).
        circ_aper_center : tuple of float
            The center of a circular aperture applied in a blazed grating in 
            extra_holos. This is taken into account in the incident intensity 
            on the SLM.
        
        Returns
        -------
        None
        """
        self.slm = slm
        self.cam = camera
        self.imager = image_handler
        self.shape = shape
        self.circ_aper_center = circ_aper_center
        self.circ_aper_radius = circ_aper_radius
        self.extra_holos = extra_holos
        self.calibration_blazing = calibration_blazing
        self.padding = padding

        self.input_waist = input_waist
        self.input_center = input_center
        self.correction_factor = correction_factor
        self.min_distance_between_traps = min_distance_between_traps
        self.remove_low_zernike = remove_low_zernike

        self.intensity_correction_iteration = 0

        self.trap_df = pd.DataFrame()
        self.traps = None
        self.T = np.zeros(self.shape)
        self.phi = None
        self.psi_N = None

        self.imager.create_dirs()
        self.trap_df_filename = self.imager.get_dir()+'/trap_df.csv'

    def set_traps(self,traps):
        """Set the location of the Gaussian traps to be used in the array.

        Parameters
        ----------
        traps : list of tuple of int
            A list containing the tuples (y,x) of the locations of the traps.
        
        Returns
        -------
        None
        """
        self.traps = traps
        holo_ys = [y for (y,x) in traps]
        holo_xs = [x for (y,x) in traps]
        self.trap_df['holo_x'] = holo_xs
        self.trap_df['holo_y'] = holo_ys
        self.trap_df['holo_x'] = self.trap_df['holo_x'].astype(int)
        self.trap_df['holo_y'] = self.trap_df['holo_y'].astype(int)
    
    def load_traps_from_image(self,filename,scale):
        traps = load_traps_from_image(filename,scale,self.shape)
        self.set_traps(traps)
    
    def generate_initial_hologram(self,iterations=50,traps=None,save_to_param=True):
        """Calculates the adaptive additive Gerchberg Saxton hologram for a 
        given list of uniform traps to create the initial Gaussian trap array 
        before any depth optimisation.

        Parameters
        ----------
        iterations : int, optional
            The number of iterations that the AAGS algorithm should run for.
        
        Returns
        -------
        None
            The created hologram is saved to the object's array_holo parameter.
        """
        if traps == None:
            traps = self.traps
        holo = aags(traps,iterations,self.padding,self.input_waist,self.input_center,
                    self.shape,remove_low_zernike=self.remove_low_zernike)
        if self.circ_aper_center is not None:
            holo = hg.apertures.circ(holo,x0=self.circ_aper_center[0],y0=self.circ_aper_center[1],radius=self.circ_aper_radius)
        if self.extra_holos is not None:
            holo = (holo+self.extra_holos)%1
        if save_to_param:
            self.array_holo = holo
        return holo

    def get_trap_depths(self,reps=1,plot=False):
        """Applies the pre-generated hologram onto the SLM and uses the 
        pre-obtained trap_df to fit Gaussians to each of the traps in the 
        array.

        Parameters
        ----------
        reps : int, optional
            The number of times the hologram should be applied before the 
            resultant trap depths are averaged and saved.
        plot : bool, optional
            Whether the camera image and the fitted array should be plotted 
            each time the array is fitted.

        Returns
        -------
        None
            The resultant trap depths are saved in the object's trap_df
            parameter.
        """
        poptss = []
        perrss = []
        for i in range(reps):
            self.slm.apply_hologram(self.array_holo)
            time.sleep(0.5)
            self.cam.auto_gain_exposure()
            image = self.cam.take_image()
            # self.slm.apply_hologram(hg.blank())
            # time.sleep(0.5)
            # bgnd = self.cam.take_image()
            # image.add_background(bgnd)
            image.add_hologram(self.array_holo)
            # array = image.get_bgnd_corrected_array()
            array = image.get_array_float()
            image.add_property('intensity_correction_iteration',self.intensity_correction_iteration)
            image.add_property('rep',i)
            self.imager.save(image)
            if self.calibration_blazing is not None:
                self.take_calibration_image()
            popts,perrs = self.find_traps(array,plot)
            poptss.append(popts)
            perrss.append(perrs)
        popts = np.average(np.asarray(poptss), axis=0).T.tolist()
        perrs = np.average(np.asarray(perrss), axis=0).T.tolist()
        print(popts,perrs)
        
        for arg,popt,perr in zip(self.gaussian2D.__code__.co_varnames[2:],popts,perrs):
            self.trap_df[arg] = popt
            self.trap_df[arg+'_err'] = perr
            self.trap_df[arg+'_'+str(self.intensity_correction_iteration)] = self.trap_df[arg]
            self.trap_df[arg+'_err_'+str(self.intensity_correction_iteration)] = self.trap_df[arg+'_err']
        holo_Is = [x[2] for x in self.traps]
        self.trap_df['holo_I'] = holo_Is
        self.trap_df['holo_I'+str(self.intensity_correction_iteration)] = holo_Is
        self.save_trap_df()

    def generate_corrected_hologram(self,iterations=50):
        """Calculates the corrected adaptive additive Gerchberg Saxton hologram
        using the measured trap depths to aim for a more uniform trap array.

        Parameters
        ----------
        iterations : int, optional
            The number of iterations that the AAGS algorithm should run for.
        
        Returns
        -------
        None
            The created hologram is saved to the object's array_holo parameter.
        """
        self.intensity_correction_iteration += 1
        average_I = self.trap_df['I0'].mean()
        average_wx = self.trap_df['wx'].mean()
        average_wy = self.trap_df['wy'].mean()
        self.traps = []
        for i, row in self.trap_df.iterrows():
            x = int(row['holo_x'])
            y = int(row['holo_y'])
            I_new = row['holo_I'] * (row['target_I']/self.trap_df['target_I'].mean()/(row['I0']/average_I))**self.correction_factor # normalised based off intensity
            # I_new = row['holo_I'] * (row['target_I']/self.trap_df['target_I'].mean()/(row['I0']/average_I)/(row['wx']/average_wx)/(row['wy']/average_wy))**self.correction_factor # normalise based off power
            self.traps.append((y,x,I_new))
            # self.T[trap[:2]] /= np.sqrt((row['I0']/I)/(self.trap_df['holo_I'].mean()/I0_N))
            # G = 0.7
            # self.T[trap[:2]] = np.sqrt((I0_N*I) / (1-G*(1-row['I0']/(I0_N*I))))
            print('================')
            print('target_I',row['target_I'])
            print('average_measured_I',average_I)
            print('average_target_I',self.trap_df['target_I'].mean())
            print('measured I',row['I0'])
            print('prev holo_I',row['holo_I'])
            print('new holo_I',I_new)
            print('\n')
        N = len(self.traps)
        print('Corrected array generation: {} traps'.format(N))
        self.array_holo = aags(self.traps,iterations,self.padding,self.input_waist,
                               self.input_center,self.shape,
                               remove_low_zernike=self.remove_low_zernike)
        if self.circ_aper_center is not None:
            self.array_holo = hg.apertures.circ(self.array_holo,x0=self.circ_aper_center[0],y0=self.circ_aper_center[1],radius=self.circ_aper_radius)
        if self.extra_holos is not None:
            self.array_holo = (self.array_holo+self.extra_holos)%1

    def gaussian2D(self,xy_tuple,I0,x0,y0,wx,wy,theta):
        (x,y) = xy_tuple
        sigma_x = wx/2
        sigma_y = wy/2 #function defined in terms of sigmax, sigmay
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = I0*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()

    def find_traps(self,array,plot=False,width=2):
        min_distance_between_traps = self.min_distance_between_traps
        popts = []
        perrs = []
        cam_roi = self.cam.get_roi()
        show = True
        for i, row in self.trap_df.iterrows():
            print(i)
            x0 = row['img_x']
            y0 = row['img_y']
            r = width
            print(x0,y0)
            if cam_roi is not None:
                xmin = max(round(x0-cam_roi[0]-min_distance_between_traps*0.75),0)
                xmax = min(round(x0-cam_roi[0]+min_distance_between_traps*0.75),array.shape[1])
                ymin = max(round(y0-cam_roi[1]-min_distance_between_traps*0.75),0)
                ymax = min(round(y0-cam_roi[1]+min_distance_between_traps*0.75),array.shape[0])
            else:
                xmin = max(round(x0-min_distance_between_traps*0.75),0)
                xmax = min(round(x0+min_distance_between_traps*0.75),array.shape[1])
                ymin = max(round(y0-min_distance_between_traps*0.75),0)
                ymax = min(round(y0+min_distance_between_traps*0.75),array.shape[0])
            print(xmin,ymin,xmax,ymax)
            roi = array[ymin:ymax,xmin:xmax]
            if show:
                plt.figure()
                plt.imshow(array)
                plt.show(block=False)
                plt.pause(3)
                plt.close()
                show = False
            # plt.figure()
            # plt.imshow(roi)
            # plt.show()
            # time.sleep(10)
            print('array_max',np.max(array))
            print('roi_max',np.max(roi))
            max_val = np.max(array)
            x,y = np.meshgrid(np.arange(xmin,xmax),np.arange(ymin,ymax))
            if cam_roi is not None:
                x += cam_roi[0]
                y += cam_roi[1]
            # plt.pcolormesh(x,y,roi)
            # plt.colorbar()
            # plt.show()
            popt, pcov = curve_fit(self.gaussian2D, (x,y), roi.ravel(), p0=[max_val,x0,y0,r,r,0])
            perr = np.sqrt(np.diag(pcov))
            popts.append(popt)
            perrs.append(perr)
            data_fitted = self.gaussian2D((x,y),*popt).reshape(roi.shape[0],roi.shape[1])
            for arg,val,err in zip(self.gaussian2D.__code__.co_varnames[2:],popt,perr):
                prec = floor(log10(err))
                err = round(err/10**prec)*10**prec
                val = round(val/10**prec)*10**prec
                if prec > 0:
                    valerr = '{:.0f}({:.0f})'.format(val,err)
                else:
                    valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
                print(arg,'=',valerr)
            print('\n')

        if plot:
            fitted_img = np.zeros_like(array)
            x, y = np.meshgrid(np.arange(array.shape[1]), range(array.shape[0]))
            if cam_roi is not None:
                x += cam_roi[0]
                y += cam_roi[1]
            for popt in popts:
                fitted_img += self.gaussian2D((x,y),*popt).reshape(array.shape[0],array.shape[1])

            fig, (ax1,ax2) = plt.subplots(2, 1)
            fig.set_size_inches(9, 5)
            fig.set_dpi(100)
            c1 = ax1.pcolormesh(x,y,array,cmap=plt.cm.gray,shading='auto')
            ax1.invert_yaxis()
            ax1.set_title('camera image')
            fig.colorbar(c1,ax=ax1,label='pixel count')
            c2 = ax2.pcolormesh(x,y,fitted_img,cmap=plt.cm.viridis,shading='auto')
            ax2.invert_yaxis()
            ax2.set_title('fitted array')
            fig.colorbar(c2,ax=ax2,label='intensity (arb.)')
            fig.tight_layout()
            plt.show(block=False)
            plt.pause(3)
            plt.close()
        return popts,perrs

    def get_single_trap_coords(self):
        cam_roi = self.cam.get_roi()
        for i,trap in enumerate(self.traps):
            print('---TRAP {} OF {}---'.format(i+1,len(self.traps)))
            self.trap_df.loc[i,'holo_x'] = trap[1]
            self.trap_df.loc[i,'holo_y'] = trap[0]
            try:
                self.trap_df.loc[i,'target_I'] = trap[2]
            except IndexError: # if intensity not specified set as 1
                self.trap_df.loc[i,'target_I'] = 1
            try:
                need_to_measure = np.isnan(self.trap_df.loc[i,'img_x']) or np.isnan(self.trap_df.loc[i,'img_x'])
            except KeyError:
                need_to_measure = True
            if need_to_measure:
                holo = self.generate_initial_hologram(5,[trap],save_to_param=False)
                
                self.slm.apply_hologram(holo)
                time.sleep(0.5)
                self.cam.auto_gain_exposure()
                image = self.cam.take_image()
                # self.slm.apply_hologram(hg.blank())
                # time.sleep(0.5)
                # bgnd = self.cam.take_image()
                # image.add_background(bgnd)
                
                # array = image.get_bgnd_corrected_array()
                array = image.get_array_float()
                x,y = np.meshgrid(np.arange(array.shape[1]),np.arange(array.shape[0]))
                cent_ind = [x for x in np.unravel_index(array.argmax(), array.shape)]
                if cam_roi is not None:
                    x += cam_roi[0]
                    y += cam_roi[1]
                    cent_ind[0] += cam_roi[1]
                    cent_ind[1] += cam_roi[0]
                print(cent_ind[1],cent_ind[0])
                popt, pcov = curve_fit(self.gaussian2D, (x,y), array.ravel(), p0=[np.max(array),cent_ind[1],cent_ind[0],15,15,0])
                perr = np.sqrt(np.diag(pcov))
                for arg,val,err in zip(self.gaussian2D.__code__.co_varnames[2:],popt,perr):
                    prec = floor(log10(err))
                    err = round(err/10**prec)*10**prec
                    val = round(val/10**prec)*10**prec
                    if prec > 0:
                        valerr = '{:.0f}({:.0f})'.format(val,err)
                    else:
                        valerr = '{:.{prec}f}({:.0f})'.format(val,err*10**-prec,prec=-prec)
                    print(arg,'=',valerr)
                print('\n')
                self.trap_df.loc[i,'img_x'] = popt[1]
                self.trap_df.loc[i,'img_y'] = popt[2]
                self.save_trap_df()
                # fig, (ax1,ax2) = plt.subplots(1, 2)
                # fig.set_size_inches(9, 5)
                # fig.set_dpi(100)
                # c1 = ax1.pcolormesh(x,y,array,cmap=plt.cm.gray,shading='auto')
                # ax1.invert_yaxis()
                # ax1.set_title('camera image')
                # fig.colorbar(c1,ax=ax1,label='pixel count')
                # c2 = ax2.pcolormesh(x,y,self.gaussian2D((x,y),*popt).reshape(array.shape[0],array.shape[1]),cmap=plt.cm.viridis,shading='auto')
                # ax2.invert_yaxis()
                # ax2.set_title('fitted array')
                # fig.colorbar(c2,ax=ax2,label='intensity (arb.)')
                # fig.tight_layout()
                # plt.show()
            else:
                print('Read position from csv. Skipping.')
    
    def save_trap_df(self,filename=None):
        if filename is None:
            filename = self.trap_df_filename
        self.trap_df['holo_x'] = self.trap_df['holo_x'].astype(int)
        self.trap_df['holo_y'] = self.trap_df['holo_y'].astype(int)
        self.trap_df.to_csv(filename,float_format='{:f}'.format,encoding='utf-8')

    def load_trap_df(self,filename=None):
        if filename is None:
            filename = self.trap_df_filename
        self.trap_df = pd.read_csv(filename,index_col=0)
        self.trap_df['holo_x'] = self.trap_df['holo_x'].astype(int)
        self.trap_df['holo_y'] = self.trap_df['holo_y'].astype(int)
        self.traps = [(y,x) for y,x in zip(self.trap_df['holo_y'],self.trap_df['holo_x'])]

    def load_prev_trap_locs(self,filename):
        prev_df = pd.read_csv(filename,index_col=0)
        prev_df['holo_x'] = self.trap_df['holo_x'].astype(int)
        prev_df['holo_y'] = self.trap_df['holo_y'].astype(int)
        for i, row in prev_df.iterrows():
            holo_x = row['holo_x']
            holo_y = row['holo_y']
            try:
                trap_df_i = (self.trap_df[(self.trap_df['holo_x']  == holo_x) & (self.trap_df['holo_y'] == holo_y)].index.tolist())
                self.trap_df.loc[trap_df_i,'img_x'] = row['img_x']
                self.trap_df.loc[trap_df_i,'img_y'] = row['img_y']
            except IndexError:
                pass
        self.save_trap_df()

    def take_calibration_image(self):
        for i in range(5):
            holo = self.calibration_blazing
            #holo = hg.apertures.circ(holo,self.circ_aper_center,self.circ_aper_radius)
            self.slm.apply_hologram(holo)
            time.sleep(0.5)
            image = self.cam.take_image()
            self.slm.apply_hologram(hg.blank())
            time.sleep(0.5)
            bgnd = self.cam.take_image()
            image.add_background(bgnd)
            pixel_sum = image.get_pixel_count()
            image.add_property('intensity_correction_iteration',self.intensity_correction_iteration)
            image.add_property('calibration_pixel_sum',pixel_sum)
            self.imager.save(image)
