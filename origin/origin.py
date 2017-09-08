"""
ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

This software has been developped by Carole Clastres under the supervision of
David Mary (Lagrange institute, University of Nice) and ported to python by
Laure Piqueras (CRAL). From November 2016 the software is updated by 
Antony Schutz.

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL). 
Please contact Carole for more info at carole.clastres@univ-lyon1.fr
Please contact Antony for more info at antonyschutz@gmail.com

origin.py contains an oriented-object interface to run the ORIGIN software
"""

from __future__ import absolute_import, division

from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os.path
import shutil
import sys
import yaml

from mpdaf.log import setup_logging, setup_logfile, clear_loggers
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.MUSE import FieldsMap, get_FSF_from_cube_keywords
from mpdaf.sdetect import Catalog
from mpdaf.tools import write_hdulist_to
from .lib_origin import Spatial_Segmentation, \
    Correlation_GLR_test, \
    Construct_Object_Catalogue, \
    dct_residual, \
    Compute_Standardized_data, \
    Compute_Segmentation_test, \
    Compute_GreedyPCA_area, \
    Compute_local_max_zone, \
    Create_local_max_cat, \
    Estimation_Line, \
    SpatioSpectral_Merging, \
    Segmentation, \
    Spatial_Merging_Circle, \
    Correlation_GLR_test_zone, O2test, \
    Compute_thresh_PCA_hist, \
    Compute_threshold_segmentation, \
    Purity_Estimation, \
    thresholdVsPFA_purity, \
    __version__

class ORIGIN(object):
    """ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

       This software has been developped by Carole Clastres under the
       supervision of David Mary (Lagrange institute, University of Nice).
       From November 2016 the software is updated by Antony Schutz. 

       The project is funded by the ERC MUSICOS (Roland Bacon, CRAL).
       Please contact Carole for more info at carole.clastres@univ-lyon1.fr
       Please contact Antony for more info at antonyschutz@gmail.com      

       An Origin object is mainly composed by:
        - cube data (raw data and covariance)
        - 1D dictionary of spectral profiles
        - MUSE PSF

       The class contains the methods that compose the ORIGIN software.

        Attributes
        ----------
        path               : string
                             Path where the ORIGIN data will be stored.
        name               : string
                             Name of the session and basename for the sources.
        param              : dict
                             Parameters values.
        cube_raw           : array (Nz, Ny, Nx)
                             Raw data.
        var                : array (Nz, Ny, Nx)
                             Variance.               
        Nx                 : integer
                             Number of columns
        Ny                 : integer
                             Number of rows
        Nz                 : int
                             Number of spectral channels
        wcs                : `mpdaf.obj.WCS`
                             RA-DEC coordinates.
        wave               : `mpdaf.obj.WaveCoord`
                             Spectral coordinates.
        NbAreas            : integer
                             Number of area (segmentation) for the PCA calculus
        profiles           : list of array
                             List of spectral profiles to test
        FWHM_profiles      : list
                             FWHM of the profiles in pixels.
        wfields            : None or list of arrays
                             List of weight maps (one per fields in the case
                             of MUSE mosaic)
                             None: just one field
        PSF                : array (Nz, Nfsf, Nfsf) or list of arrays
                             MUSE PSF (one per field)
        FWHM_PSF           : float or list of float
                             Mean of the fwhm of the PSF in pixel (one per
                             field).                
        setx               : array
                             Limits in pixels of the columns for each area.
        sety               : array
                             Limits in pixels of the rows for each area.                             
        cube_faint         : `~mpdaf.obj.Cube`
                             Projection on the eigenvectors associated to the
                             lower eigenvalues of the data cube (representing
                             the faint signal). Result of step03_compute_PCA.
        cube_correl        : `~mpdaf.obj.Cube`
                             Cube of T_GLR values. Result of
                             step04_compute_TGLR. From Max correlations
        cube_profile       : `~mpdaf.obj.Cube` (type int)
                             Number of the profile associated to the T_GLR.
                             Result of step04_compute_TGLR.
        cube_pval_correl: `~mpdaf.obj.Cube`
                             Cube of thresholded p-values associated to the
                             local maxima of T_GLR values. 
                             Result of step04_compute_TGLR.                                                          
        cube_correl_min    : `~mpdaf.obj.Cube`
                             Cube of T_GLR values. Result of
                             step04_compute_TGLR. From Min correlations                            
        cube_local_max     : `~mpdaf.obj.Cube`
                             Cube of Local maxima of T_GLR values. Result of                             
                             step04_compute_TGLR. From Max correlations
        cube_local_min     : `~mpdaf.obj.Cube`
                             Cube of Local maximam of T_GLR values. Result of
                             step04_compute_TGLR. From Min correlations                             

        Cat0               : astropy.Table
                             Catalog returned by step05_threshold_pval
        Cat1               : astropy.Table
                             Catalog returned by step06_compute_spectra.
        spectra            : list of `~mpdaf.obj.Spectrum`
                             Estimated lines. Result of step06_compute_spectra.
        continuum          : list of `~mpdaf.obj.Spectrum`
                             Roughly Estimated continuum. 
                             Result of step06_compute_spectra.                             
        Cat2               : astropy.Table
                             Catalog returned by step07_spatiospectral_merging.
    """
    
    def __init__(self, path, name, filename, profiles, PSF,
                 FWHM_PSF, cube_faint, mapO2, histO2, freqO2,
                 thresO2, cube_correl, maxmap, NbAreas, cube_profile, Cat0, 
                 Pval_r, index_pval, Det_M, Det_m, ThresholdPval,
                 Cat1, spectra, Cat2, param, cube_std, var, expmap,
                 cube_pval_correl, cube_local_max, cont_dct, segmentation_test,
                 segmentation_map_threshold, segmentation_map_spatspect,
                 cube_local_min, cube_correl_min, continuum, mapThresh, setx, 
                 sety, mapO2_full):
        #loggers
        setup_logging(name='origin', level=logging.DEBUG,
                           color=False,
                           fmt='%(name)s[%(levelname)s]: %(message)s',
                           stream=sys.stdout)
                           
        if os.path.exists('%s/%s/%s.log'%(path, name,name)):
            setup_logfile(name='origfile', level=logging.DEBUG,
                                       logfile='%s/%s/%s.log'%(path, name, name),
                                       fmt='%(asctime)s %(message)s')
        else:
            setup_logfile(name='origfile', level=logging.DEBUG,
                                       logfile='%s/%s.log'%(path, name),
                                       fmt='%(asctime)s %(message)s')                           
        self._log_stdout = logging.getLogger('origin')
        self._log_file = logging.getLogger('origfile')
        self._log_file.setLevel(logging.INFO)
                                       
        self._log_file.info('00 - Initialization ORIGIN v%s'%__version__)
        self._log_stdout.info('Step 00 - Initialization')
        self._log_stdout.info('Read the Data Cube')
        
        self.path = path
        self.name = name
        if param is None:
            self.param = {}
        else:
            self.param = param
        
        # MUSE data cube
        self.param['cubename'] = filename
        cub = Cube(filename)
        
        # Flux - set to 0 the Nan
        self.cube_raw = cub.data.filled(fill_value=0)
        # exposures map
        if expmap is None:
            self.expmap = (~cub.mask).astype(np.int)
        else:
            self.expmap = Cube(expmap)._data
        
        # variance - set to Inf the Nan
        if var is None:
            self.var = cub._var
            self.var[np.isnan(self.var)] = np.inf
        else:
            self.var = var
        # RA-DEC coordinates
        self.wcs = cub.wcs
        # spectral coordinates
        self.wave = cub.wave
        # Dimensions
        self.Nz, self.Ny, self.Nx = cub.shape
        
        # ORIGIN parameters
        if NbAreas is None:
            NbAreas = 0 
        self.param['nbareas'] = NbAreas
        self.NbAreas = NbAreas
        
        # List of spectral profile
        self.param['profiles'] = profiles
        if profiles is None:
            self._log_stdout.info('Load dictionary of spectral profile')
            DIR = os.path.dirname(__file__)
            profiles = DIR + '/Dico_FWHM_2_12.fits'
        self.profiles = []
        self.FWHM_profiles = []
        fprof = fits.open(profiles)
        for hdu in fprof[1:]:
            self.profiles.append(hdu.data)
            self.FWHM_profiles.append(hdu.header['FWHM'])
        fprof.close()
        
        #FSF
        # FSF cube(s)
        self._log_stdout.info('Load FSF')
        step_arcsec = self.wcs.get_step(unit=u.arcsec)[0]
        # map fileds in the case of MUSE mosaic
        self.wfields = None
        if PSF is None or FWHM_PSF is None:
            Nfsf=13
            if 'FSFMODE' in cub.primary_header:
                # FSF created from FSF*** keywords
                PSF, fwhm_pix, fwhm_arcsec = get_FSF_from_cube_keywords(cub,
                                                                        Nfsf)
                self.param['PSF'] = cub.primary_header['FSFMODE']
                nfields = cub.primary_header['NFIELDS']
                if nfields == 1: # just one FSF
                    # Normalization
                    self.PSF = PSF / np.sum(PSF, axis=(1, 2))[:, np.newaxis,
                                                                 np.newaxis]
                    # mean of the fwhm of the FSF in pixel
                    self.FWHM_PSF = np.mean(fwhm_pix)
                else: # mosaic: one FSF cube per field
                    self.PSF = []
                    self.FWHM_PSF = []
                    for i in range(nfields):
                        # Normalization 
                        self.PSF.append(PSF[i] / np.sum(PSF[i], axis=(1, 2))\
                                                    [:, np.newaxis,np.newaxis])
                        # mean of the fwhm of the FSF in pixel
                        self.FWHM_PSF.append(np.mean(fwhm_pix[i]))
                    fmap = FieldsMap(filename, extname='FIELDMAP')
                    # weighted field map
                    self.wfields = fmap.compute_weights()
            else:
                raise IOError('PSF are not described in the FITS header of the cube')

        else:
            self.param['PSF'] = PSF
            if type(PSF) is str:
                cubePSF = Cube(PSF)
                if cubePSF.shape[1] != cubePSF.shape[2]:
                    raise IOError('PSF must be a square image.')
                if not cubePSF.shape[1]%2:
                    raise IOError('The spatial size of the PSF must be odd.')
                if cubePSF.shape[0] != self.Nz:
                    raise IOError('PSF and data cube have not the same dimensions',
                                  ' along the spectral axis.')
                if not np.isclose(cubePSF.wcs.get_step(unit=u.arcsec)[0],
                                  step_arcsec):
                    raise IOError('PSF and data cube have not the same pixel ',
                                  'sizes.')
    
                self.PSF = cubePSF._data
            else:
                nfields = len(PSF)
                self.PSF = []
                for n in range(nfields):
                    self.PSF.append(Cube(PSF[i]))
                fmap = FieldsMap(filename, extname='FIELDMAP')
                # weighted field map
                self.wfields = fmap.compute_weights()
            # mean of the fwhm of the FSF in pixel
            self.FWHM_PSF = np.mean(FWHM_PSF)
            self.param['FWHM PSF'] = FWHM_PSF.tolist()
        
        del cub        
        
        # step1
        self.cube_std = cube_std
        self.cont_dct = cont_dct       
        self.segmentation_test = segmentation_test      
        # step 2 
        self.NbAreas = NbAreas
        self.sety = sety
        self.setx = setx          
        # step3
        self.cube_faint = cube_faint
        self.mapO2 = mapO2
        self.histO2 = histO2
        self.freqO2 = freqO2
        self.thresO2 = thresO2
        self.mapO2_full = mapO2_full
        # step4
        self.cube_correl = cube_correl
        self.cube_correl_min = cube_correl_min        
        self.cube_local_max = cube_local_max     
        self.cube_local_min = cube_local_min             
        self.cube_profile = cube_profile
        self.maxmap = maxmap
        # step5
        self.cube_pval_correl = cube_pval_correl
        self.segmentation_map_threshold = segmentation_map_threshold
        self.segmentation_map_spatspect = segmentation_map_spatspect        
        self.mapThresh = mapThresh        
        self.Cat0 = Cat0
        self.Pval_r = Pval_r
        self.index_pval = index_pval
        self.Det_M = Det_M
        self.Det_m = Det_m   
        self.ThresholdPval = ThresholdPval
        # step6
        self.Cat1 = Cat1
        self.spectra = spectra
        self.continuum = continuum
        # step7
        self.Cat2 = Cat2
        
        self._log_file.info('00 Done')
        
    @classmethod
    def init(cls, cube, profiles=None, PSF=None, FWHM_PSF=None, name='origin'):
        """Create a ORIGIN object.

        An Origin object is composed by:
        - cube data (raw data and covariance)
        - 1D dictionary of spectral profiles
        - MUSE PSF
        - parameters used to segment the cube in different zones.


        Parameters
        ----------
        cube        : string
                      Cube FITS file name
        profiles    : string
                      FITS of spectral profiles
                      If None, a default dictionary of 20 profiles is used.
        PSF         : string
                      Cube FITS filename containing a MUSE PSF per wavelength.
                      If None, PSF are computed with a Moffat function
                      (13x13 pixels, beta=2.6, fwhm1=0.76, fwhm2=0.66,
                      lambda1=4750, lambda2=7000)
        FWHM_PSF    : array (Nz)
                      FWHM of the PSFs in pixels.
        name        : str
                      Name of this session and basename for the sources.
        """
        return cls(path='.',  name=name, filename=cube, 
                   profiles=profiles, PSF=PSF, FWHM_PSF=FWHM_PSF, 
                   cube_faint=None, mapO2=None, histO2=None,
                   freqO2=None, thresO2=None, cube_correl=None, maxmap=None,
                   NbAreas=None, cube_profile=None, Cat0=None, 
                   Pval_r=None, index_pval=None, Det_M=None, Det_m=None,
                   ThresholdPval=None, Cat1=None, spectra=None, Cat2=None,
                   param=None, cube_std=None, var=None, expmap=None,
                   cube_pval_correl=None,cube_local_max=None,cont_dct=None,
                   segmentation_test=None, segmentation_map_threshold=None, 
                   segmentation_map_spatspect=None, cube_local_min=None,
                   cube_correl_min=None, continuum=None, mapThresh=None,
                   setx=None, sety=None, mapO2_full=None)
        
    @classmethod
    def load(cls, folder, newpath=None, newname=None):
        # sauver les PSFs ?
        """Load a previous session of ORIGIN
        
        Parameters
        ----------
        folder : string
                 path
        newpath : string
                  The session is loaded from the given folder but it will be
                  saved in a new folder under the new path.
        newname : string
                  The session is loaded from the given folder but it will be
                  saved in a new folder.
        """
        path = os.path.dirname(os.path.abspath(folder))
        name = os.path.basename(folder)
        
        stream = open('%s/%s.yaml'%(folder, name), 'r')
        param = yaml.load(stream)
        stream.close()
        
        if 'expmap' in param:
            expmap = param['expmap']
        else:
            expmap = None
            
        if 'FWHM PSF' in param:
            FWHM_PSF = np.asarray(param['FWHM PSF'])
        else:
            FWHM_PSF = None

        if os.path.isfile(param['PSF']):
            PSF = param['PSF']
        else:
            PSF_files = glob.glob('%s/cube_psf_*.fits'%folder)
            if len(PSF_files) == 0:
                PSF = None
            if len(PSF_files) == 1:
                PSF = PSF_files[0]
            else:
                PSF = sorted(PSF_files)

        NbAreas = param['nbareas']
        # step1
        if os.path.isfile('%s/cube_std.fits'%folder):
            cube_std = Cube('%s/cube_std.fits'%folder)
            var = cube_std._var
            cube_std._var = None
        else:
            cube_std = None
            var = None
        if os.path.isfile('%s/cont_dct.fits'%folder):
            cont_dct = Cube('%s/cont_dct.fits'%folder)
        else:
            cont_dct = None            
        if os.path.isfile('%s/segmentation_test.fits'%folder):
            segmentation_test = Cube('%s/segmentation_test.fits'%folder)
        else:
            segmentation_test = None                        
            
        # step2
        setx = None
        if os.path.isfile('%s/setx_0.txt'%folder):
            setx = []
            i = 0
            while(os.path.isfile('%s/setx_%d.txt'%(folder,i))):
                setx.append(np.loadtxt('%s/setx_%d.txt'%(folder,i)).astype(np.int))
                i = i + 1
        sety = None
        if os.path.isfile('%s/sety_0.txt'%folder):
            sety = []
            i = 0
            while(os.path.isfile('%s/sety_%d.txt'%(folder,i))):
                sety.append(np.loadtxt('%s/sety_%d.txt'%(folder,i)).astype(np.int))
                i = i + 1
        
        # step3
        if os.path.isfile('%s/cube_faint.fits'%folder):
            cube_faint = Cube('%s/cube_faint.fits'%folder)
        else:
            cube_faint = None
            
        if os.path.isfile('%s/mapO2_0.fits'%folder):
            mapO2 = []
            i = 0
            while(os.path.isfile('%s/mapO2_%d.fits'%(folder,i))):   
                mapO2.append(Image('%s/mapO2_%d.fits'%(folder,i)))
                i = i + 1                
        else:
            mapO2 = None
            
        if os.path.isfile('%s/histO2_0.txt'%folder):
            histO2 = []
            i = 0
            while(os.path.isfile('%s/histO2_%d.txt'%(folder,i))):                
                histO2.append(np.loadtxt('%s/histO2_%d.txt'%(folder, i)))
                i = i + 1
        else:
            histO2 = None
            
        if os.path.isfile('%s/freqO2_0.txt'%folder):
            freqO2 = []
            i = 0
            while(os.path.isfile('%s/freqO2_%d.txt'%(folder,i))):                
                freqO2.append(np.loadtxt('%s/freqO2_%d.txt'%(folder,i)))
                i = i + 1
        else:
            freqO2 = None
            
        if os.path.isfile('%s/thresO2.txt'%(folder)):
            thresO2 = np.loadtxt('%s/thresO2.txt'%(folder))
        else:
            thresO2 = None
            
        if os.path.isfile('%s/mapO2_full.fits'%folder):
            mapO2_full = Cube('%s/mapO2_full.fits'%folder)
        else:
            mapO2_full = None            
            
        # step4
        if os.path.isfile('%s/cube_correl.fits'%folder):
            cube_correl = Cube('%s/cube_correl.fits'%folder)
        else:
            cube_correl = None
        if os.path.isfile('%s/cube_correl_min.fits'%folder):
            cube_correl_min = Cube('%s/cube_correl_min.fits'%folder)
        else:
            cube_correl_min = None            
        if os.path.isfile('%s/cube_local_max.fits'%folder):
            cube_local_max = Cube('%s/cube_local_max.fits'%folder)
        else:
            cube_local_max = None     
        if os.path.isfile('%s/cube_local_min.fits'%folder):
            cube_local_min = Cube('%s/cube_local_min.fits'%folder)
        else:
            cube_local_min = None                                    
        if os.path.isfile('%s/maxmap.fits'%folder):
            maxmap = Image('%s/maxmap.fits'%folder)
        else:
            maxmap = None
        if os.path.isfile('%s/cube_profile.fits'%folder):
            cube_profile = Cube('%s/cube_profile.fits'%folder)
        else:
            cube_profile = None

        # step5
        if os.path.isfile('%s/cube_pval_correl.fits'%folder):
            cube_pval_correl = Cube('%s/cube_pval_correl.fits'%folder,
                                    mask=np.ma.nomask, dtype=np.float64)            
        else:
            cube_pval_correl = None
        if os.path.isfile('%s/Cat0.fits'%folder):
            Cat0 = Table.read('%s/Cat0.fits'%folder)
        else:
            Cat0 = None        
        if os.path.isfile('%s/Pval_r0.txt'%folder):
            Pval_r = []
            i = 0
            while(os.path.isfile('%s/Pval_r%d.txt'%(folder,i))):
                Pval_r.append(np.loadtxt('%s/Pval_r%d.txt'%(folder,i)).astype(np.float))
                i = i + 1
        else:
            Pval_r = None
        if os.path.isfile('%s/index_pval0.txt'%folder):
            index_pval = []
            i = 0
            while(os.path.isfile('%s/index_pval%d.txt'%(folder,i))):
                index_pval.append(np.loadtxt('%s/index_pval%d.txt'%(folder,i)).astype(np.float))  
                i = i + 1
        else:
            index_pval = None
        if os.path.isfile('%s/Det_M0.txt'%folder):
            Det_M = []
            i = 0
            while(os.path.isfile('%s/Det_M%d.txt'%(folder,i))):
                Det_M.append(np.loadtxt('%s/Det_M%d.txt'%(folder,i)).astype(np.int))
                i = i + 1
        else:
            Det_M = None
        if os.path.isfile('%s/Det_min0.txt'%folder):
            Det_m = []
            i = 0
            while(os.path.isfile('%s/Det_min%d.txt'%(folder,i))):
                Det_m.append(np.loadtxt('%s/Det_min%d.txt'%(folder,i)).astype(np.int))
                i = i + 1
        else:
            Det_m = None
        if os.path.isfile('%s/ThresholdPval.txt'%folder):
            ThresholdPval = np.loadtxt('%s/ThresholdPval.txt'%folder).astype(np.float)
        else:
            ThresholdPval = None
            
        # step6
        if os.path.isfile('%s/Cat1.fits'%folder):
            Cat1 = Table.read('%s/Cat1.fits'%folder)
        else:
            Cat1 = None
        if os.path.isfile('%s/spectra.fits'%folder):
            spectra = []
            fspectra = fits.open('%s/spectra.fits'%folder)
            for i in range(len(fspectra)//2):
                spectra.append(Spectrum('%s/spectra.fits'%folder,
                                        hdulist=fspectra,
                                        ext=('DATA%d'%i, 'STAT%d'%i)))
            fspectra.close()
        else:
            spectra = None
            
        if os.path.isfile('%s/continuum.fits'%folder):
            continuum = []
            fcontinuum = fits.open('%s/continuum.fits'%folder)
            for i in range(len(fcontinuum)//2):
                continuum.append(Spectrum('%s/continuum.fits'%folder,
                                        hdulist=fcontinuum,
                                        ext=('DATA%d'%i, 'STAT%d'%i)))
            fcontinuum.close()
        else:
            continuum = None            
        
        # step7
        if os.path.isfile('%s/Cat2.fits'%folder):
            Cat2 = Table.read('%s/Cat2.fits'%folder)
        else:
            Cat2 = None
        if os.path.isfile('%s/segmentation_map_threshold.fits'%folder):
            segmentation_map_threshold = Image('%s/segmentation_map_threshold.fits'%folder)
        else:
            segmentation_map_threshold = None
            
        if os.path.isfile('%s/segmentation_map_spatspect.fits'%folder):
            segmentation_map_spatspect = Image('%s/segmentation_map_spatspect.fits'%folder)
        else:
            segmentation_map_spatspect = None            
            
        if os.path.isfile('%s/mapThresh.fits'%folder):
            mapThresh = Image('%s/mapThresh.fits'%folder)
        else:
            mapThresh = None            
            
        if newpath is not None:
            path = newpath
        if newname is not None:
            name = newname
                
        return cls(path=path,  name=name, filename=param['cubename'],
                   profiles=param['profiles'], PSF=PSF, FWHM_PSF=FWHM_PSF,
                   cube_std=cube_std, var=var, mapO2_full=mapO2_full,
                   cube_faint=cube_faint, mapO2=mapO2, histO2=histO2,
                   freqO2=freqO2, thresO2=thresO2, cube_correl=cube_correl,
                   maxmap=maxmap, NbAreas=NbAreas, cube_profile=cube_profile, 
                   Cat0=Cat0, Pval_r=Pval_r,
                   index_pval=index_pval, Det_M=Det_M, Det_m=Det_m,
                   ThresholdPval= ThresholdPval, Cat1=Cat1, spectra=spectra,
                   Cat2=Cat2, param=param,expmap=expmap,
                   cube_pval_correl=cube_pval_correl,
                   cube_local_max=cube_local_max, cont_dct=cont_dct,
                   segmentation_test=segmentation_test,
                   segmentation_map_threshold=segmentation_map_threshold,
                   segmentation_map_spatspect=segmentation_map_spatspect,
                   cube_local_min=cube_local_min, 
                   cube_correl_min=cube_correl_min, continuum=continuum,
                   mapThresh=mapThresh, setx=setx, sety=sety)
                   
    def write(self, path=None, overwrite=False):
        """Save the current session in a folder
        
        Parameters
        ----------
        path      : string
                    Path where the ORIGIN data will be stored.
                    If None, the name of the session is used
        overwrite : bool
                    remove the folder if it exists
        """
        self._log_stdout.info('Writing...')
        # path
        if path is not None:
            self.path = path
        if not os.path.exists(self.path):
            raise IOError("Invalid path: {0}".format(self.path))

        path = os.path.abspath(self.path)

        path2 = path + '/' + self.name
        if not os.path.exists(path2):
            os.makedirs(path2)
        else:
            if overwrite:
                shutil.rmtree(path2)
                os.makedirs(path2)
        
        # parameters in .yaml
        stream = open('%s/%s.yaml'%(path2, self.name), 'w')
        yaml.dump(self.param, stream)
        stream.close()
        
        # log file
        currentlog = self._log_file.handlers[0].baseFilename
        newlog = os.path.abspath('%s/%s.log'%(path2, self.name))
        if (currentlog != newlog):
            clear_loggers('origfile')
            shutil.move(currentlog, newlog)
            setup_logfile(name='origfile', level=logging.DEBUG,
                                           logfile=newlog,
                                           fmt='%(asctime)s %(message)s')
            self._log_file = logging.getLogger('origfile')
            self._log_file.setLevel(logging.INFO)
            
        # PSF
        if type(self.PSF) is list:
            for i, psf in enumerate(self.PSF):
                Cube(data=psf, mask=np.ma.nomask).write('%s/cube_psf_%02d.fits'%(path2,i))
        else:
            Cube(data=self.PSF, mask=np.ma.nomask).write('%s/cube_psf.fits'%path2)
            
        #step1
        if self.cube_std is not None:
            if self.var is not None:
                self.cube_std._var = self.var
            self.cube_std.write('%s/cube_std.fits'%path2)
        if self.cont_dct is not None:
            self.cont_dct.write('%s/cont_dct.fits'%path2)
        if self.segmentation_test is not None:
            self.segmentation_test.write('%s/segmentation_test.fits'%path2)


        #step2
        if self.setx is not None:
            for i in range(self.NbAreas):
                np.savetxt('%s/setx_%d.txt'%(path2, i), self.setx[i])
        if self.sety is not None:
            for i in range(self.NbAreas):
                np.savetxt('%s/sety_%d.txt'%(path2, i), self.sety[i])           
                
        #step3
        if self.histO2 is not None:
            for i in range(self.NbAreas):
                np.savetxt('%s/histO2_%d.txt'%(path2, i), self.histO2[i])
        if self.freqO2 is not None:
            for i in range(self.NbAreas):
                np.savetxt('%s/freqO2_%d.txt'%(path2, i), self.freqO2[i])           
        if self.thresO2 is not None:
            np.savetxt('%s/thresO2.txt'%path2, self.thresO2)
        if self.mapO2 is not None:
            for i in range(self.NbAreas):
                self.mapO2[i].write('%s/mapO2_%d.fits'%(path2, i))
        if self.cube_faint is not None:
            self.cube_faint.write('%s/cube_faint.fits'%path2)
        if self.mapO2_full is not None:
            self.mapO2_full.write('%s/mapO2_full.fits'%path2)            

        # step4
        if self.cube_correl is not None:
            self.cube_correl.write('%s/cube_correl.fits'%path2)
        if self.cube_correl_min is not None:
            self.cube_correl_min.write('%s/cube_correl_min.fits'%path2)            
        if self.cube_profile is not None:
            self.cube_profile.write('%s/cube_profile.fits'%path2)
        if self.maxmap is not None:
            self.maxmap.write('%s/maxmap.fits'%path2)

        # step5
        if self.cube_local_max is not None:
            self.cube_local_max.write('%s/cube_local_max.fits'%path2)    
        if self.cube_local_min is not None:
            self.cube_local_min.write('%s/cube_local_min.fits'%path2)                                
        if self.cube_pval_correl is not None:
            self.cube_pval_correl.write('%s/cube_pval_correl.fits'%path2)
        if self.segmentation_map_threshold is not None:
            self.segmentation_map_threshold.write('%s/segmentation_map_threshold.fits'%path2)  
        if self.segmentation_map_spatspect is not None:
            self.segmentation_map_spatspect.write('%s/segmentation_map_spatspect.fits'%path2)              
        if self.mapThresh is not None:
            self.mapThresh.write('%s/mapThresh.fits'%path2)              
        if self.Cat0 is not None:
            self.Cat0.write('%s/Cat0.fits'%path2, overwrite=True)
        if self.Pval_r is not None:
            for i, pval in enumerate(self.Pval_r):
                np.savetxt('%s/Pval_r%d.txt'%(path2,i), pval)
        if self.index_pval is not None:
            for i, pval in enumerate(self.index_pval):
                np.savetxt('%s/index_pval%d.txt'%(path2,i), pval)
        if self.Det_M is not None:
            for i, det in enumerate(self.Det_M):
                np.savetxt('%s/Det_M%d.txt'%(path2,i), det)
        if self.Det_m is not None:
            for i, det in enumerate(self.Det_m):
                np.savetxt('%s/Det_min%d.txt'%(path2,i), det)
        if self.ThresholdPval is not None:
            np.savetxt('%s/ThresholdPval.txt'%path2, self.ThresholdPval)

        # step6
        if self.Cat1 is not None:
            self.Cat1.write('%s/Cat1.fits'%path2, overwrite=True)
        if self.spectra is not None:
            hdulist = fits.HDUList([fits.PrimaryHDU()])
            for i in range(len(self.spectra)):
                hdu = self.spectra[i].get_data_hdu(name='DATA%d'%i,
                                                   savemask='nan')
                hdulist.append(hdu)
                hdu = self.spectra[i].get_stat_hdu(name='STAT%d'%i)
                if hdu is not None:
                    hdulist.append(hdu)
            write_hdulist_to(hdulist, '%s/spectra.fits'%path2, overwrite=True)
            
        if self.continuum is not None:
            hdulist = fits.HDUList([fits.PrimaryHDU()])
            for i in range(len(self.continuum)):
                hdu = self.continuum[i].get_data_hdu(name='DATA%d'%i,
                                                   savemask='nan')
                hdulist.append(hdu)
                hdu = self.continuum[i].get_stat_hdu(name='STAT%d'%i)
                if hdu is not None:
                    hdulist.append(hdu)
            write_hdulist_to(hdulist, '%s/continuum.fits'%path2,overwrite=True)
            
        # step7
        if self.Cat2 is not None:
            self.Cat2.write('%s/Cat2.fits'%path2, overwrite=True)
      
           
        
    def step01_preprocessing(self, expmap=None, dct_order=10):
        """ Preprocessing of data, dct, standardization and noise compensation         
        
        Parameters
        ----------
        expmap      : string
                      Exposures map FITS file name
        dct_order   : integer
                      The number of atom to keep for the dct decomposition

        Returns
        -------
        self.var    : array
                        new cube of variance                 
        self.cube_std : `~mpdaf.obj.Cube`
                        standardized data for PCA
        self.cont_dct : `~mpdaf.obj.Cube`
                        DCT continuum
        """        
        self.param['dct_order'] = dct_order
        self._log_file.info('01 - Preprocessing, dct order=%d'%dct_order)

        self._log_stdout.info('01 - Preprocessing')

        newvar = False
        # exposures map
        if expmap is not None:
            _expmap = Cube(expmap)._data
            if not np.array_equal(self.cube_raw.shape, _expmap.shape):
                raise ValueError('cube and expmap with a different shape')
            _expmap[self.expmap==0] = 0                         
            self.expmap = _expmap
            self.param['expmap'] = expmap
            newvar = True
        
        self._log_stdout.info('Step 01 - DCT computation')
        self._log_stdout.info('reweighted data')
            
        weighted_cube_raw = self.cube_raw * np.sqrt(self.expmap)
            
        self._log_stdout.info('Compute the DCT residual')
        faint_dct, cont_dct = dct_residual(weighted_cube_raw, dct_order)
        
        
        # compute standardized data
        self._log_stdout.info('Standard data')
        cube_std, var = Compute_Standardized_data(faint_dct, self.expmap,
                                                  self.var, newvar)
        var[np.isnan(var)] = np.inf
        cont_dct = cont_dct / np.sqrt(var)
        
        # compute test for segmentation map 
        self._log_stdout.info('Segmentation test')
        segmentation_test = Compute_Segmentation_test(cont_dct)
        
        if newvar:        
            self._log_stdout.info('self.var is computed')   
            self.var = var
        
        self._log_stdout.info('Save the std signal in self.cube_std')        
        self.cube_std = Cube(data=cube_std, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)  
        self.cont_dct = Cube(data=cont_dct, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)
        self.segmentation_test = Cube(data=segmentation_test, wave=self.wave, 
                                      wcs=self.wcs, mask=np.ma.nomask)        
        self._log_file.info('01 Done')

# become step 02
    def step02_areas(self, area_option=None):
        
        self.inty, self.intx = Spatial_Segmentation(self.Nx, self.Ny, 2)        
        
        setx = []
        sety = []
        
        Setx1 = np.arange(self.intx[0],self.intx[1])
        Setx2 = np.arange(self.intx[1],self.intx[2])
        
        Sety2 = np.arange(self.inty[2],self.inty[1])
        Sety1 = np.arange(self.inty[1],self.inty[0])
        
        xset11 = np.repeat(Setx1[:,np.newaxis],len(Sety1),axis=1)
        xset12 = np.repeat(Setx2[:,np.newaxis],len(Sety1),axis=1)
        xset21 = np.repeat(Setx1[:,np.newaxis],len(Sety2),axis=1)
        xset22 = np.repeat(Setx2[:,np.newaxis],len(Sety2),axis=1)
        
        yset11 = np.repeat(Sety1[:,np.newaxis],len(Setx1),axis=1)
        yset12 = np.repeat(Sety1[:,np.newaxis],len(Setx2),axis=1)
        yset21 = np.repeat(Sety2[:,np.newaxis],len(Setx1),axis=1)
        yset22 = np.repeat(Sety2[:,np.newaxis],len(Setx2),axis=1)
                
        setx.append(np.ravel(xset11.T))
        setx.append(np.ravel(xset12.T))
        setx.append(np.ravel(xset21.T))
        setx.append(np.ravel(xset22.T))
        
        sety.append(np.ravel(yset11))
        sety.append(np.ravel(yset12))
        sety.append(np.ravel(yset21))
        sety.append(np.ravel(yset22)) 
        
        self.NbAreas = len(sety)
        self.param['nbareas'] = self.NbAreas
        self.sety = sety
        self.setx = setx
        
# become step 03 
    def step03_compute_greedy_PCA(self, mixing=False,
                              Noise_population=50, pfa_test=.01,
                              itermax=100, threshold_list=None):
        """ Loop on each zone of the data cube and compute the greedy PCA.
        The test (test_fun) and the threshold (threshold_test) define the part
        of the each zone of the cube to segment in nuisance and background. 
        A part of the background part (1/Noise_population %) is used to compute 
        a mean background, a signature. 
        The Nuisance part is orthogonalized to this signature in order to not 
        loose this part during the greedy process. SVD is performed on nuisance
        in order to modelized the nuisance part and the principal eigen vector, 
        only one, is used to perform the projection of the whole set of data:
        Nuisance and background. The Nuisance spectra which satisfied the test
        are updated in the background computation and the background is so 
        cleaned from sources signature. The iteration stop when all the spectra
        satisfy the criteria

        Parameters
        ----------
                    
        mixing              :   bool
                                if True the output of PCA is mixed with its
                                input according to the pvalue of a test based
                                on the continuum of the faint (output PCA)
        
        Noise_population    :   float                
                                the fraction of spectra used to estimate 
                                the background signature
                                
        pfa_test            :   float
                                the threshold of the test (default=0.01)  
                                
        itermax             :   integer
                                maximum iterations

        threshold_list      :   list
                                User given list of threshold (not pfa) to apply
                                on each area, the list is of lenght NbAreas
                                or of lenght 1. Before using this option 
                                make sure to have good correspondance between
                                the Areas and the threshold in list. 
                                Use: self.plot_areas() to be sure.

        Returns
        -------
        self.cube_faint : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the lower
                     eigenvalues of the data cube
                     (representing the faint signal)
        self.mapO2 : list(`~mpdaf.obj.Image`)
                     For each area, the numbers of iterations used by testO2
                     for each spaxel
        self.histO2 : list(array)
                      For each area, histogram
        self.freqO2 : list(array)
                      For each area, frequency
        self.thresO2 : list
                       For each area, Threshold value
        """
        self._log_stdout.info('03 - greedy PCA computation:')
        self._log_file.info('03 - greedy PCA computation')  
        
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
        if self.sety is None:
            raise IOError('Run the step 02 to initialize self.sety/setx ')
            
        self._log_file.info('   - Noise_population=%0.2f'%Noise_population)

        if threshold_list is None:
            self._log_file.info('   - pfa of the test=%0.2f'%pfa_test)            
            self.param['pfa_test'] = pfa_test   
            userlist=False
            pfa_test = np.repeat(pfa_test,self.NbAreas)
        else: 
            self._log_file.info('   - User given list of threshold')     
            userlist=True            
            pfa_test = threshold_list
            if len(pfa_test)==1:
                pfa_test = pfa_test*self.NbAreas
            self.param['threshold_list'] = threshold_list     
            
        self.param['Noise_population'] = Noise_population                
     
        self.param['itermax'] = itermax
        self._log_stdout.info('Step 03 - greedy PCA computation')                
        self._log_stdout.info('Compute greedy PCA on each zone')          
        
        faint, mapO2, self.histO2, self.freqO2, self.thresO2, mapO2_full = \
        Compute_GreedyPCA_area(self.NbAreas, self.cube_std._data,
                                  self.setx, self.sety, 
                                  Noise_population, pfa_test,itermax, userlist)
        if mixing:
            continuum = np.sum(faint,axis=0)**2 / faint.shape[0]
            pval = 1 - stats.chi2.cdf(continuum, 2) 
            faint = pval*faint + (1-pval)*self.cube_std._data 

        self._log_stdout.info('Save the faint signal in self.cube_faint')
        self.cube_faint = Cube(data=faint, wave=self.wave, wcs=self.wcs,
                          mask=np.ma.nomask)
        self._log_stdout.info('Save the numbers of iterations used by the' + \
                              ' testO2 for each spaxel in the dictionary' + \
                              ' self.mapO2') 

        self.mapO2 = []
        for area in range(self.NbAreas):
            self.mapO2.append( Image(data=mapO2[area], wcs=self.wcs) )   
        self.mapO2_full = Image(data=mapO2_full, wcs=self.wcs) 
            
        self._log_file.info('03 Done')              


# become 04
    def step04_compute_TGLR(self, NbSubcube=None, neighboors=26):
        """Compute the cube of GLR test values.
        The test is done on the cube containing the faint signal
        (self.cube_faint) and it uses the PSF and the spectral profile.
        The correlation can be computed per "area"  for low memory system. 
        Then a Loop on each zone of self.cube_correl is performed to
        compute for each zone:
        
        - The local maxima distribution of each zone
        - the p-values associated to the local maxima,
        - the p-values associated to the number of thresholded p-values
          of the correlations per spectral channel,
        - the final p-values which are the thresholded pvalues associated
          to the T_GLR values divided by twice the pvalues associated to the
          number of thresholded p-values of the correlations per spectral
          channel.

        Note
        ----
        area       :    If NbSubcube is given as a parameter 
                        the correlation and local maximas and minimas are 
                        performed on smaller subcube and combined after. 
                        Useful to avoid swapp             
          
        Parameters
        ----------                    
        NbSubcube   :   integer
                        Number of sub-cubes for the spatial segmentation
        neighboors  :   integer
                        Connectivity of contiguous voxels                                                       
                                
        Returns
        -------
        self.cube_correl  : `~mpdaf.obj.Cube`
                            Cube of T_GLR values
        self.cube_profile : `~mpdaf.obj.Cube` (type int)
                             Number of the profile associated to the T_GLR
        self.maxmap       : `~mpdaf.obj.Image`
                             Map of maxima along the wavelength axis
                             
        self.cube_local_max    : `~mpdaf.obj.Cube`
                                 Local maxima from max correlation
        self.cube_local_min    : `~mpdaf.obj.Cube`
                                 Local maxima from minus min correlation
        """
        self._log_file.info('04 GLR test')
        self._log_stdout.info('Step 04 - GLR test')
        if self.cube_faint is None:
            raise IOError('Run the step 02 to initialize self.cube_faint')
        self.param['neighboors'] = neighboors

        # TGLR computing (normalized correlations)           
        if 'expmap' in self.param: 
            var = self.var/self.expmap
        else:
            var = self.var
            var[self.expmap==0] = np.inf

        if NbSubcube is None:
            correl, profile, cm = Correlation_GLR_test(self.cube_faint._data, 
                                            var, self.PSF, self.wfields,
                                               self.profiles)  
            NbSubcube = 1
            inty, intx = Spatial_Segmentation(self.Nx, self.Ny, NbSubcube)
        else:              
            self._log_stdout.info('Spatial segmentation')
            self._log_file.info('Spatial segmentation')
            inty, intx = Spatial_Segmentation(self.Nx, self.Ny, NbSubcube)            
            self._log_stdout.info('segmented Correlation')
            self._log_file.info('segmented Correlation')            
            correl, profile, cm = Correlation_GLR_test_zone( \
                    self.cube_faint._data, var, self.PSF, self.wfields,
                    self.profiles, intx, inty, NbSubcube)  
                                                                         
        
        self._log_stdout.info('Save the TGLR value in self.cube_correl')
        
        mask = (self.expmap == 0)        
        
        correl[mask] = 0
        
        self.cube_correl = Cube(data=correl, wave=self.wave, wcs=self.wcs,
                      mask=np.ma.nomask)
        self.cube_correl_min = Cube(data=cm, wave=self.wave, wcs=self.wcs,
                      mask=np.ma.nomask)        
        self._log_stdout.info('Save the number of profile associated to the TGLR in self.cube_profile')
        
        profile[mask] = 0       
        self.cube_profile = Cube(data=profile, wave=self.wave, wcs=self.wcs,
                       mask=np.ma.nomask, dtype=int)
        
        self._log_stdout.info('Save the map of maxima in self.maxmap')              
        carte_2D_correl = np.amax(self.cube_correl._data, axis=0)
        self.maxmap = Image(data=carte_2D_correl, wcs=self.wcs)               
                       
        self._log_file.info('04 - Correlation done')

        self._log_stdout.info('Step 04 - Local maximum and p-values computation')
        self._log_stdout.info('Compute p-values of local maximum of correlation values')

        cube_local_max, cube_local_min = Compute_local_max_zone(
                                                    self.cube_correl._data,
                                                    self.cube_correl_min._data,
                                                    self.expmap==0,
                                                    intx, inty, NbSubcube,
                                                    neighboors)
        self._log_stdout.info('Save self.cube_local_max from max correlations')
        self.cube_local_max = Cube(data=cube_local_max, \
                                     wave=self.wave,
                                     wcs=self.wcs, mask=np.ma.nomask)      
        self._log_stdout.info('Save self.cube_local_min from min correlations')        
        self.cube_local_min = Cube(data=cube_local_min, \
                                     wave=self.wave,
                                     wcs=self.wcs, mask=np.ma.nomask)                                 
        
        self._log_file.info('04 Done')  

    def step05_threshold_pval(self, purity=.9, threshold_option=None, pfa=0.15):        
        """Threshold the Pvalue with the given threshold, if the threshold is
        None the threshold is automaticaly computed from confidence applied
        on local maximam from maximum correlation and local maxima from 
        minus minimum correlation

        Parameters
        ----------
        purity : float
                 fidelity to automatically compute the threshold        
        threshold_option : float, 'background' or None
                           float -> it is a manual threshold.
                           string 'background' -> threshold based on background
                           threshold
                           None -> estimated
        pfa              : float
                           Pvalue for the test which performs segmentation
                            
        Returns
        -------                               
        self.cube_pval_correl  : `~mpdaf.obj.Cube`
                                 Cube of thresholded p-values associated
                                 to the local max of T_GLR values
        self.Cat0 : astropy.Table
                    Catalogue of the referent voxels for each group.
                    Columns: x y z ra dec lbda T_GLR profile pvalC
                    Coordinates are in pixels.
        """
        self._log_stdout.info('Step 05 - p-values Thresholding')
        self._log_file.info('Step 05 - p-values Thresholding')   
        
        self._log_stdout.info('Threshold the Pvalues')
        if threshold_option is None:
            self._log_file.info('   computation of threshold with purity =%.1f'%purity)
        elif threshold_option == 'background' :
            self._log_file.info('   computation of threshold with purity =%.1f (background option)'%purity)
        else: 
            self._log_file.info('   threshold =%.1f '%threshold_option)            
        self.param['purity'] = purity
        self.param['threshold_option'] = threshold_option
        if self.cube_local_max is None:
            raise IOError('Run the step 04 to initialize self.cube_local_max and self.cube_local_min')


        self.ThresholdPval, self.Pval_r, self.index_pval, \
        cube_pval_correl, mapThresh, segmap, self.Det_M, self.Det_m \
                                         = Compute_threshold_segmentation(
                                           purity, 
                                           self.cube_local_max.data,
                                           self.cube_local_min.data,
                                           threshold_option, 
                                           self.segmentation_test.data, pfa)
        self._log_stdout.info('Threshold: %.1f (background) %.1f (sources)'%(self.ThresholdPval[0], self.ThresholdPval[1]))
        self._log_stdout.info('Save the threshold map in self.mapThresh')                                          
        self.mapThresh = Image(data=mapThresh, wcs=self.wcs, mask=np.ma.nomask)                
            
        
        self.param['pfa'] = pfa
        
        self._log_stdout.info('Save self.cube_pval_correl')
        self.cube_pval_correl = Cube(data=cube_pval_correl, \
                                     wave=self.wave,
                                     wcs=self.wcs, mask=np.ma.nomask)      
        
        self.Cat0 = Create_local_max_cat(self.cube_correl._data,
                                         self.cube_profile._data,
                                         self.cube_pval_correl._data,
                                         self.wcs, self.wave)
        self._log_stdout.info('Save a first version of the catalogue of ' + \
                              'emission lines in self.Cat0 (%d lines)' \
                              %(len(self.Cat0))) 
        
        self.segmentation_map_threshold = Image(data=segmap,
                                    wcs=self.wcs, mask=np.ma.nomask)
        self._log_stdout.info('Save the segmentation map for threshold in self.segmentation_map_threshold')          
        
        self._log_file.info('05 Done')
        
    def step06_compute_spectra(self, grid_dxy=0):
        """compute the estimated emission line and the optimal coordinates
        for each detected lines in a spatio-spectral grid (each emission line
        is estimated with the deconvolution model :
        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))
        Via PCA LS or denoised PCA LS Method

        Parameters
        ----------
        grid_dxy   : integer
                     Maximum spatial shift for the grid

        Returns
        -------
        self.Cat1    : astropy.Table
                       Catalogue of parameters of detected emission lines.
                       Columns: x y z ra dec lbda T_GLR profile pvalC
                                residual flux num_line
        self.spectra : list of `~mpdaf.obj.Spectrum`
                       Estimated lines
        self.continuum : list of `~mpdaf.obj.Spectrum`
                       Roughly estimated continuum
        """
        self._log_file.info('06 Lines estimation grid_dxy=%d' %(grid_dxy))
        self._log_stdout.info('Step 06 - Lines estimation')
        self.param['grid_dxy'] = grid_dxy

        if self.Cat0 is None:
            raise IOError('Run the step 05 to initialize self.Cat0 catalogs')
        if self.cube_std.var is None:
            var = self.var
        else:
            var = self.cube_std.var * self.expmap
            
        self.Cat1, Cat_est_line_raw_T, Cat_est_line_var_T, Cat_est_cnt_T = \
        Estimation_Line(self.Cat0, self.cube_raw, var, self.PSF, \
                     self.wfields, self.wcs, self.wave, size_grid = grid_dxy, \
                     criteria = 'flux', order_dct = 30, horiz_psf = 1, \
                     horiz = 5)
            
        self._log_stdout.info('Step 06 - estimate fidelity')    
        # 0 for background and 1 for sources; to know which self.index_pval 
        # is correponding to the pixel (y,x)
        bck_or_src = self.mapThresh.data == self.ThresholdPval[0]
        self.Cat1 = Purity_Estimation(self.Cat1, self.cube_correl.data, 
                                        self.Pval_r, self.index_pval, 
                                        bck_or_src)
                   
        
        self._log_stdout.info('Save the updated catalogue in self.Cat1 (%d lines)'%len(self.Cat1))
        self.spectra = [] 
   
        for data, vari in zip(Cat_est_line_raw_T, Cat_est_line_var_T): 
            spe = Spectrum(data=data, var=vari, wave=self.wave,
                           mask=np.ma.nomask)
            self.spectra.append(spe)
        self._log_stdout.info('Save the estimated spectrum of each line in self.spectra')
            
        self.continuum = []                  
        for data, vari in zip(Cat_est_cnt_T, Cat_est_line_var_T): 
            cnt = Spectrum(data=data, var=vari, wave=self.wave,
                           mask=np.ma.nomask)
            self.continuum.append(cnt)         
            
        self._log_stdout.info('Save the estimated continuum of each line in self.continuum, CAUTION: rough estimate!')
        self._log_file.info('06 Done')       

    def step07_spatiospectral_merging(self, deltaz=20, pfa=0.05):
        """Construct a catalogue of sources by spatial merging of the
        detected emission lines in a circle with a diameter equal to
        the mean over the wavelengths of the FWHM of the FSF.
        Then, merge the detected emission lines distants in an estimated source 
        area.

        Parameters
        ----------
        deltaz : integer
                 Distance maximum between 2 different lines
        pfa    : float
                 Pvalue for the test which performs segmentation                 

        Returns
        -------
        self.segmentation_map : `~mpdaf.obj.Image`
                                 Segmentation map
        self.Cat2 : astropy.Table
                    Catalogue
                    Columns: ID x_circle y_circle ra_circle dec_circle
                    x_centroid y_centroid ra_centroid dec_centroid nb_lines x y
                    z ra dec lbda T_GLR profile pvalC residual flux
                    num_line
        """
        self._log_file.info('07 spatio spectral merging deltaz=%d'%deltaz)
        self._log_stdout.info('Step 07 - Spectral merging')
        if self.wfields is None:
            fwhm = self.FWHM_PSF
        else:
            fwhm = np.max(np.array(self.FWHM_PSF)) # to be improved
        self.param['deltaz'] = deltaz
        if self.Cat1 is None:
            raise IOError('Run the step 06 to initialize self.Cat1')
        cat = Spatial_Merging_Circle(self.Cat1, fwhm, self.wcs)
        self.Cat2, segmap = SpatioSpectral_Merging(cat, pfa,
                                           self.segmentation_test.data, \
                                           self.cube_correl.data, \
                                           self.var, deltaz)
        self.segmentation_map_spatspect = Image(data=segmap,
                                    wcs=self.wcs, mask=np.ma.nomask)
        self._log_stdout.info('Save the segmentation map for spatio-spectral merging in self.segmentation_map_spatspect')  
        
        self._log_stdout.info('Save the updated catalogue in self.Cat2 (%d objects, %d lines)'%(np.unique(self.Cat2['ID']).shape[0], len(self.Cat2)))
        self._log_file.info('07 Done')

    def step08_write_sources(self, path=None, overwrite=True,
                             fmt='default', src_vers='0.1',
                             author='undef', ncpu=1):
        """add corresponding RA/DEC to each referent pixel of each group and
        write the final sources.


        Parameters
        ----------
        name : str
            Basename for the sources.
        path : str
            path where the sources will be saved.
        overwrite : bool
            Overwrite the folder if it already exists
        fmt : str, 'working' or 'default'
            Format of the catalog. The format differs for the LINES table.

        Returns
        -------
        CatF : mpdaf.sdetect.Catalog
               Final catalog
                  
        Each Source object O consists of:
            - O.header: pyfits header instance that contains all parameters
                        used during the ORIGIN detection process
            - O.lines: astropy table that contains the parameters of spectral
                       lines.
            - O.spectra: Dictionary that contains spectra. It contains for each
                         line, the estimated spectrum (LINE_**), the estimated
                         continuum (CONT_**) and the correlation (CORR_**).
            - O.images: Dictionary that contains images: the white image
                        (MUSE_WHITE), the map of maxima along the wavelength
                        axis (MAXMAP), the segmentation map (SEG_ORIG) and
                        narrow band images (NB_LINE_** and NB_CORR_**) 
            - O.cubes: Dictionary that contains the small data cube around the
                       source (MUSE-CUBE)
        """
        self._log_file.info('08 Sources creation')
        # Add RA-DEC to the catalogue
        self._log_stdout.info('Step 08 - Sources creation')
        self._log_stdout.info('Add RA-DEC to the catalogue')
        if self.Cat2 is None:
            raise IOError('Run the step 07 to initialize self.Cat2')

        # path
        if path is not None and not os.path.exists(path):
            raise IOError("Invalid path: {0}".format(path))
            
        if path is None:
            path_src = '%s/%s/sources'%(self.path, self.name)
            catname = '%s/%s/%s.fits'%(self.path, self.name, self.name)
        else:
            path = os.path.normpath(path)
            path_src = '%s/%s/sources'%(path, self.name)
            catname = '%s/%s/%s.fits'%(path, self.name, self.name)
           
        if not os.path.exists(path_src):
            os.makedirs(path_src)
        else:
            if overwrite:
                shutil.rmtree(path_src)
                os.makedirs(path_src)

        # list of source objects
        self._log_stdout.info('Create the list of sources')
        if self.cube_correl is None:
            raise IOError('Run the step 04 to initialize self.cube_correl')
        if self.spectra is None:
            raise IOError('Run the step 06 to initialize self.spectra')
        nsources = Construct_Object_Catalogue(self.Cat2, self.spectra,
                                              self.cube_correl,
                                              self.wave, self.FWHM_profiles,
                                              path_src, self.name, self.param,
                                              src_vers, author,
                                              self.path, self.maxmap,
                                              self.segmentation_map_threshold, 
#                                              self.segmentation_map_spatspect,                                               
                                              self.continuum,
                                              self.ThresholdPval, ncpu)                                            
                                              
        # create the final catalog
        self._log_stdout.info('Create the final catalog- %d sources'%nsources)
        catF = Catalog.from_path(path_src, fmt='working')
        catF.write(catname, overwrite=overwrite)
                      
        self._log_file.info('08 Done')

        return catF

    def plot_areas(self, ax=None):
        """ Plot the 2D segmentation for PCA from self.step02_areas()
            on the test used to perform this segmentation
        
        Parameters
        ----------
        ax  : matplotlib.Axes
              The Axes instance in which the image is drawn
        """
        if ax is None:
            ax = plt.gca()
        test = self.segmentation_test.data
        tmp = np.zeros(test.shape)
        
        meanx=[]
        meany=[]        
        for n in range(self.NbAreas):
            tmp[self.sety[n],self.setx[n]] = n+1
            
            meanx.append( np.mean(self.setx[n]) )
            meany.append( np.mean(self.sety[n]) )            
        
        ax.imshow(test, origin='lower', cmap='jet', interpolation='nearest')
        ax.imshow(tmp, origin='lower', cmap='jet', interpolation='nearest',alpha=.7)
        for n in range(self.NbAreas):
            ax.text(meanx[n],meany[n],str(n),color='w',fontweight='bold')
        ax.set_title('continuum test with areas')        
        
    def plot_PCA_threshold_before(self, i, pfa_test=.01, ax=None, 
                           log10=True, threshold_list=None):
        """ Plot the histogram and the threshold for the starting point of the 
        PCA, this version of the plot is to do before doing the PCA
        
        Parameters
        ----------
        i: integer in [0, NbAreas[           
        pfa_test :   float
                     the pfa of the test (default=.01) 
                            
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
            
        if self.NbAreas is None:
            raise IOError('Run the step 02 to initialize self.NbAreas')            
            
        if ax is None:
            ax = plt.gca()

        if threshold_list is None:
            userlist=False
            pfa_test = np.repeat(pfa_test,self.NbAreas)
        else: 
            userlist=True            
            pfa_test = threshold_list
            if len(pfa_test)==1:
                pfa_test = pfa_test*self.NbAreas    
                
        # Data in this spatio-spectral area
        test = O2test(self.cube_std.data[:, self.sety[i], self.setx[i]])
        
        # automatic threshold computation     
        hist, bins, thre = Compute_thresh_PCA_hist(test, pfa_test[i])    
        if userlist:
            thre = pfa_test[i]
        
        ind = np.argmax(hist)
        mod = bins[ind]
        ind2 = np.argmin(( hist[ind]/2 - hist[:ind] )**2)
        fwhm = mod - bins[ind2]
        sigma = fwhm/np.sqrt(2*np.log(2))
        
        center = (bins[:-1] + bins[1:]) / 2
        gauss = stats.norm.pdf(center,loc=mod,scale=sigma)

        if log10:
            hist = np.log10(hist)
            gauss = np.log10(gauss)
            
        ax.plot(center, hist,'-k')
        ax.plot(center, hist,'.r')
        ym,yM = ax.get_ylim()
        ax.plot(center, gauss,'-b',alpha=.5)
        ax.plot([thre,thre],[ym,yM],'b',lw=2,alpha=.5)
        ax.grid()
        ax.set_xlim((center.min(),center.max()))
        ax.set_ylim((ym,yM))
        ax.set_title('zone %d - threshold %f' %(i,thre))  
        
    def plot_PCA_threshold(self, i, ax=None, log10=True):
        """ Plot the histogram and the threshold for the starting point of the 
        PCA, this version of the plot is to do before doing the PCA
        
        Parameters
        ----------
        i: integer in [0, NbAreas[           
        pfa_test :   float
                     the pfa of the test (default=.01) 
                            
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')         
            
        if ax is None:
            ax = plt.gca()

        bins = self.freqO2[i]
        hist = self.histO2[i]
        thre = self.thresO2[i]
        
        ind = np.argmax(hist)
        mod = bins[ind]
        ind2 = np.argmin(( hist[ind]/2 - hist[:ind] )**2)
        fwhm = mod - bins[ind2]
        sigma = fwhm/np.sqrt(2*np.log(2))

        center = (bins[:-1] + bins[1:]) / 2
        gauss = stats.norm.pdf(center,loc=mod,scale=sigma)

        if log10:
            hist = np.log10(hist)
            gauss = np.log10(gauss)

        ax.plot(center, hist,'-k')
        ax.plot(center, hist,'.r')
        ym,yM = ax.get_ylim()
        ax.plot(center, gauss,'-b',alpha=.5)
        ax.plot([thre,thre],[ym,yM],'b',lw=2,alpha=.5)
        ax.grid()
        ax.set_xlim((center.min(),center.max()))
        ax.set_ylim((ym,yM))
        ax.set_title('zone %d - threshold %f' %(i,thre))        
        
    def plot_mapPCA(self, area=None, ax=None, iteration=None):
        """ Plot the histogram and the threshold for the starting point of the PCA
        
        Parameters
        ----------
        area: integer in [0, NbAreas[
                if None draw the full map for all areas
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        iteration : Display the nuisance/bacground pixels at itartion k
        """

        if self.mapO2 is None:
            raise IOError('Run the step 03 to initialize self.mapO2')

        if area is None:
            themap = self.mapO2_full
            title = 'Full map'
        else:
            themap = self.mapO2
            title = 'zone %d' %area
            
        if ax is None:
            ax = plt.gca()
    
        if iteration is None:
            mapO2 = themap[area].data
        else:
            mapO2 = themap[area].data>iteration
            
        cax = ax.imshow(mapO2,origin='lower',cmap='jet',interpolation='nearest')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax, cax=cax2)
        ax.set_title(title)        
        
    def plot_segmentation(self, pfa=5e-2, ax=None):
        """ Plot the 2D segmentation map associated to a PFA
        This function draw the segmentation map which is used, not with the 
        same pfa, in self.step05_threshold_pval() to compute the threshold 
        of the local maxima of correlations and in 
        self.step07_spatiospectral_merging() to merge the detected lines.
        
        Parameters
        ----------
        pfa : float
              Pvalue for the test which performs segmentation
        ax  : matplotlib.Axes
              The Axes instance in which the image is drawn
        """
        if self.cont_dct is None:
            raise IOError('Run the step 01 to initialize self.cont_dct')        
            
        if ax is None:
            ax = plt.gca()
            
        map_in = Segmentation(self.cont_dct.data, pfa)            
        
        ax.imshow(map_in, origin='lower', cmap='jet', interpolation='nearest')
        ax.set_title('Labels of segmentation, pfa: %f' %(pfa))

    def plot_thresholdVsPFA_step05(self, purity=.9, 
                                   pfaset=np.linspace(1e-3,0.5,41), ax=None):
        """Draw threshold of local maxima as a function of the segmentation
        map using PFA to create source/background mask of step05.
        
        Parameters
        ----------
        purity : the purity for wich the function is plotted
        pfaset : the list of PFA to test
        ax : matplotlib.Axes
             The Axes instance in which the image is drawn
        """       
        
        cube_local_max = self.cube_local_max.data
        cube_local_min = self.cube_local_min.data
        test = self.segmentation_test.data
        
        threshold = \
        thresholdVsPFA_purity(test,cube_local_max,cube_local_min,purity,pfaset)
            
        if ax is None:
            ax = plt.gca()  
        
        ax.plot(pfaset,threshold)      
        ax.set_xlabel('PFA')
        ax.set_ylabel('Threshold')        
        ax.set_title('Purity %f' %purity)
        
    def plot_purity(self, i, ax=None, log10=True):
        """Draw number of sources per threshold computed in step04
        
        Parameters
        ----------
        i  : integer
             Pvalue for the test which performs segmentation
             i = 0 : background
             i = 1 : source
        ax : matplotlib.Axes
             The Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """
                
        if i == 0: 
            i_titre = 'background'
        else: 
            i_titre = 'sources'
            
        if self.cube_correl is None:
            raise IOError('Run the step 04 to initialize self.cube_correl')
            
        if ax is None:
            ax = plt.gca()        
        
        threshold = self.ThresholdPval[i]
        Pval_r = self.Pval_r[i]
        index_pval = self.index_pval[i]
        purity = self.param['purity']
        Det_M = self.Det_M[i]
        Det_m = self.Det_m[i]
        
        ax2 = ax.twinx()
        if log10:
            ax2.semilogy(index_pval, Pval_r, 'y.-', label = 'purity' )
            ax.semilogy( index_pval, Det_M, 'b.-', label = 'n detections (+DATA)' )
            ax.semilogy( index_pval, Det_m, 'g.-', label = 'n detections (-DATA)' )
            ax2.semilogy(threshold, purity,'xr') 
            
        else:
            ax2.plot(index_pval, Pval_r, 'y.-', label = 'purity' )
            ax.plot( index_pval, Det_M, 'b.-', label = 'n detections (+DATA)' )
            ax.plot( index_pval, Det_m, 'g.-', label = 'n detections (-DATA)' )
            ax2.plot(threshold, purity,'xr') 
            
        ym,yM = ax.get_ylim()
        ax.plot([threshold,threshold],[ym,yM],'r', alpha=.25, lw=2 , \
                 label='automatic threshold' )
               
        ax.set_ylim((ym,yM))
        ax.set_xlabel('Threshold')
        ax2.set_ylabel('Purity')
        ax.set_ylabel('Number of detections')
        ax.set_title('%s - threshold %f' %(i_titre,threshold))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc=2)                      
        
    def plot_NB(self, i, ax1=None, ax2=None, ax3=None):
        """Plot the narrow bands images
        
        i : integer
            index of the object in self.Cat0
        ax1 : matplotlib.Axes
              The Axes instance in which the NB image
              around the source is drawn
        ax2 : matplotlib.Axes
              The Axes instance in which a other NB image for check is drawn
        ax3 : matplotlib.Axes
              The Axes instance in which the difference is drawn
        """
        if self.Cat0 is None:
            raise IOError('Run the step 04 to initialize self.Cat0')
            
        if ax1 is None and ax2 is None and ax3 is None:
            ax1 = plt.subplot(1,3,1)
            ax2 = plt.subplot(1,3,2)
            ax3 = plt.subplot(1,3,3)
            
        # Coordinates of the source
        x0 = self.Cat0[i]['x']
        y0 = self.Cat0[i]['y']
        z0 = self.Cat0[i]['z']
        # Larger spatial ranges for the plots
        longxy0 = 20
        y01 = max(0, y0 - longxy0)
        y02 = min(self.cube_raw.shape[1], y0 + longxy0 + 1)
        x01 = max(0, x0 - longxy0)
        x02 = min(self.cube_raw.shape[2], x0 + longxy0 + 1)
        # Coordinates in this window
        y00 = y0 - y01
        x00 = x0 - x01
        # spectral profile
        num_prof = self.Cat0[i]['profile']
        profil0 = self.profiles[num_prof]
        # length of the spectral profile
        profil1 = profil0[profil0 > 1e-13]
        long0 = profil1.shape[0]
        # half-length of the spectral profile
        longz = long0 // 2
        # spectral range
        intz1 = max(0, z0 - longz)
        intz2 = min(self.cube_raw.shape[0], z0 + longz + 1)
        # subcube for the plot
        cube_test_plot = self.cube_raw[intz1:intz2, y01:y02, x01:x02]
        wcs = self.wcs[y01:y02, x01:x02]
        # controle cube
        nb_ranges = 3
        if (z0 + longz + nb_ranges * long0) < self.cube_raw.shape[0]:
            intz1c = intz1 + nb_ranges * long0
            intz2c = intz2 + nb_ranges * long0
        else:
            intz1c = intz1 - nb_ranges * long0
            intz2c = intz2 - nb_ranges * long0
        cube_controle_plot = self.cube_raw[intz1c:intz2c, y01:y02, x01:x02]
        # (1/sqrt(2)) * difference of the 2 sububes
        diff_cube_plot = (1. / np.sqrt(2)) * (cube_test_plot - cube_controle_plot)
        
        if ax1 is not None:
            ax1.plot(x00, y00, 'm+')
            ima_test_plot = Image(data=cube_test_plot.sum(axis=0), wcs=wcs)
            title = 'cube test - (%d,%d)\n' % (x0, y0) + \
                    'lambda=%d int=[%d,%d[' % (z0, intz1, intz2)
            ima_test_plot.plot(colorbar='v', title=title, ax=ax1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

        if ax2 is not None:
            ax2.plot(x00, y00, 'm+')
            ima_controle_plot = Image(data=cube_controle_plot.sum(axis=0), wcs=wcs)
            title = 'check - (%d,%d)\n' % (x0, y0) + \
                        'int=[%d,%d[' % (intz1c, intz2c)
            ima_controle_plot.plot(colorbar='v', title=title, ax=ax2)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

        if ax3 is not None:
            ax3.plot(x00, y00, 'm+')
            ima_diff_plot = Image(data=diff_cube_plot.sum(axis=0), wcs=wcs)
            title = 'Difference narrow band - (%d,%d)\n' % (x0, y0) + \
                    'int=[%d,%d[' % (intz1c, intz2c)
            ima_diff_plot.plot(colorbar='v', title=title, ax=ax3)
            ax3.get_xaxis().set_visible(False)
            ax3.get_yaxis().set_visible(False)
    

    def plot_sources(self, x, y, circle=False, vmin=0, vmax=30, title=None,
                     ax=None, **kwargs):
        """Plot detected emission lines on the 2D map of maximum of the T_GLR
        values over the spectral channels.

        Parameters
        ----------
        x      : array
                 Coordinates along the x-axis of the estimated lines
                 in pixels (column).
        y      : array
                 Coordinates along the y-axis of the estimated lines
                 in pixels (column).
        circle  : bool
                  If true, plot circles with a diameter equal to the
                  mean of the fwhm of the PSF.
        vmin : float
                Minimum pixel value to use for the scaling.
        vmax : float
                Maximum pixel value to use for the scaling.
        title : str
                An optional title for the figure (None by default).
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        kwargs : matplotlib.artist.Artist
                 Optional extra keyword/value arguments to be passed to
                 the ``ax.imshow()`` function. 
        """
        if self.cube_correl is None:
            raise IOError('Run the step 02 to initialize self.cube_correl')
        if self.wfields is None:
            fwhm = self.FWHM_PSF
        else:
            fwhm = np.max(np.array(self.FWHM_PSF))

        if ax is None:
            ax = plt.gca()

        ax.plot(x, y, 'k+')
        if circle:
            for px, py in zip(x, y):
                c = plt.Circle((px, py), np.round(fwhm / 2), color='k',
                               fill=False)
                ax.add_artist(c)
        self.maxmap.plot(vmin=vmin, vmax=vmax, title=title, ax=ax, **kwargs)
        
    def info(self):
        """ plot information
        """
        currentlog = self._log_file.handlers[0].baseFilename
        with open(currentlog) as f:
            for line in f:
                if line.find('Done') == -1:
                    self._log_stdout.info(line)
        