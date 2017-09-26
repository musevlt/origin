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
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import stats
import os.path
import shutil
import sys
import warnings
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
    area_segmentation_square_fusion, \
    area_segmentation_sources_fusion, \
    area_segmentation_convex_fusion, \
    area_growing, \
    area_segmentation_final, \
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
        areamap            : `~mpdaf.obj.Image`
                             PCA area
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
    
    def __init__(self, path, name, filename, profiles, PSF, FWHM_PSF,
                 cube_faint, mapO2, thresO2, cube_correl, maxmap, NbAreas,
                 cube_profile, Cat0, Pval_r, index_pval, Det_M, Det_m,
                 ThresholdPval, Cat1, spectra, Cat2, param, cube_std, var,
                 cube_pval_correl, cube_local_max, cont_dct,
                 segmentation_test, segmentation_map_threshold,
                 segmentation_map_spatspect, cube_local_min, continuum,
                 mapThresh, areamap):
        #loggers
        setup_logging(name='origin', level=logging.DEBUG,
                           color=False,
                           fmt='%(name)s[%(levelname)s]: %(message)s',
                           stream=sys.stdout)
                           
        if os.path.exists('%s/%s/%s.log'%(path, name,name)):
            setup_logfile(name='origfile', level=logging.DEBUG,
                                       logfile='%s/%s/%s.log'%(path, name,
                                                               name),
                                       fmt='%(asctime)s %(message)s')
        else:
            setup_logfile(name='origfile', level=logging.DEBUG,
                                       logfile='%s/%s.log'%(path, name),
                                       fmt='%(asctime)s %(message)s')                           
        self._log_stdout = logging.getLogger('origin')
        self._log_file = logging.getLogger('origfile')
        self._log_file.setLevel(logging.INFO)
                                       
        # log
        self._loginfo('Step 00 - Initialization (ORIGIN v%s)'%__version__)
        
        # init
        self.path = path
        self.name = name
        if param is None:
            self.param = {}
        else:
            self.param = param
        
        # MUSE data cube
        self._loginfo('Read the Data Cube %s'%filename)
        self.param['cubename'] = filename
        cub = Cube(filename)
        
        # Flux - set to 0 the Nan
        self.cube_raw = cub.data.filled(fill_value=0)
        self.mask = cub._mask
        
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
            DIR = os.path.dirname(__file__)
            profiles = DIR + '/Dico_FWHM_2_12.fits'
        self._loginfo('Load dictionary of spectral profile %s'%profiles)
        self.profiles = []
        self.FWHM_profiles = []
        fprof = fits.open(profiles)
        for hdu in fprof[1:]:
            self.profiles.append(hdu.data)
            self.FWHM_profiles.append(hdu.header['FWHM'])
        fprof.close()
        
        #FSF
        # FSF cube(s)
        step_arcsec = self.wcs.get_step(unit=u.arcsec)[0]
        # map fileds in the case of MUSE mosaic
        self.wfields = None
        if PSF is None or FWHM_PSF is None:
            self._loginfo('Compute FSFs from the datacube FITS header' + \
                                                                    'keywords')
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
                    self._loginfo('mean FWHM of the FSFs = %.2f pixels'\
                                                                %self.FWHM_PSF)
                else: # mosaic: one FSF cube per field
                    self.PSF = []
                    self.FWHM_PSF = []
                    for i in range(nfields):
                        # Normalization 
                        self.PSF.append(PSF[i] / np.sum(PSF[i], axis=(1, 2))\
                                                    [:, np.newaxis,np.newaxis])
                        # mean of the fwhm of the FSF in pixel
                        fwhm = np.mean(fwhm_pix[i])
                        self.FWHM_PSF.append(fwhm)
                        self._loginfo('mean FWHM of the FSFs' + \
                        ' (field %d) = %.2f pixels'%(i, fwhm))
                    fmap = FieldsMap(filename, extname='FIELDMAP')
                    # weighted field map
                    self.wfields = fmap.compute_weights()
            else:
                raise IOError('PSF are not described in the FITS header' + \
                                                                 'of the cube')

        else:
            if type(PSF) is str:
                self._loginfo('Load FSFs from %s'%PSF)
                self.param['PSF'] = PSF
                
                cubePSF = Cube(PSF)
                if cubePSF.shape[1] != cubePSF.shape[2]:
                    raise IOError('PSF must be a square image.')
                if not cubePSF.shape[1]%2:
                    raise IOError('The spatial size of the PSF must be odd.')
                if cubePSF.shape[0] != self.Nz:
                    raise IOError('PSF and data cube have not the same' + \
                                         'dimensions along the spectral axis.')
                if not np.isclose(cubePSF.wcs.get_step(unit=u.arcsec)[0],
                                  step_arcsec):
                    raise IOError('PSF and data cube have not the same pixel ',
                                  'sizes.')
    
                self.PSF = cubePSF._data
            else:
                nfields = len(PSF)
                self.PSF = []
                for n in range(nfields):
                    self._loginfo('Load FSF from %s'%PSF[i])
                    self.PSF.append(Cube(PSF[i]))
                fmap = FieldsMap(filename, extname='FIELDMAP')
                # weighted field map
                self.wfields = fmap.compute_weights()
            # mean of the fwhm of the FSF in pixel
            self.FWHM_PSF = np.mean(FWHM_PSF)
            self.param['FWHM PSF'] = FWHM_PSF.tolist()
            self._loginfo('mean FWHM of the FSFs = %.2f pixels'%self.FWHM_PSF)
        
        del cub        
        
        # step1
        self.cube_std = cube_std
        self.cont_dct = cont_dct       
        self.segmentation_test = segmentation_test      
        # step 2 
        self.NbAreas = NbAreas
        self.areamap = areamap
        # step3
        self.cube_faint = cube_faint
        self.thresO2 = thresO2
        self.mapO2 = mapO2
        # step4
        self.cube_correl = cube_correl
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
        
        self._loginfo('00 Done')
        
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
                   cube_faint=None, mapO2=None, thresO2=None, cube_correl=None,
                   maxmap=None, NbAreas=None, cube_profile=None, Cat0=None, 
                   Pval_r=None, index_pval=None, Det_M=None, Det_m=None,
                   ThresholdPval=None, Cat1=None, spectra=None, Cat2=None,
                   param=None, cube_std=None, var=None,
                   cube_pval_correl=None,cube_local_max=None,cont_dct=None,
                   segmentation_test=None, segmentation_map_threshold=None, 
                   segmentation_map_spatspect=None, cube_local_min=None,
                   continuum=None, mapThresh=None, areamap=None)
        
    @classmethod
    def load(cls, folder, newpath=None, newname=None):
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
            segmentation_test = Image('%s/segmentation_test.fits'%folder)
        else:
            segmentation_test = None                        
            
        # step2
        if os.path.isfile('%s/areamap.fits'%folder):
            areamap = Image('%s/areamap.fits'%folder, dtype=np.int)
        else:
            areamap = None
        
        # step3
        if os.path.isfile('%s/cube_faint.fits'%folder):
            cube_faint = Cube('%s/cube_faint.fits'%folder)
        else:
            cube_faint = None
            
        if os.path.isfile('%s/thresO2.txt'%(folder)):
            thresO2 = np.loadtxt('%s/thresO2.txt'%(folder), ndmin=1)
            thresO2 = thresO2.tolist()
        else:
            thresO2 = None
            
        if os.path.isfile('%s/mapO2.fits'%folder):
            mapO2 = Image('%s/mapO2.fits'%folder)
        else:
            mapO2 = None            
            
        # step4
        if os.path.isfile('%s/cube_correl.fits'%folder):
            cube_correl = Cube('%s/cube_correl.fits'%folder)
        else:
            cube_correl = None
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
                Pval_r.append(np.loadtxt('%s/Pval_r%d.txt'%(folder,i))\
                .astype(np.float))
                i = i + 1
        else:
            Pval_r = None
        if os.path.isfile('%s/index_pval0.txt'%folder):
            index_pval = []
            i = 0
            while(os.path.isfile('%s/index_pval%d.txt'%(folder,i))):
                index_pval.append(np.loadtxt('%s/index_pval%d.txt'%(folder,i))\
                .astype(np.float))  
                i = i + 1
        else:
            index_pval = None
        if os.path.isfile('%s/Det_M0.txt'%folder):
            Det_M = []
            i = 0
            while(os.path.isfile('%s/Det_M%d.txt'%(folder,i))):
                Det_M.append(np.loadtxt('%s/Det_M%d.txt'%(folder,i))\
                .astype(np.int))
                i = i + 1
        else:
            Det_M = None
        if os.path.isfile('%s/Det_min0.txt'%folder):
            Det_m = []
            i = 0
            while(os.path.isfile('%s/Det_min%d.txt'%(folder,i))):
                Det_m.append(np.loadtxt('%s/Det_min%d.txt'%(folder,i)).\
                astype(np.int))
                i = i + 1
        else:
            Det_m = None
        if os.path.isfile('%s/ThresholdPval.txt'%folder):
            ThresholdPval = np.loadtxt('%s/ThresholdPval.txt'%folder).\
            astype(np.float)
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
            segmentation_map_threshold = Image('%s'%folder + \
            '/segmentation_map_threshold.fits')
        else:
            segmentation_map_threshold = None
            
        if os.path.isfile('%s/segmentation_map_spatspect.fits'%folder):
            segmentation_map_spatspect = Image('%s'%folder +  \
            '/segmentation_map_spatspect.fits')
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
                   cube_std=cube_std, var=var,
                   cube_faint=cube_faint, mapO2=mapO2, thresO2=thresO2,
                   cube_correl=cube_correl,
                   maxmap=maxmap, NbAreas=NbAreas, cube_profile=cube_profile, 
                   Cat0=Cat0, Pval_r=Pval_r,
                   index_pval=index_pval, Det_M=Det_M, Det_m=Det_m,
                   ThresholdPval= ThresholdPval, Cat1=Cat1, spectra=spectra,
                   Cat2=Cat2, param=param,
                   cube_pval_correl=cube_pval_correl,
                   cube_local_max=cube_local_max, cont_dct=cont_dct,
                   segmentation_test=segmentation_test,
                   segmentation_map_threshold=segmentation_map_threshold,
                   segmentation_map_spatspect=segmentation_map_spatspect,
                   cube_local_min=cube_local_min, continuum=continuum,
                   mapThresh=mapThresh, areamap = areamap)
                   
    def _loginfo(self, logstr):
        self._log_file.info(logstr) 
        self._log_stdout.info(logstr)
                   
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
        self._loginfo('Writing...')
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
                Cube(data=psf, mask=np.ma.nomask).write('%s'%path2 + \
                '/cube_psf_%02d.fits'%i)
        else:
            Cube(data=self.PSF, mask=np.ma.nomask).write('%s'%path2 + \
            '/cube_psf.fits')
            
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
        if self.areamap is not None:
            self.areamap.write('%s/areamap.fits'%path2)
                
        #step3
        if self.thresO2 is not None:
            np.savetxt('%s/thresO2.txt'%path2, self.thresO2)
        if self.cube_faint is not None:
            self.cube_faint.write('%s/cube_faint.fits'%path2)
        if self.mapO2 is not None:
            self.mapO2.write('%s/mapO2.fits'%path2)            

        # step4
        if self.cube_correl is not None:
            self.cube_correl.write('%s/cube_correl.fits'%path2)
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
            self.segmentation_map_threshold.write('%s'%path2 + \
            '/segmentation_map_threshold.fits')  
        if self.segmentation_map_spatspect is not None:
            self.segmentation_map_spatspect.write('%s'%path2 + \
            '/segmentation_map_spatspect.fits')              
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
        
        self._loginfo("Current session saved in %s"%path2)
      
           
        
    def step01_preprocessing(self, dct_order=10):
        """ Preprocessing of data, dct, standardization and noise compensation         
        
        Parameters
        ----------
        dct_order   : integer
                      The number of atom to keep for the dct decomposition

        Returns
        -------
        self.cube_std          : `~mpdaf.obj.Cube`
                                 standardized data for PCA
        self.cont_dct          : `~mpdaf.obj.Cube`
                                 DCT continuum
        self.segmentation_test : `~mpdaf.obj.Image`
                                 2D map where sources and background are
                                 separated
        """
        self._loginfo('Step 01 - Preprocessing, dct order=%d'%dct_order)
            
        self._loginfo('DCT computation')
        self.param['dct_order'] = dct_order
        faint_dct, cont_dct = dct_residual(self.cube_raw, dct_order)
         
        # compute standardized data
        self._loginfo('Data standardizing')
        cube_std  = Compute_Standardized_data(faint_dct, self.mask, self.var)
        cont_dct = cont_dct / np.sqrt(self.var)
        
        # compute test for segmentation map 
        self._loginfo('Segmentation test')
        segmentation_test = Compute_Segmentation_test(cont_dct)
        
        self._loginfo('Std signal saved in self.cube_std')        
        self.cube_std = Cube(data=cube_std, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)  
        self._loginfo('DCT continuum saved in self.cont_dct')
        self.cont_dct = Cube(data=cont_dct, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)
        self._loginfo('Segmentation map saved in self.segmentation_test')
        self.segmentation_test = Image(data=segmentation_test, 
                                      wcs=self.wcs, mask=np.ma.nomask)        
        self._loginfo('01 Done')

    def step02_areas(self,  pfa=.2, size=120, minsize=None):
        """ Creation of automatic area         
        
        Parameters
        ----------
        pfa      :  float
                    PFA of the segmentation test to estimates sources with
                    strong continuum
        size   :    int
                    Lenght in pixel of the side of typical square wanted                        
                        enough big area to satisfy the PCA
        minsize :   int
                      Minimum size in pixels^2 for a label
                      if label is less than minsize the label is merged with  
                      the first found one ---- To Improve 
                            

        Returns
        -------
     
        self.NbAreas    :   int
                            number of areas                    
        self.areamap : `~mpdaf.obj.Image`
                       The map of areas
        """           
        self._loginfo('02 - Areas creation')
        
        if self.segmentation_test is None:
            raise IOError('Run the step 01 to initialize ' + \
            'self.segmentation_test')
        
        self._loginfo('   - pfa of the test = %0.2f'%pfa)
        self._loginfo('   - side size = %d pixels'%size)
        if minsize is None:
            self._loginfo('   - minimum size = None')
        else:
            self._loginfo('   - minimum size = %d pixels**2'%minsize)
        
        self.param['pfa_areas'] = pfa
        self.param['size_areas'] = size
        self.param['minsize_areas'] = minsize
        
        nexpmap = (np.sum(~self.mask, axis=0) >0).astype(np.int)
        
        NbSubcube = 1 + int( self.Nx*self.Ny / (size**2) )
        
        if NbSubcube > 1:
        
            self._loginfo('First segmentation of %d^2 square'%NbSubcube)
            self._loginfo('Squares segmentation and fusion') 
            square_cut_fus = area_segmentation_square_fusion(nexpmap, \
                                                NbSubcube, self.Ny, self.Nx)
            self._loginfo('Sources fusion')         
            square_src_fus, src = \
            area_segmentation_sources_fusion(self.segmentation_test.data, \
                                             square_cut_fus, pfa, \
                                             self.Ny, self.Nx)        
            self._loginfo('Convex envelope')                 
            convex_lab = area_segmentation_convex_fusion(square_src_fus,src)
            self._loginfo('Areas dilation')                 
            Grown_label = area_growing(convex_lab, nexpmap)        
            
            if minsize is None:
                minsize = int(self.Ny*self.Nx/(NbSubcube**2))
            areamap = area_segmentation_final(Grown_label, minsize)
        elif NbSubcube == 1:
            areamap = nexpmap
            
        self._loginfo('Save the map of areas in self.areamap') 

        self.areamap = Image(data=areamap, wcs=self.wcs, dtype=np.int)
            
        labels = np.unique(areamap)
        if 0 in labels: #expmap=0
            self.NbAreas = len(labels) - 1
        else:
            self.NbAreas = len(labels)
        self._loginfo('%d areas generated'%self.NbAreas)        
        self.param['nbareas'] = self.NbAreas
        
        self._loginfo('02 Done') 
        
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
                                If True the output of PCA is mixed with its
                                input according to the pvalue of a test based
                                on the continuum of the faint (output PCA)
        
        Noise_population    :   float                
                                Fraction of spectra used to estimate 
                                the background signature
                                
        pfa_test            :   float
                                Threshold of the test (default=0.01)  
                                
        itermax             :   integer
                                Maximum number of iterations

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
        self.mapO2 : `~mpdaf.obj.Image`
                     For each area, the numbers of iterations used by testO2
                     for each spaxel
        self.thresO2 : list
                       For each area, threshold value
        """
        self._loginfo('Step 03 - Greedy PCA computation')  
        
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
        if self.areamap is None:
            raise IOError('Run the step 02 to initialize self.areamap ')
            
        self._loginfo('   - Noise_population = %0.2f'%Noise_population)

        if threshold_list is None:
            self._loginfo('   - pfa of the test = %0.2f'%pfa_test)            
            self.param['pfa_test'] = pfa_test   
        else: 
            self._loginfo('   - User given list of threshold = ' + \
            " ".join(str(x) for x in threshold_list))     
            self.param['threshold_list'] = threshold_list 
            
        self._loginfo('   - mixing = %d'%mixing)
        self._loginfo('   - Max number of iterations = %d'%itermax)
            
        self.param['Noise_population'] = Noise_population                
        self.param['itermax'] = itermax
        self.param['mixing'] = mixing
        
        self._loginfo('Compute greedy PCA on each zone')          
        faint, mapO2, self.thresO2 = \
        Compute_GreedyPCA_area(self.NbAreas, self.cube_std._data,
                               self.areamap._data, Noise_population,
                               threshold_list, pfa_test, itermax)
        if mixing:
            continuum = np.sum(faint,axis=0)**2 / faint.shape[0]
            pval = 1 - stats.chi2.cdf(continuum, 2) 
            faint = pval*faint + (1-pval)*self.cube_std._data 

        self._loginfo('Save the faint signal in self.cube_faint')
        self.cube_faint = Cube(data=faint, wave=self.wave, wcs=self.wcs,
                          mask=np.ma.nomask)
        self._loginfo('Save the numbers of iterations used by the' + \
                              ' testO2 for each spaxel in self.mapO2') 

        self.mapO2 = Image(data=mapO2, wcs=self.wcs) 
            
        self._loginfo('03 Done')              


    def step04_compute_TGLR(self, NbSubcube=1, neighboors=26):
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
          
        Parameters
        ----------                    
        NbSubcube   :   integer
                        Number of sub-cubes for the spatial segmentation
                        If NbSubcube>1 the correlation and local maximas and
                        minimas are performed on smaller subcube and combined
                        after. Useful to avoid swapp
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
        self._loginfo('Step 04 - GLR test(NbSubcube=%d'%NbSubcube +\
        ', neighboors=%d)'%neighboors)
        
        if self.cube_faint is None:
            raise IOError('Run the step 03 to initialize self.cube_faint')
            
        self.param['neighboors'] = neighboors
        self.param['NbSubcube'] = NbSubcube

        # TGLR computing (normalized correlations)           
        self._loginfo('Correlation')
        inty, intx = Spatial_Segmentation(self.Nx, self.Ny, NbSubcube)
        if NbSubcube == 1:
            correl, profile, cm = Correlation_GLR_test(self.cube_faint._data, 
                                            self.var, self.PSF, self.wfields,
                                               self.profiles)  
        else:              
            correl, profile, cm = Correlation_GLR_test_zone( \
                    self.cube_faint._data, self.var, self.PSF, self.wfields,
                    self.profiles, intx, inty, NbSubcube)  
                                                                         
        
        self._loginfo('Save the TGLR value in self.cube_correl')
        correl[self.mask] = 0
        self.cube_correl = Cube(data=correl, wave=self.wave, wcs=self.wcs,
                      mask=np.ma.nomask)
                      
        self._loginfo('Save the number of profile associated to the TGLR in self.cube_profile')
        profile[self.mask] = 0       
        self.cube_profile = Cube(data=profile, wave=self.wave, wcs=self.wcs,
                       mask=np.ma.nomask, dtype=int)
        
        self._loginfo('Save the map of maxima in self.maxmap')              
        carte_2D_correl = np.amax(self.cube_correl._data, axis=0)
        self.maxmap = Image(data=carte_2D_correl, wcs=self.wcs)               
                       
        self._loginfo('Compute p-values of local maximum of correlation values')
        cube_local_max, cube_local_min = Compute_local_max_zone(correl, cm,
                                                    self.mask,
                                                    intx, inty, NbSubcube,
                                                    neighboors)
        self._loginfo('Save self.cube_local_max from max correlations')
        self.cube_local_max = Cube(data=cube_local_max, \
                                     wave=self.wave,
                                     wcs=self.wcs, mask=np.ma.nomask)      
        self._loginfo('Save self.cube_local_min from min correlations')        
        self.cube_local_min = Cube(data=cube_local_min, \
                                     wave=self.wave,
                                     wcs=self.wcs, mask=np.ma.nomask)                                 
        
        self._loginfo('04 Done')  

    def step05_threshold_pval(self, purity=.9, threshold_option=None, pfa=0.15):        
        """Threshold the Pvalue with the given threshold, if the threshold is
        None the threshold is automaticaly computed from confidence applied
        on local maximam from maximum correlation and local maxima from 
        minus minimum correlation

        Parameters
        ----------
        purity : float
                 purity to automatically compute the threshold        
        threshold_option : float, 'background' or None
                           float -> it is a manual threshold.
                           string 'background' -> threshold based on background
                           threshold
                           None -> estimated
        pfa              : float
                           Pvalue for the test which performs segmentation
                            
        Returns
        -------
        self.ThresholdPval     : [float, float]
                                 Background and source threshold values  
        self.mapThresh         : `~mpdaf.obj.Image`
                                 Threshold map
        self.cube_pval_correl  : `~mpdaf.obj.Cube`
                                 Cube of thresholded p-values associated
                                 to the local max of T_GLR values
        self.Cat0 : astropy.Table
                    Catalogue of the referent voxels for each group.
                    Columns: x y z ra dec lbda T_GLR profile pvalC
                    Coordinates are in pixels.
        self.segmentation_map_threshold : `~mpdaf.obj.Image`
                                          Segmentation map for threshold
        """
        self._loginfo('Step 05 - p-values Thresholding')  
        
        if self.cube_local_max is None:
            raise IOError('Run the step 04 to initialize ' + \
            'self.cube_local_max and self.cube_local_min')
        
        if threshold_option is None:
            self._loginfo('Estimation of threshold with purity = %.1f'%purity)
        elif threshold_option == 'background' :
            self._loginfo('Computation of threshold (based on background)' + \
            ' with purity = %.1f (background option)'%purity)
        else: 
            self._loginfo('Threshold = %.1f '%threshold_option)
            
        self._loginfo('PFA = %.2f '%pfa)
            
        self.param['purity'] = purity
        self.param['threshold_option'] = threshold_option
        self.param['pfa'] = pfa

        self.ThresholdPval, self.Pval_r, self.index_pval, \
        cube_pval_correl, mapThresh, segmap, self.Det_M, self.Det_m \
                                         = Compute_threshold_segmentation(
                                           purity, 
                                           self.cube_local_max.data,
                                           self.cube_local_min.data,
                                           threshold_option, 
                                           self.segmentation_test.data, pfa)
        self._loginfo('Threshold: %.1f (background)'%self.ThresholdPval[0] + \
        ' %.1f (sources)'%self.ThresholdPval[1])
        self._loginfo('Save the threshold map in self.mapThresh')                                          
        self.mapThresh = Image(data=mapThresh, wcs=self.wcs, mask=np.ma.nomask)                
        
        self._loginfo('Save the thresholded p-values associated to the ' + \
        'local max of T_GLR values in self.cube_pval_correl')
        self.cube_pval_correl = Cube(data=cube_pval_correl, \
                                     wave=self.wave,
                                     wcs=self.wcs, mask=np.ma.nomask)      
        
        self.Cat0 = Create_local_max_cat(self.cube_correl._data,
                                         self.cube_profile._data,
                                         self.cube_pval_correl._data,
                                         self.wcs, self.wave)
        self._loginfo('Save a first version of the catalogue of ' + \
                              'emission lines in self.Cat0 (%d lines)' \
                              %(len(self.Cat0))) 
        
        self.segmentation_map_threshold = Image(data=segmap,
                                    wcs=self.wcs, mask=np.ma.nomask)
        self._loginfo('Save the segmentation map for threshold in ' + \
        'self.segmentation_map_threshold')          
        
        self._loginfo('05 Done')
        
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
                                residual flux num_line purity
        self.spectra : list of `~mpdaf.obj.Spectrum`
                       Estimated lines
        self.continuum : list of `~mpdaf.obj.Spectrum`
                       Roughly estimated continuum
        """
        self._loginfo('Step06 - Lines estimation (grid_dxy=%d)' %(grid_dxy))
        self.param['grid_dxy'] = grid_dxy

        if self.Cat0 is None:
            raise IOError('Run the step 05 to initialize self.Cat0 catalogs')
            
        self.Cat1, Cat_est_line_raw_T, Cat_est_line_var_T, Cat_est_cnt_T = \
        Estimation_Line(self.Cat0, self.cube_raw, self.var, self.PSF, \
                     self.wfields, self.wcs, self.wave, size_grid = grid_dxy, \
                     criteria = 'flux', order_dct = 30, horiz_psf = 1, \
                     horiz = 5)
            
        self._loginfo('Purity estimation')    
        # 0 for background and 1 for sources; to know which self.index_pval 
        # is correponding to the pixel (y,x)
        bck_or_src = self.mapThresh.data == self.ThresholdPval[0]
        self.Cat1 = Purity_Estimation(self.Cat1, self.cube_correl.data, 
                                        self.Pval_r, self.index_pval, 
                                        bck_or_src)
                   
        
        self._loginfo('Save the updated catalogue in self.Cat1' + \
        ' (%d lines)'%len(self.Cat1))
        
        self.spectra = [] 
        for data, vari in zip(Cat_est_line_raw_T, Cat_est_line_var_T): 
            spe = Spectrum(data=data, var=vari, wave=self.wave,
                           mask=np.ma.nomask)
            self.spectra.append(spe)
        self._loginfo('Save the estimated spectrum of each line in ' + \
        'self.spectra')
            
        self.continuum = []                  
        for data, vari in zip(Cat_est_cnt_T, Cat_est_line_var_T): 
            cnt = Spectrum(data=data, var=vari, wave=self.wave,
                           mask=np.ma.nomask)
            self.continuum.append(cnt)          
        self._loginfo('Save the estimated continuum of each line in ' + \
        'self.continuum, CAUTION: rough estimate!')
        
        self._loginfo('06 Done')       

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
        self.Cat2 : astropy.Table
                    Catalogue
                    Columns: ID x_circle y_circle ra_circle dec_circle
                    x_centroid y_centroid ra_centroid dec_centroid nb_lines x y
                    z ra dec lbda T_GLR profile pvalC residual flux
                    num_line purity ID_old seg_label
        self.segmentation_map_spatspect : `~mpdaf.obj.Image`
                                          Segmentation map
        """
        self._loginfo('Step07 Spatio spectral merging ' + \
        '(deltaz=%d, pfa=%d)'%(deltaz, pfa))
        if self.wfields is None:
            fwhm = self.FWHM_PSF
        else:
            fwhm = np.max(np.array(self.FWHM_PSF)) # to be improved
        self.param['deltaz'] = deltaz
        self.param['pfa_merging'] = pfa
        
        if self.Cat1 is None:
            raise IOError('Run the step 06 to initialize self.Cat1')

        cat = Spatial_Merging_Circle(self.Cat1, fwhm, self.wcs)
        self.Cat2, segmap = SpatioSpectral_Merging(cat, pfa,
                                           self.segmentation_test.data, \
                                           self.cube_correl.data, \
                                           self.var, deltaz)
        self.segmentation_map_spatspect = Image(data=segmap,
                                    wcs=self.wcs, mask=np.ma.nomask)
        self._loginfo('Save the segmentation map for spatio-spectral ' + \
        'merging in self.segmentation_map_spatspect')  
        
        self._loginfo('Save the updated catalogue in self.Cat2 ' + \
        '(%d objects, %d lines)'%(np.unique(self.Cat2['ID']).shape[0],
          len(self.Cat2)))
        self._loginfo('07 Done')

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
        # Add RA-DEC to the catalogue
        self._loginfo('Step 08 - Sources creation')
        self._loginfo('Add RA-DEC to the catalogue')
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
        self._loginfo('Create the list of sources')
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
                                              self.segmentation_map_spatspect,                                               
                                              self.continuum,
                                              self.ThresholdPval, ncpu)                                            
                                              
        # create the final catalog
        self._loginfo('Create the final catalog- %d sources'%nsources)
        catF = Catalog.from_path(path_src, fmt='working')
        catF.write(catname, overwrite=overwrite)
                      
        self._loginfo('08 Done')

        return catF

    def plot_areas(self, ax=None, **kwargs):
        """ Plot the 2D segmentation for PCA from self.step02_areas()
            on the test used to perform this segmentation
        
        Parameters
        ----------
        ax  : matplotlib.Axes
              The Axes instance in which the image is drawn
        kwargs : matplotlib.artist.Artist
                 Optional extra keyword/value arguments to be passed to
                 the ``ax.imshow()`` function.
        """
        if ax is None:
            ax = plt.gca()       
        
        self.segmentation_test.plot(ax=ax)
        
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'jet'
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.7
        if 'interpolation' not in kwargs:
            kwargs['interpolation'] = 'nearest'
        kwargs['origin'] = 'lower'
        
        cax = ax.imshow(self.areamap._data, **kwargs)
        
        i0 = np.min(self.areamap._data)
        i1 = np.max(self.areamap._data)
        if i0 != i1:
            n = i1 - i0 + 1
            bounds = np.linspace(i0, i1+1, n+1) - 0.5
            norm = BoundaryNorm(bounds, n+1)
            divider = make_axes_locatable(ax)
            cax2 = divider.append_axes("right", size="5%", pad=1)
            plt.colorbar(cax, cax=cax2, cmap=kwargs['cmap'], norm=norm,
                         spacing='proportional', ticks=bounds+0.5,
                         boundaries=bounds, format='%1i')
            ax.set_title('continuum test with areas')
        else:
            ax.set_title('continuum test (1 area)')
    
    def plot_step03(self, log10=True, fig=None, **fig_kw):
        """ Plot the histogram and the threshold for the starting point of the 
        PCA, this version of the plot is to do before doing the PCA
        
        Parameters
        ----------
        log10     : bool
                    Draw histogram in logarithmic scale or not
        **fig_kw : 
        All additional keyword arguments are passed to the figure() call.

        """
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
            
        if self.NbAreas is None:
            raise IOError('Run the step 02 to initialize self.NbAreas')
            
        if 'pfa_test' in self.param:
            pfa_test = self.param['pfa_test']
        else:
            raise IOError('pfa_test param is None: set a value or run' + \
            ' the Step03')
         
        if 'threshold_list' is self.param:
            threshold_list = self.param['threshold_list']
        else:
            threshold_list = None
                       
        if fig is None:
            fig = plt.figure()
            
        if self.NbAreas<= 3:
            n = 1
            m = self.NbAreas
        else:
            n = self.NbAreas//3
            m = 3
            if (n*m)<self.NbAreas:
                n = n + 1

        for area in range(1, self.NbAreas+1):
            if threshold_list is None:
                threshold = None
            else:
                threshold = threshold_list[area]
            ax = fig.add_subplot(n, m, area, **fig_kw)
            self.plot_PCA_threshold(area, pfa_test, threshold, log10, ax)
           
        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        for a in fig.axes[:-1]:
            a.set_xlabel("")
        for a in fig.axes[1:]:
            a.set_ylabel("")
        
    def plot_PCA_threshold(self, area, pfa_test=None, threshold=None,
                           log10=True, ax=None):
        """ Plot the histogram and the threshold for the starting point of the 
        PCA, this version of the plot is to do before doing the PCA
        
        Parameters
        ----------
        area      : integer in [1, NbAreas] 
                    Area ID          
        pfa_test  : float
                    PFA of the test (if None, the value set during step03 is
                    used)
        threshold : float
                    Threshold value (estimated if None)
        log10     : bool
                    Draw histogram in logarithmic scale or not                    
        ax        : matplotlib.Axes
                    Axes instance in which the image is drawn
        """
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
            
        if self.NbAreas is None:
            raise IOError('Run the step 02 to initialize self.NbAreas')
            
        if pfa_test is None:
            if 'pfa_test' in self.param:
                pfa_test = self.param['pfa_test']
            else:
                raise IOError('pfa_test param is None: set a value or run' + \
                ' the Step03')
            
            
        if ax is None:
            ax = plt.gca()
                
        # Data in this spatio-spectral area
        test = O2test(self.cube_std.data[:, self.areamap._data==area])
        
        # automatic threshold computation     
        hist, bins, thre = Compute_thresh_PCA_hist(test, pfa_test)
        center = (bins[:-1] + bins[1:]) / 2
        
        if threshold is not None:
            thre = threshold
        else:
            ind = np.argmax(hist)
            mod = bins[ind]
            ind2 = np.argmin(( hist[ind]/2 - hist[:ind] )**2)
            fwhm = mod - bins[ind2]
            sigma = fwhm/np.sqrt(2*np.log(2))           
            gauss = stats.norm.pdf(center, loc=mod, scale=sigma)
            if log10:
                gauss = np.log10(gauss)
                
        if log10:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = np.log10(hist)
            
        ax.plot(center, hist,'-k')
        ax.plot(center, hist,'.r')
        ym,yM = ax.get_ylim()
        if threshold is None:
            ax.plot(center, gauss,'-b', alpha=.5)
        ax.plot([thre,thre],[ym,yM],'b', lw=2, alpha=.5)
        ax.grid()
        ax.set_xlim((center.min(),center.max()))
        ax.set_ylim((ym,yM))
        ax.set_xlabel('frequency')
        ax.set_ylabel('value')
        ax.text(0.7, 0.8 ,'zone %d\nthreshold %.2f'%(area, thre),
                transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.5)) 
            
        
    def plot_mapPCA(self, area=None, iteration=None, ax=None, **kwargs):
        """ Plot at a given iteration (or at the end) the number of times
        a spaxel got cleaned by the PCA
        
        Parameters
        ----------
        area: integer in [1, NbAreas]
                if None draw the full map for all areas
        iteration : integer
                    Display the nuisance/bacground pixels at iteration k
        ax        : matplotlib.Axes
                    The Axes instance in which the image is drawn
        kwargs : matplotlib.artist.Artist
                 Optional extra keyword/value arguments to be passed to
                 the ``ax.imshow()`` function
        """

        if self.mapO2 is None:
            raise IOError('Run the step 03 to initialize self.mapO2')

        themap = self.mapO2.copy()
        title = 'Number of times the spaxel got cleaned by the PCA'
        if iteration is not None:
            title += '\n%d iterations'%iteration
        if area is None:
            title += ' (Full map)'
        else:
            mask = np.ones_like(self.mapO2._data, dtype=np.bool)
            mask[self.areamap._data == area] = False
            themap._mask = mask
            title += ' (zone %d)' %area
            
        if iteration is not None:
            themap[themap._data<iteration] = np.ma.masked
            
        if ax is None:
            ax = plt.gca()
            
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'jet'
            
        themap.plot(title=title, colorbar='v', ax=ax, **kwargs)
  
    
        
    def plot_segmentation(self, pfa=5e-2, step=7, maxmap=True, ax=None, **kwargs):
        """ Plot the 2D segmentation map associated to a PFA
        This function draw the labels od the segmentation map which is used, 
        not with the same pfa, in :
            - self.step02_areas() to compute the automatic areas splitting for 
            the PCA
            - self.step05_threshold_pval() to compute the threshold of the 
            local maxima of correlations
            - self.step07_spatiospectral_merging() to merge the detected lines
            from the same sources.
            
        Parameters
        ----------
        pfa  : float
               Pvalue for the test which performs segmentation
        step : int
               The Segmentation map as used in this step: (2/5/7)
        maxmap : bool
                 If true, segmentation map is plotted as contours on the maxmap
        ax   : matplotlib.Axes
               The Axes instance in which the image is drawn
        kwargs : matplotlib.artist.Artist
                 Optional extra keyword/value arguments to be passed to
                 the ``ax.imshow()`` function
        """
        if self.cont_dct is None:
            raise IOError('Run the step 01 to initialize self.cont_dct') 
        if maxmap and self.maxmap is None:
            raise IOError('Run the step 04 to initialize self.maxmap')
            
        if ax is None:
            ax = plt.gca()
            
        if step == 2: 
            radius=2        
            dxy = 2 * radius
            x = np.linspace(-dxy,dxy,1 + (dxy)*2)
            y = np.linspace(-dxy,dxy,1 + (dxy)*2)
            xv, yv = np.meshgrid(x, y)   
            r = np.sqrt(xv**2 + yv**2)
            disk = (np.abs(r)<=radius)  
            mask = disk 
            clean = True 
        elif step == 5:
            mask = None 
            clean = False 
        elif step == 7:
            mask = None 
            clean = True
        else:
            raise IOError('sept must be equal to 2 or 5 or 7')
            
        map_in = Segmentation(self.segmentation_test.data, pfa, \
                              clean=clean, mask=mask)
                              
        if maxmap:
            self.maxmap[self.maxmap._data == 0] = np.ma.masked
            self.maxmap.plot(ax=ax)
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'Greys'
            if 'interpolation' not in kwargs:
                kwargs['interpolation'] = 'nearest'
            kwargs['origin'] = 'lower'
            ax.contour(map_in, [0], **kwargs)
        else:
            ima = Image(data=map_in, wcs=self.wcs)
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'jet'
            ima.plot(title='Labels of segmentation, pfa: %f' %(pfa), ax=ax,
                     **kwargs)


    def plot_thresholdVsPFA_background(self, purity=.9, 
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
        
        ax.plot(pfaset,threshold,'-o')              
        ax.set_xlabel('PFA')
        ax.set_ylabel('Threshold')        
        ax.set_title('Purity %f' %purity)
        
    def plot_purity(self, thr_type, ax=None, log10=True):
        """Draw number of sources per threshold computed in step05
        
        Parameters
        ----------
        thr_type  : str
                     Pvalue for the test which performs segmentation
                     'background'/'sources'
        ax : matplotlib.Axes
             The Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """
                
        if thr_type == 'background':
            i = 0
        elif thr_type == 'sources':
            i = 1
        else:
            raise IOError('thr_type must be "background" or "sources"')
        
            
        if self.Det_M is None:
            raise IOError('Run the step 05')
            
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
            ax.semilogy( index_pval, Det_M, 'b.-',
                        label = 'n detections (+DATA)' )
            ax.semilogy( index_pval, Det_m, 'g.-',
                        label = 'n detections (-DATA)' )
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
        ax.set_title('%s - threshold %f' %(thr_type, threshold))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, loc=2)                      
        
    def plot_NB(self, src_ind, ax1=None, ax2=None, ax3=None):
        """Plot the narrow bands images
        
        src_ind : integer
                  Index of the object in self.Cat0
        ax1     : matplotlib.Axes
                  The Axes instance in which the NB image around the source is
                  drawn
        ax2     : matplotlib.Axes
                  The Axes instance in which a other NB image for check is
                  drawn
        ax3     : matplotlib.Axes
                  The Axes instance in which the difference is drawn
        """
        if self.Cat0 is None:
            raise IOError('Run the step 04 to initialize self.Cat0')
            
        if ax1 is None and ax2 is None and ax3 is None:
            ax1 = plt.subplot(1,3,1)
            ax2 = plt.subplot(1,3,2)
            ax3 = plt.subplot(1,3,3)
            
        # Coordinates of the source
        x0 = self.Cat0[src_ind]['x']
        y0 = self.Cat0[src_ind]['y']
        z0 = self.Cat0[src_ind]['z']
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
        num_prof = self.Cat0[src_ind]['profile']
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
        