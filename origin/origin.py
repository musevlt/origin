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
    Compute_PCA_threshold, \
    Create_local_max_cat, \
    Estimation_Line, \
    Segmentation, \
    Correlation_GLR_test_zone, \
    Compute_threshold_purity, \
    CleanCube, \
    Purity_Estimation, \
    area_segmentation_square_fusion, \
    area_segmentation_sources_fusion, \
    area_segmentation_convex_fusion, \
    area_growing, \
    area_segmentation_final, \
    __version__    

def _format_cat(Cat, i):
    try:
        Cat['ra'].format = '.3f'
        Cat['dec'].format = '.3f'
        Cat['lbda'].format = '.2f'
        Cat['T_GLR'].format = '.2f'
        if i>0:
            Cat['residual'].format = '.3f'
            Cat['flux'].format = '.1f'
            Cat['purity'].format = '.3f'
        if i>1:  
            Cat['x2'].format = '.1f'
            Cat['y2'].format = '.1f'
    except:
        logger = logging.getLogger('origin')
        logger.info('Invalid format for the Catalog')
    

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
                             the faint signal). Result of 
                             step04_compute_greedy_PCA.
        cube_correl        : `~mpdaf.obj.Cube`
                             Cube of T_GLR values. Result of
                             step05_compute_TGLR. From Max correlations
        cube_profile       : `~mpdaf.obj.Cube` (type int)
                             Number of the profile associated to the T_GLR.
                             Result of step05_compute_TGLR.
        cube_local_max     : `~mpdaf.obj.Cube`
                             Cube of Local maxima of T_GLR values. Result of                             
                             step05_compute_TGLR. From Max correlations
        cube_local_min     : `~mpdaf.obj.Cube`
                             Cube of Local maximam of T_GLR values. Result of
                             step05_compute_TGLR. From Min correlations                             
        Cat0               : astropy.Table
                             Catalog returned by step06_threshold_pval
        Cat1               : astropy.Table
                             Catalog returned by step07_compute_spectra.
        spectra            : list of `~mpdaf.obj.Spectrum`
                             Estimated lines. Result of step06_compute_spectra.
        continuum          : list of `~mpdaf.obj.Spectrum`
                             Roughly Estimated continuum. 
                             Result of step07_compute_spectra.                             
    """
    
    def __init__(self, path, name, filename,  fieldmap, profiles, PSF, FWHM_PSF,
                 cube_faint, mapO2, thresO2, testO2, histO2, binO2, meaO2, stdO2,
                 cube_correl, maxmap, NbAreas,
                 cube_profile, Cat0, Pval_r, index_pval, Det_M, Det_m,
                 Cat1, spectra, param, cube_std, var,
                 cube_local_max, cont_dct,
                 segmentation_test, segmentation_map_threshold,
                 cube_local_min, continuum,
                 mapThresh, areamap, imawhite, imadct, imastd, zm, ym, xm):
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
        # check that the profiles have the same size
        if len(np.unique([p.shape[0] for p in self.profiles])) != 1:
            raise IOError('The profiles must have the same size')
        
        #FSF
        # FSF cube(s)
        # map fileds in the case of MUSE mosaic
        self.wfields = None
        if PSF is None or FWHM_PSF is None:
            self._loginfo('Compute FSFs from the datacube FITS header ' + \
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
                    self.param['FWHM PSF'] = self.FWHM_PSF.tolist()
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
                    self._loginfo('Compute weight maps from field map %s'%fieldmap)
                    fmap = FieldsMap(fieldmap, nfields=nfields)
                    # weighted field map
                    self.wfields = fmap.compute_weights()
                    self.param['FWHM PSF'] = self.FWHM_PSF
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
                self.PSF = cubePSF._data
                # mean of the fwhm of the FSF in pixel
                self.FWHM_PSF = np.mean(FWHM_PSF)
                self.param['FWHM PSF'] = FWHM_PSF.tolist()
                self._loginfo('mean FWHM of the FSFs = %.2f pixels'%self.FWHM_PSF)
            else:
                nfields = len(PSF)
                self.PSF = []
                self.wfields = []
                self.FWHM_PSF = FWHM_PSF.tolist()
                for n in range(nfields):
                    self._loginfo('Load FSF from %s'%PSF[n])
                    self.PSF.append(Cube(PSF[n])._data)
                    # weighted field map
                    self._loginfo('Load weight maps from %s'%fieldmap[n])
                    self.wfields.append(Image(fieldmap[n])._data)
                    self._loginfo('mean FWHM of the FSFs' + \
                        ' (field %d) = %.2f pixels'%(n, FWHM_PSF[n]))
            
        # additional images
        if imawhite is None:
            self.ima_white = cub.mean(axis=0)
        else:
            self.ima_white =  imawhite
        
        del cub
        
        # step1
        self.cube_std = cube_std
        self.cont_dct = cont_dct       
        self.segmentation_test = segmentation_test     
        if imadct is None and self.cont_dct is not None:
            self.ima_dct = self.cont_dct.mean(axis=0)
        else:
            self.ima_dct = imadct
        if imastd is None and self.cube_std is not None:
            self.ima_std = self.cube_std.mean(axis=0)
        else:
            self.ima_std = imastd
        # step 2 
        self.NbAreas = NbAreas
        self.areamap = areamap
        # step3
        self.thresO2 = thresO2
        self.testO2 = testO2
        self.histO2 = histO2
        self.binO2 = binO2
        self.meaO2 = meaO2
        self.stdO2 = stdO2
        # step4
        self.cube_faint = cube_faint
        self.mapO2 = mapO2
        # step5
        self.cube_correl = cube_correl
        self.cube_local_max = cube_local_max     
        self.cube_local_min = cube_local_min             
        self.cube_profile = cube_profile
        self.maxmap = maxmap
        # step6
        self.segmentation_map_threshold = segmentation_map_threshold    
        self.mapThresh = mapThresh
        self.Pval_r = Pval_r
        self.index_pval = index_pval
        self.Det_M = Det_M
        self.Det_m = Det_m   
        # step7
        self.Cat0 = Cat0
        if zm is not None and ym is not None and xm is not None:
            self.det_correl_min = (zm, ym, xm)
        else:
            self.det_correl_min = None
        # step8
        self.Cat1 = Cat1
        self.spectra = spectra
        # step9
#        self.Cat2 = Cat2
#        if self.Cat2 is not None:
#            self.Cat2b = self._Cat2b()
#        else:
#            self.Cat2b = None
        
        self._loginfo('00 Done')
        
    @classmethod
    def init(cls, cube, fieldmap=None, profiles=None, PSF=None, FWHM_PSF=None, name='origin'):
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
        fieldmap    : string
                      FITS file containing the field map (mosaic)
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
                      ORIGIN.write() method saves the session in a folder that
                      has this name. The ORIGIN.load() method will be used to
                      load a session, continue it or create a new from it.
        """
        return cls(path='.',  name=name, filename=cube, fieldmap=fieldmap,
                   profiles=profiles, PSF=PSF, FWHM_PSF=FWHM_PSF, 
                   cube_faint=None, mapO2=None, thresO2=None, testO2=None,
                   histO2=None, binO2=None, meaO2=None, stdO2=None, cube_correl=None,
                   maxmap=None, NbAreas=None, cube_profile=None, Cat0=None, 
                   Pval_r=None, index_pval=None, Det_M=None, Det_m=None,
                   Cat1=None, spectra=None,
                   param=None, cube_std=None, var=None,
                   cube_local_max=None,cont_dct=None,
                   segmentation_test=None, segmentation_map_threshold=None, 
                   cube_local_min=None,
                   continuum=None, mapThresh=None, areamap=None, imawhite=None,
                   imadct=None, imastd=None, zm=None, ym=None, xm=None)
        
    @classmethod
    def load(cls, folder, newname=None):
        """Load a previous session of ORIGIN.
        
        ORIGIN.write() method saves a session in a folder that has the name of
        the ORIGIN object (self.name)
        
        Parameters
        ----------
        folder  : string
                  Folder name (with the relative path) where the ORIGIN data 
                  have been stored.
        newname : string
                  New name for this session.
                  This parameter lets the user to load a previous session but
                  continue in a new one.
                  If None, the user will continue the loaded session. 
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
            if os.path.isfile('%s/cube_psf.fits'%folder):
                PSF = '%s/cube_psf.fits'%folder
            else:
                PSF_files = glob.glob('%s/cube_psf_*.fits'%folder)
                if len(PSF_files) == 0:
                    PSF = None
                elif len(PSF_files) == 1:
                    PSF = PSF_files[0]
                else:
                    PSF = sorted(PSF_files)
        wfield_files = glob.glob('%s/wfield_*.fits'%folder)
        if len(wfield_files) == 0:
            wfields = None
        else:
            wfields = sorted(wfield_files)

        if 'nbareas' in param:
            NbAreas = param['nbareas']
        else:
            NbAreas = None
        # step0
        if os.path.isfile('%s/ima_white.fits'%folder):
            ima_white = Image('%s/ima_white.fits'%folder)
        else:
            ima_white = None
            
        # step1
        if os.path.isfile('%s/cube_std.fits'%folder):
            cube_std = Cube('%s/cube_std.fits'%folder)
            var = cube_std._var
            cube_std._var = None
        else:
            cube_std = None
            var = None
        if os.path.isfile('%s/ima_std.fits'%folder):
            ima_std = Image('%s/ima_std.fits'%folder)
        else:
            ima_std = None
        if os.path.isfile('%s/cont_dct.fits'%folder):
            cont_dct = Cube('%s/cont_dct.fits'%folder)
        else:
            cont_dct = None     
        if os.path.isfile('%s/ima_dct.fits'%folder):
            ima_dct = Image('%s/ima_dct.fits'%folder)
        else:
            ima_dct = None
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
        if os.path.isfile('%s/thresO2.txt'%(folder)):
            thresO2 = np.loadtxt('%s/thresO2.txt'%(folder), ndmin=1)
            thresO2 = thresO2.tolist()
        else:
            thresO2 = None
        if NbAreas is not None: 
            if os.path.isfile('%s/testO2_1.txt'%(folder)):
                testO2 = []
                for area in range(1, NbAreas+1):
                    testO2.append(np.loadtxt('%s/testO2_%d.txt'%(folder,area),
                                             ndmin=1))
            else:
                testO2 = None
            if os.path.isfile('%s/histO2_1.txt'%(folder)):
                histO2 = []
                for area in range(1, NbAreas+1):
                    histO2.append(np.loadtxt('%s/histO2_%d.txt'%(folder, area),
                                             ndmin=1))
            else:
                histO2 = None
            if os.path.isfile('%s/binO2_1.txt'%(folder)):
                binO2 = []
                for area in range(1, NbAreas+1):
                    binO2.append(np.loadtxt('%s/binO2_%d.txt'%(folder, area),
                                             ndmin=1))
            else:
                binO2 = None
        else:
            testO2 = None
            histO2 = None
            binO2 = None
        if os.path.isfile('%s/meaO2.txt'%(folder)):
            meaO2 = np.loadtxt('%s/meaO2.txt'%(folder), ndmin=1)
            meaO2 = meaO2.tolist()
        else:
            meaO2 = None
        if os.path.isfile('%s/stdO2.txt'%(folder)):
            stdO2 = np.loadtxt('%s/stdO2.txt'%(folder), ndmin=1)
            stdO2 = stdO2.tolist()
        else:
            stdO2 = None
            
        # step4
        if os.path.isfile('%s/cube_faint.fits'%folder):
            cube_faint = Cube('%s/cube_faint.fits'%folder)
        else:
            cube_faint = None
        if os.path.isfile('%s/mapO2.fits'%folder):
            mapO2 = Image('%s/mapO2.fits'%folder)
        else:
            mapO2 = None            
            
        # step5
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

        # step6
        if os.path.isfile('%s/Pval_r.txt'%folder):
            Pval_r = np.loadtxt('%s/Pval_r.txt'%folder).astype(np.float)
        else:
            Pval_r = None
        if os.path.isfile('%s/index_pval.txt'%folder):
            index_pval = np.loadtxt('%s/index_pval.txt'%folder)\
            .astype(np.float) 
        else:
            index_pval = None
        if os.path.isfile('%s/Det_M.txt'%folder):
            Det_M = np.loadtxt('%s/Det_M.txt'%folder).astype(np.int)
        else:
            Det_M = None
        if os.path.isfile('%s/Det_min.txt'%folder):
            Det_m = np.loadtxt('%s/Det_min.txt'%folder).astype(np.int)
        else:
            Det_m = None
        # step7  
        if os.path.isfile('%s/Cat0.fits'%folder):
            Cat0 = Table.read('%s/Cat0.fits'%folder)
            _format_cat(Cat0, 0)
        else:
            Cat0 = None   
        if os.path.isfile('%s/zm.txt'%folder):
            zm = np.loadtxt('%s/zm.txt'%folder, ndmin=1).astype(np.int)
        else:
            zm = None
        if os.path.isfile('%s/ym.txt'%folder):
            ym = np.loadtxt('%s/ym.txt'%folder, ndmin=1).astype(np.int)
        else:
            ym = None
        if os.path.isfile('%s/xm.txt'%folder):
            xm = np.loadtxt('%s/xm.txt'%folder, ndmin=1).astype(np.int)
        else:
            xm = None
        # step8
        if os.path.isfile('%s/Cat1.fits'%folder):
            Cat1 = Table.read('%s/Cat1.fits'%folder)
            _format_cat(Cat1, 1)
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
        
        # step9
        if os.path.isfile('%s/segmentation_map_threshold.fits'%folder):
            segmentation_map_threshold = Image('%s'%folder + \
            '/segmentation_map_threshold.fits')
        else:
            segmentation_map_threshold = None
            
        if os.path.isfile('%s/mapThresh.fits'%folder):
            mapThresh = Image('%s/mapThresh.fits'%folder)
        else:
            mapThresh = None            

        if newname is not None:
            name = newname
                
        return cls(path=path,  name=name, filename=param['cubename'],
                   fieldmap=wfields,
                   profiles=param['profiles'], PSF=PSF, FWHM_PSF=FWHM_PSF,
                   cube_std=cube_std, var=var,
                   cube_faint=cube_faint, mapO2=mapO2, thresO2=thresO2,
                   testO2=testO2, histO2=histO2, binO2=binO2, meaO2=meaO2, stdO2=stdO2,
                   cube_correl=cube_correl,
                   maxmap=maxmap, NbAreas=NbAreas, cube_profile=cube_profile, 
                   Cat0=Cat0, Pval_r=Pval_r,
                   index_pval=index_pval, Det_M=Det_M, Det_m=Det_m,
                   Cat1=Cat1, spectra=spectra,
                   param=param,
                   cube_local_max=cube_local_max, cont_dct=cont_dct,
                   segmentation_test=segmentation_test,
                   segmentation_map_threshold=segmentation_map_threshold,
                   cube_local_min=cube_local_min, continuum=continuum,
                   mapThresh=mapThresh, areamap=areamap, imawhite=ima_white,
                   imadct=ima_dct, imastd=ima_std, zm=zm, ym=ym, xm=xm)
                   
    def _loginfo(self, logstr):
        self._log_file.info(logstr) 
        self._log_stdout.info(logstr)
                   
    def write(self, path=None, erase=False):
        """Save the current session in a folder that will have the name of the
        ORIGIN object (self.name)
        
        The ORIGIN.load(folder, newname=None) method will be used to load a
        session. The parameter newname will let the user to load a session but
        continue in a new one.
        
        Parameters
        ----------
        path  : string
                Path where the folder (self.name) will be stored.
        erase : bool
                Remove the folder if it exists
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
            if erase:
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
        if self.wfields is not None:
            for i, wfield in enumerate(self.wfields):
                Image(data=wfield, mask=np.ma.nomask).write('%s'%path2 + \
                '/wfield_%02d.fits'%i)
            
        if self.ima_white is not None:
            self.ima_white.write('%s/ima_white.fits'%path2)
            
        #step1
        if self.cube_std is not None:
            self.cube_std._var = self.var
            self.cube_std.write('%s/cube_std.fits'%path2)
            self.cube_std._var = None
        if self.cont_dct is not None:
            self.cont_dct.write('%s/cont_dct.fits'%path2)
        if self.segmentation_test is not None:
            self.segmentation_test.write('%s/segmentation_test.fits'%path2)
        if self.ima_std is not None:
            self.ima_std.write('%s/ima_std.fits'%path2)
        if self.ima_dct is not None:
            self.ima_dct.write('%s/ima_dct.fits'%path2)

        #step2
        if self.areamap is not None:
            self.areamap.write('%s/areamap.fits'%path2)
                
        #step3
        if self.thresO2 is not None:
            np.savetxt('%s/thresO2.txt'%path2, self.thresO2)
        if self.NbAreas is not None:
            if self.testO2 is not None:
                for area in range(1, self.NbAreas+1):
                    np.savetxt('%s/testO2_%d.txt'%(path2, area),
                               self.testO2[area-1])
            if self.histO2 is not None:
                for area in range(1, self.NbAreas+1):
                    np.savetxt('%s/histO2_%d.txt'%(path2, area),
                               self.histO2[area-1])
            if self.binO2 is not None:
                for area in range(1, self.NbAreas+1):
                    np.savetxt('%s/binO2_%d.txt'%(path2, area),
                               self.binO2[area-1])
        if self.meaO2 is not None:
            np.savetxt('%s/meaO2.txt'%path2, self.meaO2)
        if self.stdO2 is not None:
            np.savetxt('%s/stdO2.txt'%path2, self.stdO2)
                           
        # step4
        if self.cube_faint is not None:
            self.cube_faint.write('%s/cube_faint.fits'%path2)
        if self.mapO2 is not None:
            self.mapO2.write('%s/mapO2.fits'%path2)            

        # step5
        if self.cube_correl is not None:
            self.cube_correl.write('%s/cube_correl.fits'%path2)
        if self.cube_profile is not None:
            self.cube_profile.write('%s/cube_profile.fits'%path2)
        if self.maxmap is not None:
            self.maxmap.write('%s/maxmap.fits'%path2)
        if self.cube_local_max is not None:
            self.cube_local_max.write('%s/cube_local_max.fits'%path2)    
        if self.cube_local_min is not None:
            self.cube_local_min.write('%s/cube_local_min.fits'%path2) 

        # step6                       
        if self.segmentation_map_threshold is not None:
            self.segmentation_map_threshold.write('%s'%path2 + \
            '/segmentation_map_threshold.fits')  
        if self.mapThresh is not None:
            self.mapThresh.write('%s/mapThresh.fits'%path2)
        if self.Pval_r is not None:
            np.savetxt('%s/Pval_r.txt'%(path2), self.Pval_r)
        if self.index_pval is not None:
            np.savetxt('%s/index_pval.txt'%(path2), self.index_pval)
        if self.Det_M is not None:
            np.savetxt('%s/Det_M.txt'%(path2), self.Det_M)
        if self.Det_m is not None:
            np.savetxt('%s/Det_min.txt'%(path2), self.Det_m)
        
        # step7
        if self.Cat0 is not None:
            self.Cat0.write('%s/Cat0.fits'%path2, overwrite=True)
        if self.det_correl_min is not None:
            np.savetxt('%s/zm.txt'%(path2), self.det_correl_min[0])
            np.savetxt('%s/ym.txt'%(path2), self.det_correl_min[1])
            np.savetxt('%s/xm.txt'%(path2), self.det_correl_min[2])
        
        # step8
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
        self.ima_std          : `~mpdaf.obj.Image`
                                 Mean of standardized data for PCA along the
                                 wavelength axis
        self.ima_dct          : `~mpdaf.obj.Image`
                                 Mean of DCT continuum cube along the
                                 wavelength axis
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
        
        self._loginfo('Std signal saved in self.cube_std and self.ima_std')        
        self.cube_std = Cube(data=cube_std, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)
        self.ima_std = self.cube_std.mean(axis=0)
        self._loginfo('DCT continuum saved in self.cont_dct and self.ima_dct')
        self.cont_dct = Cube(data=cont_dct, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)
        self.ima_dct = self.cont_dct.mean(axis=0)
        self._loginfo('Segmentation map saved in self.segmentation_test')
        self.segmentation_test = Image(data=segmentation_test, 
                                      wcs=self.wcs, mask=np.ma.nomask)        
        self._loginfo('01 Done')

    def step02_areas(self,  pfa=.2, minsize=100, maxsize=None):
        """ Creation of automatic area         
        
        Parameters
        ----------
        pfa      :  float
                    PFA of the segmentation test to estimates sources with
                    strong continuum
        minsize  :  int
                    Lenght in pixel of the side of typical surface wanted                        
                    enough big area to satisfy the PCA
        maxsize :   int
                    Lenght in pixel of the side of maximum surface wanted                        
                            

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
        self._loginfo('   - side size = %d pixels'%minsize)
        if minsize is None:
            self._loginfo('   - minimum size = None')
        else:
            self._loginfo('   - minimum size = %d pixels**2'%minsize)
        
        self.param['pfa_areas'] = pfa
        self.param['minsize_areas'] = minsize
        self.param['maxsize_areas'] = maxsize
        
        nexpmap = (np.sum(~self.mask, axis=0) >0).astype(np.int)
        
        NbSubcube = np.maximum(1,int( np.sqrt( np.sum(nexpmap)/(minsize**2) )))
        if NbSubcube > 1:          
            if maxsize is None:
                maxsize = minsize*2
                
            MinSize = minsize**2 
            MaxSize = maxsize**2                
                
            self._loginfo('First segmentation of %d^2 square'%NbSubcube)
            self._loginfo('Squares segmentation and fusion') 
            square_cut_fus = area_segmentation_square_fusion(nexpmap, \
                            MinSize, MaxSize, NbSubcube, self.Ny, self.Nx)
            
            self._loginfo('Sources fusion')         
            square_src_fus, src = \
            area_segmentation_sources_fusion(self.segmentation_test.data, \
                                             square_cut_fus, pfa, \
                                             self.Ny, self.Nx)        
            
            self._loginfo('Convex envelope')                 
            convex_lab = area_segmentation_convex_fusion(square_src_fus, src)
            
            self._loginfo('Areas dilation')                 
            Grown_label = area_growing(convex_lab, nexpmap)        
            
            self._loginfo('Fusion of small area')
            self._loginfo('Minimum Size: %d px'%MinSize)
            self._loginfo('Maximum Size: %d px'%MaxSize)
            areamap = area_segmentation_final(Grown_label, MinSize, MaxSize)
            
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
        
    def step03_compute_PCA_threshold(self, pfa_test=.01):
        """ Loop on each zone of the data cube and estimate the threshold

        Parameters
        ----------
        pfa_test            :   float
                                Threshold of the test (default=0.01)  

        Returns
        -------
        self.testO2 : list
                      Result of the O2 test.
        self.histO2, self.binO2 : lists
                                  PCA histogram
        self.thresO2 : list
                       For each area, threshold value
        """
        self._loginfo('Step 03 - PCA threshold computation') 
        self._loginfo('   - pfa of the test = %0.2f'%pfa_test)            
        self.param['pfa_test'] = pfa_test
        
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
        if self.areamap is None:
            raise IOError('Run the step 02 to initialize self.areamap ')
            
        self.testO2 = [] # list of arrays
        self.histO2 = [] # list of arrays
        self.binO2 = [] # list of arrays
        self.thresO2 = []
        self.meaO2 = []
        self.stdO2 = []
        
        for area_ind in range(1, self.NbAreas + 1):
            # limits of each spatial zone
            ksel = (self.areamap._data == area_ind)
        
            # Data in this spatio-spectral zone
            cube_temp = self.cube_std._data[:, ksel]

            testO2, hO2, fO2, tO2, mea, std = Compute_PCA_threshold(cube_temp,
                                                                    pfa_test)
            self._loginfo('Area %d, estimation mean/std/threshold:'%area_ind \
                                                + ' %f/%f/%f' %(mea, std, tO2))
            self.testO2.append(testO2)
            self.histO2.append(hO2)
            self.binO2.append(fO2)
            self.thresO2.append(tO2)
            self.meaO2.append(mea)
            self.stdO2.append(std)
        
        self._loginfo('03 Done')              
        
    def step04_compute_greedy_PCA(self, mixing=False, Noise_population=50,
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
        """
        self._loginfo('Step 04 - Greedy PCA computation')  
        
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
        if self.areamap is None:
            raise IOError('Run the step 02 to initialize self.areamap')
        if threshold_list is None:
            if self.thresO2 is None:
                raise IOError('Run the step 03 to initialize self.thresO2')
            thr = self.thresO2
        else:
            thr = threshold_list
            
            
        self._loginfo('   - Noise_population = %0.2f'%Noise_population)
        self._loginfo('   - List of threshold = ' + \
            " ".join("%.2f"%x for x in thr))     
        self._loginfo('   - mixing = %d'%mixing)
        self._loginfo('   - Max number of iterations = %d'%itermax)
        
        self.param['threshold_list'] = thr
        self.param['Noise_population'] = Noise_population                
        self.param['itermax'] = itermax
        self.param['mixing'] = mixing
        
        self._loginfo('Compute greedy PCA on each zone')    
        
        faint, mapO2 = \
        Compute_GreedyPCA_area(self.NbAreas, self.cube_std._data,
                               self.areamap._data, Noise_population,
                               thr, itermax, self.testO2)
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
            
        self._loginfo('04 Done')              
        

    def step05_compute_TGLR(self, NbSubcube=1, neighboors=26, ncpu=4):
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
        ncpu        :   integer
                        Number of CPUs used
                                
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
        self._loginfo('Step 05 - GLR test(NbSubcube=%d'%NbSubcube +\
        ', neighboors=%d)'%neighboors)
        
        if self.cube_faint is None:
            raise IOError('Run the step 04 to initialize self.cube_faint')
            
        self.param['neighboors'] = neighboors
        self.param['NbSubcube'] = NbSubcube

        # TGLR computing (normalized correlations)           
        self._loginfo('Correlation')
        inty, intx = Spatial_Segmentation(self.Nx, self.Ny, NbSubcube)
        if NbSubcube == 1:
            correl, profile, cm = Correlation_GLR_test(self.cube_faint._data, 
                                            self.var, self.PSF, self.wfields,
                                               self.profiles, ncpu)  
        else:              
            correl, profile, cm = Correlation_GLR_test_zone( \
                    self.cube_faint._data, self.var, self.PSF, self.wfields,
                    self.profiles, intx, inty, NbSubcube, ncpu)  
                                                                         
        
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
        
        self._loginfo('05 Done')
        
    def step06_compute_purity_threshold(self, purity=.9, pfa=0.05, tol_spat=3,
                                        tol_spec=5, filter_act=True, 
                                        spat_size=19, spect_size=10):        
        """find the threshold  for a given purity

        Parameters
        ----------
        purity : float
                 purity to automatically compute the threshold        
        pfa    : float
                 Pvalue for the test which performs segmentation       
                 TODO le sauver directement la carte de label du step02
        tol_spat : integer
                   spatial tolerance for the spatial merging (distance in pixels)
                   TODO en fonction du FWHM
        tol_spec : integer
                   spectral tolerance for the spatial merging (distance in pixels)
        filter_act : bool
                     activate or deactivate the spatio spectral filter 
        spat_size : int
                spatiale size of the spatiale filter                
        spect_size : int
                 spectral lenght of the spectral filter
                     
        Returns
        -------
        self.segmentation_map_threshold : `~mpdaf.obj.Image`
                                          Segmentation map for threshold
        """
        self._loginfo('Step 06 - Compute Purity threshold')  
        
        if self.cube_local_max is None:
            raise IOError('Run the step 05 to initialize ' + \
            'self.cube_local_max and self.cube_local_min')
        

        self._loginfo('Estimation of threshold with purity = %.1f'%purity)

            
        self._loginfo('PFA = %.2f '%pfa)
            
        self.param['purity'] = purity
        self.param['pfa'] = pfa
        self.param['tol_spat'] = tol_spat
        self.param['tol_spec'] = tol_spec
        self.param['filter_act'] = filter_act
        self.param['spat_size'] = spat_size
        self.param['spect_size'] = spect_size

        # Pval_r=purity curves
        # Det = nb de detections
        threshold, self.Pval_r, self.index_pval, \
        segmap, self.Det_M, self.Det_m = Compute_threshold_purity(
                                           purity, 
                                           self.cube_local_max.data,
                                           self.cube_local_min.data,                                            
                                           self.segmentation_test.data, pfa, 
                                           spat_size, spect_size, 
                                           tol_spat, tol_spec, filter_act)
        self.param['threshold'] = threshold                                       
        self._loginfo('Threshold: %.1f '%threshold)
     
        self.segmentation_map_threshold = Image(data=segmap,
                                    wcs=self.wcs, mask=np.ma.nomask) #TODO a supprimer si on garde celle du step02
        self._loginfo('Save the segmentation map for threshold in ' + \
        'self.segmentation_map_threshold')          
        
        self._loginfo('06 Done')
        
    def step07_detection(self, threshold=None):
        """create first catalog which contains:
        ['x', 'y', 'z', 'ID', 'ra', 'dec', 'lbda', 
        'T_GLR', 'profile', 'seg_label']

        Parameters
        ----------
        purity : float #LPI plutot le threshold. A changer.
                 if the estimated purity is not good
                 user purity to choose in the 
                            
        Returns
        -------
        self.Cat0 : astropy.Table
                    Catalogue of the referent voxels for each group.
                    Columns: x y z ra dec lbda T_GLR profile
                    Coordinates are in pixels.
        """        

        self._loginfo('Step 07 - Thresholding and spatio-spectral merging')  
        
        if self.Pval_r is None:
            raise IOError('Run the step 06 to initialize ' + \
            'Pval_r')        

        if threshold is not None:
            self.param['threshold'] = threshold
        
        # det_correl_min: 3D positions 3D of detections in correl_min
        self.Cat0, self.det_correl_min = Create_local_max_cat(self.param['threshold'], 
                                           self.cube_local_max.data,
                                           self.cube_local_min.data,                                            
                                           self.segmentation_map_threshold.data, 
                                           self.param['spat_size'],
                                           self.param['spect_size'], 
                                           self.param['tol_spat'],
                                           self.param['tol_spec'],
                                           self.param['filter_act'],
                                           self.cube_profile._data,
                                           self.wcs, self.wave)     
                                
        _format_cat(self.Cat0, 0)
        self._loginfo('Save the catalogue in self.Cat0' + \
        ' (%d lines)'%len(self.Cat0))
        self._loginfo('07 Done')  
        
    def step07_detection_lost(self, purity=None, catalog='additional'):
        """create first catalog which contains:
        ['x', 'y', 'z', 'ID', 'ra', 'dec', 'lbda', 
        'T_GLR', 'profile', 'seg_label']

        Parameters
        ----------
        purity : float
                 if the estimated purity is not good
                 user purity to choose in the 
        catalog : type of output catalog
                  'additional' : second catalog independent from Cat0
                  'complementary' : second catalog Complementary of Cat0                  
        Returns
        -------
        self.Cat0 : astropy.Table
                    Catalogue of the referent voxels for each group.
                    Columns: x y z ra dec lbda T_GLR profile
                    Coordinates are in pixels.
        """        

        self._loginfo('Step 07 bis - Thresholding and spatio-spectral merging')  
        
        if self.Cat0 is None:
            raise IOError('Run the step 07 to initialize Cat0')        
        
        self._loginfo('Compute local maximum of std cube values')
        inty, intx = Spatial_Segmentation(self.Nx, self.Ny,
                                          self.param['NbSubcube'])
        cube_local_max_faint_dct, cube_local_min_faint_dct = \
        Compute_local_max_zone(self.cube_std.data, self.cube_std.data,
                               self.mask, intx, inty, self.param['NbSubcube'],
                               self.param['neighboors'])
        
        if catalog=='complementary': 
            cube_local_max_faint_dct, cube_local_min_faint_dct = \
            CleanCube(cube_local_max_faint_dct, cube_local_min_faint_dct, 
                      self.Cat0, self.det_correl_min, self.Nz, self.Nx, self.Ny, 
                      self.param['spat_size'], self.param['spect_size'])   

        if purity is None:
            purity = self.param['purity']        

        self._loginfo('Threshold computed with purity = %.1f'%purity)    
        
        self.cube_local_max_faint_dct = cube_local_max_faint_dct
        self.cube_local_min_faint_dct = cube_local_min_faint_dct
    
        self.ThresholdPval2, self.Pval_r2, self.index_pval2, \
        segmap2, self.Det_M2, self.Det_m2 = Compute_threshold_purity(
                                           purity, 
                                           cube_local_max_faint_dct,
                                           cube_local_min_faint_dct,                                            
                                           self.segmentation_test.data,
                                           self.param['pfa'], 
                                           self.param['spat_size'],
                                           self.param['spect_size'],
                                           self.param['tol_spat'],
                                           self.param['tol_spec'],
                                           self.param['filter_act'])
        
        self.Catcomp, inut = Create_local_max_cat(self.ThresholdPval2, 
                                           cube_local_max_faint_dct,
                                           cube_local_min_faint_dct,                                            
                                           segmap2, 
                                           self.param['spat_size'],
                                           self.param['spect_size'],
                                           self.param['tol_spat'],
                                           self.param['tol_spec'],
                                           self.param['filter_act'],
                                           self.cube_profile._data,
                                           self.wcs, self.wave)      
                                           
        #LPI a merger avec Cat0
        #LPI si filter(complematary) a additionner sinon a merger
        #(utiliser spatio spectral merging avec indicateur pour origin des sources)
        
        self._loginfo('07 Done')
        
        
    def step08_compute_spectra(self, grid_dxy=0):
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
                       Columns: ra dec lbda x0 x1 y0 y1 z0 z1 T_GLR profile
                                residual flux num_line purity
        self.spectra : list of `~mpdaf.obj.Spectrum`
                       Estimated lines
        """
        self._loginfo('Step08 - Lines estimation (grid_dxy=%d)' %(grid_dxy))
        self.param['grid_dxy'] = grid_dxy

        if self.Cat0 is None:
            raise IOError('Run the step 07 to initialize self.Cat0 catalogs')
            
        self.Cat1, Cat_est_line_raw_T, Cat_est_line_var_T = \
        Estimation_Line(self.Cat0, self.cube_raw, self.var, self.PSF, \
                     self.wfields, self.wcs, self.wave, size_grid = grid_dxy, \
                     criteria = 'flux', order_dct = 30, horiz_psf = 1, \
                     horiz = 5)
            
        self._loginfo('Purity estimation')    
        self.Cat1 = Purity_Estimation(self.Cat1, self.cube_correl.data, 
                                        self.Pval_r, self.index_pval)
                   
        _format_cat(self.Cat1, 1)
        self._loginfo('Save the updated catalogue in self.Cat1' + \
        ' (%d lines)'%len(self.Cat1))
        
        self.spectra = [] 
        for data, vari in zip(Cat_est_line_raw_T, Cat_est_line_var_T): 
            spe = Spectrum(data=data, var=vari, wave=self.wave,
                           mask=np.ma.nomask)
            self.spectra.append(spe)
        self._loginfo('Save the estimated spectrum of each line in ' + \
        'self.spectra')
        
        self._loginfo('08 Done')
#        self.Cat2b = self._Cat2b()

#    def step09_spatiospectral_merging(self, deltaz=20, pfa=0.05):
#        """Construct a catalogue of sources by spatial merging of the
#        detected emission lines in a circle with a diameter equal to
#        the mean over the wavelengths of the FWHM of the FSF.
#        Then, merge the detected emission lines distants in an estimated source 
#        area.
#
#        Parameters
#        ----------
#        deltaz : integer
#                 Distance maximum between 2 different lines
#        pfa    : float
#                 Pvalue for the test which performs segmentation                 
#
#        Returns
#        -------
#        self.Cat2 : astropy.Table
#                    Catalogue
#                    Columns: ID ra dec lbda x0 x1 x2 y0 y1 y2 z0 z1 nb_lines
#                    T_GLR profile residual flux num_line purity seg_label
#        self.segmentation_map_spatspect : `~mpdaf.obj.Image`
#                                          Segmentation map
#        """
#        self._loginfo('Step09 Spatio spectral merging ' + \
#        '(deltaz=%d, pfa=%d)'%(deltaz, pfa))
#        if self.wfields is None:
#            fwhm = self.FWHM_PSF
#        else:
#            fwhm = np.max(np.array(self.FWHM_PSF)) # to be improved
#        self.param['deltaz'] = deltaz
#        self.param['pfa_merging'] = pfa
#        
#        if self.Cat1 is None:
#            raise IOError('Run the step 08 to initialize self.Cat1')
#
#        cat = Spatial_Merging_Circle(self.Cat1, fwhm, self.wcs)
#        self.Cat2, segmap = SpatioSpectral_Merging(cat, pfa,
#                                           self.segmentation_test.data, \
#                                           self.cube_correl.data, \
#                                           self.var, deltaz, self.wcs)
#        self.segmentation_map_spatspect = Image(data=segmap,
#                                    wcs=self.wcs, mask=np.ma.nomask)
#        self._loginfo('Save the segmentation map for spatio-spectral ' + \
#        'merging in self.segmentation_map_spatspect')  
#        
#        _format_cat(self.Cat2, 2)
#        self._loginfo('Save the updated catalogue in self.Cat2' + \
#        ' and self.Cat2b' + \
#        '(%d objects, %d lines)'%(np.unique(self.Cat2['ID']).shape[0],
#          len(self.Cat2)))
#          
#        self.Cat2b = self._Cat2b()
#        self._loginfo('09 Done')
        
    def _Cat2b(self):
        from astropy.table import MaskedColumn
        cat = self.Cat2.group_by('ID')
        lmax = max([len(g['lbda']) for g in cat.groups])
        ncat = Table(names=['ID','RA','DEC','NLINE','SEG'],
                     dtype=['i4','f4','f4','i4','i4'], masked=True)
        for l in range(lmax):
            ncat.add_column(MaskedColumn(name='LBDA{}'.format(l), dtype='f4',
                                         format='.2f'))
            ncat.add_column(MaskedColumn(name='FLUX{}'.format(l), dtype='f4',
                                         format='.1f'))
            ncat.add_column(MaskedColumn(name='EFLUX{}'.format(l), dtype='f4',
                                         format='.2f'))
            ncat.add_column(MaskedColumn(name='TGLR{}'.format(l), dtype='f4',
                                         format='.2f'))
            ncat.add_column(MaskedColumn(name='PURI{}'.format(l), dtype='f4',
                                         format='.2f'))
        for key, group in zip(cat.groups.keys,cat.groups):
            dic = {'ID':key['ID'], 'RA':group['ra'].mean(),
            'DEC':group['dec'].mean(), 'NLINE':len(group['lbda']),
            'SEG':group['seg_label'][0]}
            ksort = group['T_GLR'].argsort()[::-1]
            for k, (lbda, flux, tglr, eflux, purity) in \
            enumerate(group['lbda','flux','T_GLR','residual','purity'][ksort]):
                dic['LBDA{}'.format(k)] = lbda
                dic['FLUX{}'.format(k)] = flux
                dic['EFLUX{}'.format(k)] = eflux
                dic['PURI{}'.format(k)] = purity
                dic['TGLR{}'.format(k)] = tglr
            ncat.add_row(dic)
        ncat.sort('SEG')
        return ncat

    def step10_write_sources(self, path=None, overwrite=True,
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
        self._loginfo('Step 10 - Sources creation')
        self._loginfo('Add RA-DEC to the catalogue')
        if self.Cat1 is None:
            raise IOError('Run the step 09 to initialize self.Cat1')

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
            raise IOError('Run the step 05 to initialize self.cube_correl')
        if self.spectra is None:
            raise IOError('Run the step 08 to initialize self.spectra')
        nsources = Construct_Object_Catalogue(self.Cat1, self.spectra,
                                              self.cube_correl,
                                              self.wave, self.FWHM_profiles,
                                              path_src, self.name, self.param,
                                              src_vers, author,
                                              self.path, self.maxmap,
                                              self.segmentation_map_threshold,
                                              ncpu)                                            
                                              
        # create the final catalog
        self._loginfo('Create the final catalog- %d sources'%nsources)
        catF = Catalog.from_path(path_src, fmt='working')
        catF.write(catname, overwrite=overwrite)
                      
        self._loginfo('10 Done')

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
    
    def plot_step03_PCA_threshold(self, log10=False, ncol=3, legend=True,
                                  xlim=None, fig=None, **fig_kw):
        """ Plot the histogram and the threshold for the starting point of the 
        PCA
        
        Parameters
        ----------
        log10     : bool
                    Draw histogram in logarithmic scale or not
        ncol      : integer
                    Number of colomns in the subplots
        legend    : bool
                    If true, write pfa and threshold values as legend
        xlim      : (float, float)
                    Set the data limits for the x-axes
        fig       : matplotlib.Figure
                    Figure instance in which the image is drawn
        **fig_kw  : matplotlib.artist.Artist
                    All additional keyword arguments are passed to the figure()
                    call.

        """
        if self.NbAreas is None:
            raise IOError('Run the step 02 to initialize self.NbAreas')
                       
        if fig is None:
            fig = plt.figure()
            
        if self.NbAreas<= ncol:
            n = 1
            m = self.NbAreas
        else:
            n = self.NbAreas//ncol
            m = ncol
            if (n*m)<self.NbAreas:
                n = n + 1

        for area in range(1, self.NbAreas+1):
            if area==1:
                ax = fig.add_subplot(n, m, area, **fig_kw)
            else:
                ax = fig.add_subplot(n, m, area, sharey=fig.axes[0], **fig_kw)
            self.plot_PCA_threshold(area, 'step03', log10, legend, xlim, ax)
           
        # Fine-tune figure
        for a in fig.axes[:-1]:
            a.set_xlabel("")
        for a in fig.axes[1:]:
            a.set_ylabel("")
        plt.setp([a.get_yticklabels() for a in fig.axes], visible=False)
        plt.setp([a.get_yticklabels() for a in fig.axes[0::m]], visible=True)
        plt.setp([a.get_yticklines() for a in fig.axes], visible=False)
        plt.setp([a.get_yticklines() for a in fig.axes[0::m]], visible=True)
        fig.subplots_adjust(wspace=0)
        if xlim is not None:
            plt.setp([a.get_xticklabels() for a in fig.axes[:-m]],
                      visible=False)
            plt.setp([a.get_xticklines() for a in fig.axes[:-m]],
                      visible=False)
            fig.subplots_adjust(hspace=0)
            
    def plot_step03_PCA_stat(self, cutoff=5, ax=None):
        """ Plot the thrashold value according to the area.
        Median Absolute Deviation is used to find outliers.
        
        Parameters
        ----------
        cutoff    : float
                    Median Absolute Deviation cutoff
        ax        : matplotlib.Axes
                    The Axes instance in which the image is drawn
        """
        if self.NbAreas is None:
            raise IOError('Run the step 02 to initialize self.NbAreas')
        if self.thresO2 is None:
            raise IOError('Run the step 03 to compute the threshold values')
        if ax is None:
            ax = plt.gca()
        ax.plot(np.arange(1, self.NbAreas+1), self.thresO2, '+')
        med = np.median(self.thresO2)
        diff = np.absolute(self.thresO2 - med)
        mad = np.median(diff)
        if mad != 0:
            ksel = (diff/mad)>cutoff
            if ksel.any():
                ax.plot(np.arange(1, self.NbAreas+1)[ksel],
                        np.asarray(self.thresO2)[ksel], 'ro')
        ax.set_xlabel('area')
        ax.set_ylabel('Threshold')        
        ax.set_title('PCA threshold (med=%.2f, mad= %.2f)'%(med,mad))
        

    def plot_PCA_threshold(self, area, pfa_test='step03', log10=False,
                           legend=True, xlim=None, ax=None):
        """ Plot the histogram and the threshold for the starting point of the 
        PCA
        
        Parameters
        ----------
        area      : integer in [1, NbAreas] 
                    Area ID          
        pfa_test  : float or string
                    PFA of the test (if 'step03', the value set during step03
                    is used)
        log10     : bool
                    Draw histogram in logarithmic scale or not
        legend    : bool
                    If true, write pfa and threshold values as legend
        xlim      : (float, float)
                    Set the data limits for the x-axis                  
        ax        : matplotlib.Axes
                    Axes instance in which the image is drawn
        """
        if self.NbAreas is None:
            raise IOError('Run the step 02 to initialize self.NbAreas')
            
        if pfa_test == 'step03':
            if 'pfa_test' in self.param:
                pfa_test = self.param['pfa_test']
                hist = self.histO2[area-1]
                bins = self.binO2[area-1]
                thre = self.thresO2[area-1]
                mea = self.meaO2[area-1]
                std = self.stdO2[area-1]
            else:
                raise IOError('pfa_test param is None: set a value or run' + \
                ' the Step03')
        else:
            if self.cube_std is None:
                raise IOError('Run the step 01 to initialize self.cube_std')
            # limits of each spatial zone
            ksel = (self.areamap._data == area)
            # Data in this spatio-spectral zone
            cube_temp = self.cube_std._data[:, ksel]
            # Compute_PCA_threshold
            testO2, hist, bins, thre, mea, std = \
                                     Compute_PCA_threshold(cube_temp, pfa_test)
            
        if ax is None:
            ax = plt.gca()
        
        center = (bins[:-1] + bins[1:]) / 2
        gauss = stats.norm.pdf(center, loc=mea, scale=std)
        gauss *= hist.max()/gauss.max()

        if log10:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gauss = np.log10(gauss)
                hist = np.log10(hist)
            
        ax.plot(center, hist,'-k')
        ax.plot(center, hist,'.r')
        ax.plot(center, gauss,'-b', alpha=.5)
        ax.axvline(thre,color='b', lw=2, alpha=.5)
        ax.grid()
        if xlim is None:
            ax.set_xlim((center.min(),center.max()))
        else:
            ax.set_xlim(xlim)
        ax.set_xlabel('frequency')
        ax.set_ylabel('value')
        if legend:
            ax.text(0.1, 0.8 ,'zone %d\npfa %.2f\nthreshold %.2f'%(area,
                                                            pfa_test, thre),
                transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.5))
        else:
            ax.text(0.9, 0.9 ,'%d'%area, transform=ax.transAxes,
                    bbox=dict(facecolor='red', alpha=0.5))
            
        
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
            raise IOError('Run the step 04 to initialize self.mapO2')

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
  
    
        
    def plot_segmentation(self, pfa=5e-2, step=9, maxmap=True, ax=None, **kwargs):
        """ Plot the 2D segmentation map associated to a PFA
        This function draw the labels od the segmentation map which is used, 
        not with the same pfa, in :
            - self.step02_areas() to compute the automatic areas splitting for 
            the PCA
            - step06 ans step07 to compute the threshold of the 
            local maxima of correlations
            - self.step09_spatiospectral_merging() to merge the detected lines
            from the same sources.
            
        Parameters
        ----------
        pfa  : float
               Pvalue for the test which performs segmentation
        step : int
               The Segmentation map as used in this step: (2/6/9)
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
            raise IOError('Run the step 05 to initialize self.maxmap')
            
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
        elif step == 6:
            mask = None 
            clean = False 
        elif step == 9:
            mask = None 
            clean = True
        else:
            raise IOError('sept must be equal to 2 or 6 or 9')
            
        map_in = Segmentation(self.segmentation_test.data, pfa, \
                              clean=clean, mask=mask)
                              
        if maxmap:
            self.maxmap[self.maxmap._data == 0] = np.ma.masked
            self.maxmap.plot(ax=ax, **kwargs)
            ax.contour(map_in, [0], origin='lower', cmap='Greys')
        else:
            ima = Image(data=map_in, wcs=self.wcs)
            if 'cmap' not in kwargs:
                kwargs['cmap'] = 'jet'
            ima.plot(title='Labels of segmentation, pfa: %f' %(pfa), ax=ax,
                     **kwargs)


#    def plot_thresholdVsPFA_background(self, purity=.9, 
#                                   pfaset=np.linspace(1e-3,0.5,41), ax=None):
#        """Draw threshold of local maxima as a function of the segmentation
#        map using PFA to create source/background mask of step05.
#        
#        Parameters
#        ----------
#        purity : the purity for wich the function is plotted
#        pfaset : the list of PFA to test
#        ax : matplotlib.Axes
#             The Axes instance in which the image is drawn
#        """       
#        
#        cube_local_max = self.cube_local_max.data
#        cube_local_min = self.cube_local_min.data
#        test = self.segmentation_test.data
#        
#        threshold = \
#        thresholdVsPFA_purity(test,cube_local_max,cube_local_min,purity,pfaset)
#            
#        if ax is None:
#            ax = plt.gca()  
#        
#        ax.plot(pfaset,threshold,'-o')              
#        ax.set_xlabel('PFA')
#        ax.set_ylabel('Threshold')        
#        ax.set_title('Purity %f' %purity)
        
    def plot_purity(self, ax=None, log10=True):
        """Draw number of sources per threshold computed in step06
        
        Parameters
        ----------
        ax : matplotlib.Axes
             The Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """
            
        if self.Det_M is None:
            raise IOError('Run the step 06')
            
        if ax is None:
            ax = plt.gca()        
        
        threshold = self.param['threshold']
        Pval_r = self.Pval_r
        index_pval = self.index_pval
        purity = self.param['purity']
        Det_M = self.Det_M
        Det_m = self.Det_m
        
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
        ax.set_title('threshold %f' %threshold)
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
            raise IOError('Run the step 05 to initialize self.Cat0')
            
        if ax1 is None and ax2 is None and ax3 is None:
            ax1 = plt.subplot(1,3,1)
            ax2 = plt.subplot(1,3,2)
            ax3 = plt.subplot(1,3,3)
            
        # Coordinates of the source
        x0 = self.Cat0[src_ind]['x0']
        y0 = self.Cat0[src_ind]['y0']
        z0 = self.Cat0[src_ind]['z0']
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
        