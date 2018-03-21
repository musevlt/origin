"""
ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes
---------------------------------------------------------

This software has been developped by Carole Clastres under the supervision of
David Mary (Lagrange institute, University of Nice) and ported to Python by
Laure Piqueras (CRAL). From November 2016 the software is updated by
Antony Schutz.

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL).
Please contact Carole for more info at carole.clastres@univ-lyon1.fr
Please contact Antony for more info at antonyschutz@gmail.com

origin.py contains an oriented-object interface to run the ORIGIN software
"""

from __future__ import absolute_import, division

import astropy.units as u
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import shutil
import sys
import warnings
import yaml

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.utils import lazyproperty
from matplotlib.colors import BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

from mpdaf.log import setup_logging, setup_logfile, clear_loggers
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.MUSE import FieldsMap, get_FSF_from_cube_keywords
from mpdaf.sdetect import Catalog
from mpdaf.tools import write_hdulist_to

from .lib_origin import (
    area_growing,
    area_segmentation_convex_fusion,
    area_segmentation_final,
    area_segmentation_sources_fusion,
    area_segmentation_square_fusion,
    CleanCube,
    Compute_GreedyPCA_area,
    Compute_local_max_zone,
    Compute_PCA_threshold,
    Compute_Standardized_data,
    Compute_threshold_purity,
    Construct_Object_Catalogue,
    Correlation_GLR_test,
    Correlation_GLR_test_zone,
    Create_local_max_cat,
    create_masks,
    dct_residual,
    Estimation_Line,
    merge_similar_lines,
    Purity_Estimation,
    remove_identical_duplicates,
    Spatial_Segmentation,
    trim_spectrum_list,
    unique_sources,
)
from .version import __version__

CURDIR = os.path.dirname(os.path.abspath(__file__))


def _format_cat(Cat, i):
    try:
        Cat['ra'].format = '.3f'
        Cat['dec'].format = '.3f'
        Cat['lbda'].format = '.2f'
        Cat['T_GLR'].format = '.2f'
        if i > 0:
            Cat['STD'].format = '.2f'
        if i > 1:
            Cat['residual'].format = '.3f'
            Cat['flux'].format = '.1f'
            Cat['purity'].format = '.3f'
    except Exception:
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
        path                   : str
                                 Path where the ORIGIN data will be stored.
        name                   : str
                                 Name of the session and basename for the
                                 sources.
        param                  : dict
                                 Parameters values.
        cube_raw               : array (Nz, Ny, Nx)
                                 Raw data.
        var                    : array (Nz, Ny, Nx)
                                 Variance.
        Nx                     : int
                                 Number of columns
        Ny                     : int
                                 Number of rows
        Nz                     : int
                                 Number of spectral channels
        wcs                    : `mpdaf.obj.WCS`
                                 RA-DEC coordinates.
        wave                   : `mpdaf.obj.WaveCoord`
                                 Spectral coordinates.
        profiles               : list of array
                                 List of spectral profiles to test
        FWHM_profiles          : list
                                 FWHM of the profiles in pixels.
        wfields                : None or list of arrays
                                 List of weight maps (one per fields in the
                                 case of MUSE mosaic)
                                 None: just one field
        PSF                    : array (Nz, Nfsf, Nfsf) or list of arrays
                                 MUSE PSF (one per field)
        FWHM_PSF               : float or list of float
                                 Mean of the fwhm of the PSF in pixel (one per
                                 field).
        imawhite               : `~mpdaf.obj.Image`
                                 White image
        segmap                 : `~mpdaf.obj.Image`
                                 Segmentation map
        self.cube_std          : `~mpdaf.obj.Cube`
                                 standardized data for PCA. Result of step01.
        self.cont_dct          : `~mpdaf.obj.Cube`
                                 DCT continuum. Result of step01.
        self.ima_std           : `~mpdaf.obj.Image`
                                 Mean of standardized data for PCA along the
                                 wavelength axis. Result of step01.
        self.ima_dct           : `~mpdaf.obj.Image`
                                 Mean of DCT continuum cube along the
                                 wavelength axis. Result of step01.
        nbAreas                : int
                                 Number of area (segmentation) for the PCA
                                 computation. Result of step02.
        areamap                : `~mpdaf.obj.Image`
                                 PCA area. Result of step02.
        testO2                 : list of arrays (one per PCA area)
                                 Result of the O2 test (step03).
        histO2                 : list of arrays (one per PCA area)
                                 PCA histogram (step03).
        binO2                  : list of arrays (one per PCA area)
                                 Bins for the PCA histogram (step03).
        thresO2                : list of float
                                 For each area, threshold value (step03).
        meaO2                  : list of float
                                 Location parameter of the Gaussian fit used to
                                 estimate the threshold (step03).
        stdO2                  : list of float
                                 Scale parameter of the Gaussian fit used to
                                 estimate the threshold (step03).
        cube_faint             : `~mpdaf.obj.Cube`
                                 Projection on the eigenvectors associated to
                                 the lower eigenvalues of the data cube
                                 (representing the faint signal). Result of
                                 step04.
        mapO2                  : `~mpdaf.obj.Image`
                                 The numbers of iterations used by testO2 for
                                 each spaxel. Result of step04.
        cube_correl            : `~mpdaf.obj.Cube`
                                  Cube of T_GLR values (step05).
        cube_profile           : `~mpdaf.obj.Cube` (type int)
                                 PSF profile associated to the T_GLR (step05).
        maxmap                 : `~mpdaf.obj.Image`
                                 Map of maxima along the wavelength axis
                                 (step05).
        cube_local_max         : `~mpdaf.obj.Cube`
                                 Local maxima from max correlation (step05).
        cube_local_min         : `~mpdaf.obj.Cube`
                                 Local maxima from min correlation (step05).
        threshold              : float
                                 Estimated threshold (step06).
        Pval_r                 : array
                                 Purity curves (step06).
        index_pval             : array
                                 Indexes of the purity curves (step06).
        Det_M                  : List
                                 Number of detections in +DATA (step06).
        Det_m                  : List
                                 Number of detections in -DATA  (step06).
        Cat0                   : astropy.Table
                                 Catalog returned by step07
        zm                     : array
                                 z-position of the detections from min
                                 correlation (step07)
        ym                     : array
                                 y-position of the detections from min
                                 correlation (step07)
        xm                     : array
                                 x-position of the detections from min
                                 correlation (step07)
        Pval_r_comp            : array
                                 Purity curves (step08).
        index_pval_comp        : array
                                 Indexes of the purity curves (step08).
        Det_M_comp             : List
                                 Number of detections in +DATA (step08).
        Det_m_comp             : List
                                 Number of detections in -DATA  (step08).
        Cat1                   : astropy.Table
                                 Catalog returned by step08
        spectra                : list of `~mpdaf.obj.Spectrum`
                                 Estimated lines. Result of step09.
        Cat2                   : astropy.Table
                                 Catalog returned by step09.
    """

    def __init__(self, filename, segmap, name='origin', path='.',
                 loglevel='DEBUG', logcolor=False,
                 fieldmap=None, profiles=None, PSF=None, FWHM_PSF=None,
                 param=None, imawhite=None, cube_std=None, cont_dct=None,
                 areamap=None, thresO2=None, testO2=None, histO2=None,
                 binO2=None, meaO2=None, stdO2=None, cube_faint=None,
                 mapO2=None, cube_correl=None, maxmap=None, cube_profile=None,
                 cube_local_max=None, cube_local_min=None, Pval_r=None,
                 index_pval=None, Det_M=None, Det_m=None,
                 Cat0=None, zm=None, ym=None, xm=None, Pval_r_comp=None,
                 index_pval_comp=None, Det_M_comp=None, Det_m_comp=None,
                 Cat1=None, spectra=None, Cat2=None, Cat3_lines=None,
                 Cat3_sources=None, Cat3_spectra=None):

        # stdout logger
        setup_logging(name='origin', level=loglevel, color=logcolor,
                      fmt='%(levelname)-05s: %(message)s', stream=sys.stdout)
        self._log_stdout = logging.getLogger('origin')

        # file logger
        logfile = '%s/%s/%s.log' % (path, name, name)
        if not os.path.exists(logfile):
            logfile = '%s/%s.log' % (path, name)
        setup_logfile(name='origfile', level=logging.DEBUG,
                      logfile=logfile, fmt='%(asctime)s %(message)s')
        self._log_file = logging.getLogger('origfile')
        self._log_file.setLevel(logging.INFO)

        self._loginfo('Step 00 - Initialization (ORIGIN v%s)' % __version__)

        self.path = path
        self.name = name
        self.param = param or {}

        # MUSE data cube
        self._loginfo('Read the Data Cube %s', filename)
        self.param['cubename'] = filename
        cub = Cube(filename)

        # Flux - set to 0 the Nan
        self.cube_raw = cub.data.filled(fill_value=0)
        self.mask = cub._mask

        # variance - set to Inf the Nan
        self.var = cub._var
        self.var[np.isnan(self.var)] = np.inf

        # RA-DEC coordinates
        self.wcs = cub.wcs
        # spectral coordinates
        self.wave = cub.wave
        # Dimensions
        self.Nz, self.Ny, self.Nx = cub.shape

        # segmap
        self._loginfo('Read the Segmentation Map %s', segmap)
        self.param['segmap'] = segmap
        self.segmap = Image(segmap)

        # List of spectral profile
        self._read_profiles(profiles)

        # FSF
        self._read_fsf(cub, fieldmap, PSF=PSF, FWHM_PSF=FWHM_PSF)

        # additional images
        self.ima_white = cub.mean(axis=0) if imawhite is None else imawhite

        # free memory
        cub = None

        # Define attributes with default values for all the processing steps
        # step1
        self.cube_std = cube_std
        self.cont_dct = cont_dct
        # step 2
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
        # step7
        self.Pval_r = Pval_r
        self.index_pval = index_pval
        self.Det_M = Det_M
        self.Det_m = Det_m
        # step8
        self.Cat0 = Cat0
        if zm is not None and ym is not None and xm is not None:
            self.det_correl_min = (zm, ym, xm)
        else:
            self.det_correl_min = None
        # step9
        self.Cat1 = Cat1
        self.Pval_r_comp = Pval_r_comp
        self.index_pval_comp = index_pval_comp
        self.Det_M_comp = Det_M_comp
        self.Det_m_comp = Det_m_comp
        # step09
        self.spectra = spectra
        self.Cat2 = Cat2
        self._Cat2b = None
        self.Cat3_lines = Cat3_lines
        self.Cat3_sources = Cat3_sources
        self.Cat3_spectra = Cat3_spectra
        self._loginfo('00 Done')

    @classmethod
    def init(cls, cube, segmap, fieldmap=None, profiles=None, PSF=None,
             FWHM_PSF=None, name='origin', loglevel='DEBUG', logcolor=False):
        """Create a ORIGIN object.

        An Origin object is composed by:
        - cube data (raw data and covariance)
        - 1D dictionary of spectral profiles
        - MUSE PSF
        - parameters used to segment the cube in different zones.

        Parameters
        ----------
        cube        : str
                      Cube FITS file name
        segmap      : str
                      Segmentation map FITS filename
        fieldmap    : str
                      FITS file containing the field map (mosaic)
        profiles    : str
                      FITS of spectral profiles
                      If None, a default dictionary of 20 profiles is used.
        PSF         : str
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
        return cls(path='.', name=name, filename=cube, fieldmap=fieldmap,
                   profiles=profiles, PSF=PSF, FWHM_PSF=FWHM_PSF,
                   segmap=segmap, loglevel=loglevel, logcolor=logcolor)

    @classmethod
    def load(cls, folder, newname=None):
        """Load a previous session of ORIGIN.

        ORIGIN.write() method saves a session in a folder that has the name of
        the ORIGIN object (self.name)

        Parameters
        ----------
        folder  : str
                  Folder name (with the relative path) where the ORIGIN data
                  have been stored.
        newname : str
                  New name for this session.
                  This parameter lets the user to load a previous session but
                  continue in a new one.
                  If None, the user will continue the loaded session.
        """
        path = os.path.dirname(os.path.abspath(folder))
        name = os.path.basename(folder)

        with open('%s/%s.yaml' % (folder, name), 'r') as stream:
            param = yaml.load(stream)

        if 'FWHM PSF' in param:
            FWHM_PSF = np.asarray(param['FWHM PSF'])
        else:
            FWHM_PSF = None

        if os.path.isfile(param['PSF']):
            PSF = param['PSF']
        else:
            if os.path.isfile('%s/cube_psf.fits' % folder):
                PSF = '%s/cube_psf.fits' % folder
            else:
                PSF_files = glob.glob('%s/cube_psf_*.fits' % folder)
                if len(PSF_files) == 0:
                    PSF = None
                elif len(PSF_files) == 1:
                    PSF = PSF_files[0]
                else:
                    PSF = sorted(PSF_files)
        wfield_files = glob.glob('%s/wfield_*.fits' % folder)
        if len(wfield_files) == 0:
            wfields = None
        else:
            wfields = sorted(wfield_files)

        # step0
        if os.path.isfile('%s/ima_white.fits' % folder):
            ima_white = Image('%s/ima_white.fits' % folder)
        else:
            ima_white = None

        # step1
        if os.path.isfile('%s/cube_std.fits' % folder):
            cube_std = Cube('%s/cube_std.fits' % folder)
        else:
            cube_std = None
        if os.path.isfile('%s/cont_dct.fits' % folder):
            cont_dct = Cube('%s/cont_dct.fits' % folder)
        else:
            cont_dct = None

        # step2
        if os.path.isfile('%s/areamap.fits' % folder):
            areamap = Image('%s/areamap.fits' % folder, dtype=np.int)
        else:
            areamap = None
        if 'nbareas' in param:
            NbAreas = param['nbareas']
        else:
            NbAreas = None

        # step3
        if os.path.isfile('%s/thresO2.txt' % (folder)):
            thresO2 = np.loadtxt('%s/thresO2.txt' % (folder), ndmin=1)
            thresO2 = thresO2.tolist()
        else:
            thresO2 = None
        if NbAreas is not None:
            if os.path.isfile('%s/testO2_1.txt' % (folder)):
                testO2 = []
                for area in range(1, NbAreas + 1):
                    testO2.append(np.loadtxt('%s/testO2_%d.txt' % (folder, area),
                                             ndmin=1))
            else:
                testO2 = None
            if os.path.isfile('%s/histO2_1.txt' % (folder)):
                histO2 = []
                for area in range(1, NbAreas + 1):
                    histO2.append(np.loadtxt('%s/histO2_%d.txt' % (folder, area),
                                             ndmin=1))
            else:
                histO2 = None
            if os.path.isfile('%s/binO2_1.txt' % (folder)):
                binO2 = []
                for area in range(1, NbAreas + 1):
                    binO2.append(np.loadtxt('%s/binO2_%d.txt' % (folder, area),
                                            ndmin=1))
            else:
                binO2 = None
        else:
            testO2 = None
            histO2 = None
            binO2 = None
        if os.path.isfile('%s/meaO2.txt' % (folder)):
            meaO2 = np.loadtxt('%s/meaO2.txt' % (folder), ndmin=1)
            meaO2 = meaO2.tolist()
        else:
            meaO2 = None
        if os.path.isfile('%s/stdO2.txt' % (folder)):
            stdO2 = np.loadtxt('%s/stdO2.txt' % (folder), ndmin=1)
            stdO2 = stdO2.tolist()
        else:
            stdO2 = None

        # step4
        if os.path.isfile('%s/cube_faint.fits' % folder):
            cube_faint = Cube('%s/cube_faint.fits' % folder)
        else:
            cube_faint = None
        if os.path.isfile('%s/mapO2.fits' % folder):
            mapO2 = Image('%s/mapO2.fits' % folder)
        else:
            mapO2 = None

        # step5
        if os.path.isfile('%s/cube_correl.fits' % folder):
            cube_correl = Cube('%s/cube_correl.fits' % folder)
        else:
            cube_correl = None
        if os.path.isfile('%s/cube_local_max.fits' % folder):
            cube_local_max = Cube('%s/cube_local_max.fits' % folder)
        else:
            cube_local_max = None
        if os.path.isfile('%s/cube_local_min.fits' % folder):
            cube_local_min = Cube('%s/cube_local_min.fits' % folder)
        else:
            cube_local_min = None
        if os.path.isfile('%s/maxmap.fits' % folder):
            maxmap = Image('%s/maxmap.fits' % folder)
        else:
            maxmap = None
        if os.path.isfile('%s/cube_profile.fits' % folder):
            cube_profile = Cube('%s/cube_profile.fits' % folder)
        else:
            cube_profile = None
        # step06
        if os.path.isfile('%s/Pval_r.txt' % folder):
            Pval_r = np.loadtxt('%s/Pval_r.txt' % folder).astype(np.float)
        else:
            Pval_r = None
        if os.path.isfile('%s/index_pval.txt' % folder):
            index_pval = np.loadtxt('%s/index_pval.txt' % folder)\
                .astype(np.float)
        else:
            index_pval = None
        if os.path.isfile('%s/Det_M.txt' % folder):
            Det_M = np.loadtxt('%s/Det_M.txt' % folder).astype(np.int)
        else:
            Det_M = None
        if os.path.isfile('%s/Det_min.txt' % folder):
            Det_m = np.loadtxt('%s/Det_min.txt' % folder).astype(np.int)
        else:
            Det_m = None
        # step07
        if os.path.isfile('%s/Cat0.fits' % folder):
            Cat0 = Table.read('%s/Cat0.fits' % folder)
            _format_cat(Cat0, 0)
        else:
            Cat0 = None
        if os.path.isfile('%s/zm.txt' % folder):
            zm = np.loadtxt('%s/zm.txt' % folder, ndmin=1).astype(np.int)
        else:
            zm = None
        if os.path.isfile('%s/ym.txt' % folder):
            ym = np.loadtxt('%s/ym.txt' % folder, ndmin=1).astype(np.int)
        else:
            ym = None
        if os.path.isfile('%s/xm.txt' % folder):
            xm = np.loadtxt('%s/xm.txt' % folder, ndmin=1).astype(np.int)
        else:
            xm = None
        # step08
        if os.path.isfile('%s/Pval_r_comp.txt' % folder):
            Pval_r_comp = np.loadtxt('%s/Pval_r_comp.txt' % folder).astype(np.float)
        else:
            Pval_r_comp = None
        if os.path.isfile('%s/index_pval_comp.txt' % folder):
            index_pval_comp = np.loadtxt('%s/index_pval_comp.txt' % folder)\
                .astype(np.float)
        else:
            index_pval_comp = None
        if os.path.isfile('%s/Det_M_comp.txt' % folder):
            Det_M_comp = np.loadtxt('%s/Det_M_comp.txt' % folder).astype(np.int)
        else:
            Det_M_comp = None
        if os.path.isfile('%s/Det_min_comp.txt' % folder):
            Det_m_comp = np.loadtxt('%s/Det_min_comp.txt' % folder).astype(np.int)
        else:
            Det_m_comp = None
        if os.path.isfile('%s/Cat1.fits' % folder):
            Cat1 = Table.read('%s/Cat1.fits' % folder)
            _format_cat(Cat1, 1)
        else:
            Cat1 = None

        # step09
        def load_spectra(filename):
            spectra = []
            with fits.open(filename) as fspectra:
                for i in range(len(fspectra) // 2):
                    spectra.append(Spectrum('%s/spectra.fits' % folder,
                                            hdulist=fspectra,
                                            ext=('DATA%d' % i, 'STAT%d' % i)))
            return spectra

        if os.path.isfile('%s/spectra.fits' % folder):
            spectra = load_spectra('%s/spectra.fits' % folder)
        else:
            spectra = None
        if os.path.isfile('%s/Cat2.fits' % folder):
            Cat2 = Table.read('%s/Cat2.fits' % folder)
            _format_cat(Cat2, 2)
        else:
            Cat2 = None

        # step10
        if os.path.isfile('%s/Cat3_lines.fits' % folder):
            Cat3_lines = Table.read('%s/Cat3_lines.fits' % folder)
        else:
            Cat3_lines = None
        if os.path.isfile('%s/Cat3_sources.fits' % folder):
            Cat3_sources = Table.read('%s/Cat3_sources.fits' % folder)
        else:
            Cat3_sources = None
        if os.path.isfile('%s/Cat3_spectra.fits' % folder):
            Cat3_spectra = load_spectra('%s/Cat3_spectra.fits' % folder)
        else:
            Cat3_spectra = None

        if newname is not None:
            name = newname

        return cls(path=path, name=name, param=param,
                   filename=param['cubename'], fieldmap=wfields,
                   profiles=param['profiles'], PSF=PSF, FWHM_PSF=FWHM_PSF,
                   imawhite=ima_white, cube_std=cube_std, cont_dct=cont_dct,
                   areamap=areamap,
                   thresO2=thresO2, testO2=testO2, histO2=histO2, binO2=binO2,
                   meaO2=meaO2, stdO2=stdO2, cube_faint=cube_faint,
                   mapO2=mapO2, cube_correl=cube_correl, maxmap=maxmap,
                   cube_profile=cube_profile, cube_local_max=cube_local_max,
                   cube_local_min=cube_local_min, segmap=param['segmap'],
                   Pval_r=Pval_r, index_pval=index_pval, Det_M=Det_M,
                   Det_m=Det_m, Cat0=Cat0, zm=zm, ym=ym, xm=xm,
                   Pval_r_comp=Pval_r_comp, index_pval_comp=index_pval_comp,
                   Det_M_comp=Det_M_comp, Det_m_comp=Det_m_comp, Cat1=Cat1,
                   spectra=spectra, Cat2=Cat2, Cat3_lines=Cat3_lines,
                   Cat3_sources=Cat3_sources, Cat3_spectra=Cat3_spectra)

    def _loginfo(self, *args):
        self._log_file.info(*args)
        self._log_stdout.info(*args)

    def _logwarning(self, *args):
        self._log_file.warning(*args)
        self._log_stdout.warning(*args)

    @lazyproperty
    def ima_dct(self):
        """DCT image"""
        if self.cont_dct is not None:
            return self.cont_dct.mean(axis=0)

    @property
    def ima_std(self):
        """STD image"""
        if self.cube_std is not None:
            return self.cube_std.mean(axis=0)

    @lazyproperty
    def nbAreas(self):
        """Number of area (segmentation) for the PCA"""
        if self.areamap is not None:
            labels = np.unique(self.areamap._data)
            if 0 in labels:  # expmap=0
                return len(labels) - 1
            else:
                return len(labels)

    @property
    def threshold_correl(self):
        """Estimated threshold used to detect lines on local maxima of max
        correl"""
        return self.param.get('threshold')

    @property
    def threshold_std(self):
        """Estimated threshold used to detect complementary lines on local
        maxima of std cube"""
        return self.param.get('threshold2')

    @property
    def Cat2b(self):
        if self._Cat2b is None and self.Cat2 is not None:
            from astropy.table import MaskedColumn
            cat = self.Cat2.group_by('ID')
            lmax = max([len(g['lbda']) for g in cat.groups])
            ncat = Table(names=['ID', 'RA', 'DEC', 'DLINE', 'NLINE', 'SEG', 'COMP'],
                         dtype=['i4', 'f4', 'f4', 'f4', 'i4', 'i4', 'i4'], masked=True)
            ncat['DLINE'].format = '.2f'
            for l in range(lmax):
                ncat.add_column(MaskedColumn(name='LBDA{}'.format(l),
                                             dtype='f4', format='.2f'))
                ncat.add_column(MaskedColumn(name='FLUX{}'.format(l),
                                             dtype='f4', format='.1f'))
                ncat.add_column(MaskedColumn(name='EFLUX{}'.format(l),
                                             dtype='f4', format='.2f'))
                ncat.add_column(MaskedColumn(name='TGLR{}'.format(l),
                                             dtype='f4', format='.2f'))
                ncat.add_column(MaskedColumn(name='STD{}'.format(l),
                                             dtype='f4', format='.2f'))
                ncat.add_column(MaskedColumn(name='PURI{}'.format(l),
                                             dtype='f4', format='.2f'))
            for key, group in zip(cat.groups.keys, cat.groups):
                # compute average ra,dec and peak-to-peak distance in arcsec
                mra = group['ra'].mean()
                mdec = group['dec'].mean()
                c0 = SkyCoord([mra], [mdec], unit=(u.deg, u.deg))
                c = SkyCoord([group['ra']], [group['dec']], unit=(u.deg, u.deg))
                dist = c0.separation(c)
                ptp = dist.arcsec.ptp()
                dic = {'ID': key['ID'], 'RA': mra,
                       'DEC': mdec, 'DLINE': ptp, 'NLINE': len(group['lbda']),
                       'SEG': group['seg_label'][0], 'COMP': group['comp'][0]}
                ksort = group['T_GLR'].argsort()[::-1]
                for k, (lbda, flux, tglr, std, eflux, purity) in \
                        enumerate(group['lbda', 'flux', 'T_GLR', 'STD', 'residual', 'purity'][ksort]):
                    dic['LBDA{}'.format(k)] = lbda
                    dic['FLUX{}'.format(k)] = flux
                    dic['EFLUX{}'.format(k)] = eflux
                    dic['PURI{}'.format(k)] = purity
                    dic['TGLR{}'.format(k)] = tglr
                    dic['STD{}'.format(k)] = std
                ncat.add_row(dic)
            ncat.sort('SEG')
            self._Cat2b = ncat
        return self._Cat2b

    def _read_profiles(self, profiles=None):
        """Read the list of spectral profile."""
        self.param['profiles'] = profiles
        if profiles is None:
            profiles = CURDIR + '/Dico_FWHM_2_12.fits'
        self._loginfo('Load dictionary of spectral profile %s' % profiles)
        self.profiles = []
        self.FWHM_profiles = []
        with fits.open(profiles) as fprof:
            for hdu in fprof[1:]:
                self.profiles.append(hdu.data)
                self.FWHM_profiles.append(hdu.header['FWHM'])

        # check that the profiles have the same size
        if len(np.unique([p.shape[0] for p in self.profiles])) != 1:
            raise IOError('The profiles must have the same size')

    def _read_fsf(self, cube, fieldmap, PSF=None, FWHM_PSF=None):
        """Read FSF cube(s).
        With fieldmap in the case of MUSE mosaic
        """
        self.wfields = None
        if PSF is None or FWHM_PSF is None:
            self._loginfo('Compute FSFs from the datacube FITS header '
                          'keywords')
            if 'FSFMODE' not in cube.primary_header:
                raise IOError('PSF are not described in the FITS header'
                              'of the cube')

            # FSF created from FSF*** keywords
            Nfsf = 13
            PSF, fwhm_pix, _ = get_FSF_from_cube_keywords(cube, Nfsf)
            self.param['PSF'] = cube.primary_header['FSFMODE']
            nfields = cube.primary_header['NFIELDS']
            if nfields == 1:  # just one FSF
                # Normalization
                self.PSF = PSF / np.sum(PSF, axis=(1, 2))[:, None, None]
                # mean of the fwhm of the FSF in pixel
                self.FWHM_PSF = np.mean(fwhm_pix)
                self.param['FWHM PSF'] = self.FWHM_PSF.tolist()
                self._loginfo('mean FWHM of the FSFs = %.2f pixels',
                              self.FWHM_PSF)
            else:  # mosaic: one FSF cube per field
                self.PSF = []
                self.FWHM_PSF = []
                for i in range(nfields):
                    # Normalization
                    norm = np.sum(PSF[i], axis=(1, 2))[:, None, None]
                    self.PSF.append(PSF[i] / norm)
                    # mean of the fwhm of the FSF in pixel
                    fwhm = np.mean(fwhm_pix[i])
                    self.FWHM_PSF.append(fwhm)
                    self._loginfo('mean FWHM of the FSFs'
                                  ' (field %d) = %.2f pixels', i, fwhm)
                self._loginfo('Compute weight maps from field map %s',
                              fieldmap)
                fmap = FieldsMap(fieldmap, nfields=nfields)
                # weighted field map
                self.wfields = fmap.compute_weights()
                self.param['FWHM PSF'] = self.FWHM_PSF
        else:
            if isinstance(PSF, str):
                self._loginfo('Load FSFs from %s' % PSF)
                self.param['PSF'] = PSF

                cubePSF = Cube(PSF)
                if cubePSF.shape[1] != cubePSF.shape[2]:
                    raise IOError('PSF must be a square image.')
                if not cubePSF.shape[1] % 2:
                    raise IOError('The spatial size of the PSF must be odd.')
                if cubePSF.shape[0] != self.Nz:
                    raise IOError('PSF and data cube have not the same' +
                                  'dimensions along the spectral axis.')
                self.PSF = cubePSF._data
                # mean of the fwhm of the FSF in pixel
                self.FWHM_PSF = np.mean(FWHM_PSF)
                self.param['FWHM PSF'] = FWHM_PSF.tolist()
                self._loginfo('mean FWHM of the FSFs = %.2f pixels',
                              self.FWHM_PSF)
            else:
                nfields = len(PSF)
                self.PSF = []
                self.wfields = []
                self.FWHM_PSF = FWHM_PSF.tolist()
                for n in range(nfields):
                    self._loginfo('Load FSF from %s' % PSF[n])
                    self.PSF.append(Cube(PSF[n])._data)
                    # weighted field map
                    self._loginfo('Load weight maps from %s' % fieldmap[n])
                    self.wfields.append(Image(fieldmap[n])._data)
                    self._loginfo('mean FWHM of the FSFs'
                                  ' (field %d) = %.2f pixels', n, FWHM_PSF[n])

    def write(self, path=None, erase=False):
        """Save the current session in a folder that will have the name of the
        ORIGIN object (self.name)

        The ORIGIN.load(folder, newname=None) method will be used to load a
        session. The parameter newname will let the user to load a session but
        continue in a new one.

        Parameters
        ----------
        path  : str
            Path where the folder (self.name) will be stored.
        erase : bool
            Remove the folder if it exists.

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
        with open('%s/%s.yaml' % (path2, self.name), 'w') as stream:
            yaml.dump(self.param, stream)

        # log file
        currentlog = self._log_file.handlers[0].baseFilename
        newlog = os.path.abspath('%s/%s.log' % (path2, self.name))
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
                Cube(data=psf, mask=np.ma.nomask).write(
                    '%s' % path2 + '/cube_psf_%02d.fits' % i)
        else:
            Cube(data=self.PSF, mask=np.ma.nomask).write(
                '%s' % path2 + '/cube_psf.fits')
        if self.wfields is not None:
            for i, wfield in enumerate(self.wfields):
                Image(data=wfield, mask=np.ma.nomask).write(
                    '%s' % path2 + '/wfield_%02d.fits' % i)

        if self.ima_white is not None:
            self.ima_white.write('%s/ima_white.fits' % path2)

        # step1
        if self.cube_std is not None:
            self.cube_std.write('%s/cube_std.fits' % path2)
        if self.cont_dct is not None:
            self.cont_dct.write('%s/cont_dct.fits' % path2)
        if self.ima_std is not None:
            self.ima_std.write('%s/ima_std.fits' % path2)
        if self.ima_dct is not None:
            self.ima_dct.write('%s/ima_dct.fits' % path2)

        # step2
        if self.areamap is not None:
            self.areamap.write('%s/areamap.fits' % path2)

        # step3
        if self.thresO2 is not None:
            np.savetxt('%s/thresO2.txt' % path2, self.thresO2)
        if self.nbAreas is not None:
            if self.testO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt('%s/testO2_%d.txt' % (path2, area),
                               self.testO2[area - 1])
            if self.histO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt('%s/histO2_%d.txt' % (path2, area),
                               self.histO2[area - 1])
            if self.binO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt('%s/binO2_%d.txt' % (path2, area),
                               self.binO2[area - 1])
        if self.meaO2 is not None:
            np.savetxt('%s/meaO2.txt' % path2, self.meaO2)
        if self.stdO2 is not None:
            np.savetxt('%s/stdO2.txt' % path2, self.stdO2)

        # step4
        if self.cube_faint is not None:
            self.cube_faint.write('%s/cube_faint.fits' % path2)
        if self.mapO2 is not None:
            self.mapO2.write('%s/mapO2.fits' % path2)

        # step5
        if self.cube_correl is not None:
            self.cube_correl.write('%s/cube_correl.fits' % path2)
        if self.cube_profile is not None:
            self.cube_profile.write('%s/cube_profile.fits' % path2)
        if self.maxmap is not None:
            self.maxmap.write('%s/maxmap.fits' % path2)
        if self.cube_local_max is not None:
            hdu = fits.PrimaryHDU(header=self.cube_local_max.primary_header)
            hdui = fits.ImageHDU(name='DATA',
                                 data=self.cube_local_max.data.filled(fill_value=np.nan),
                                 header=self.cube_local_max.data_header)
            hdul = fits.HDUList([hdu, hdui])
            hdul.writeto('%s/cube_local_max.fits' % path2, overwrite=True)
#            self.cube_local_max.write('%s/cube_local_max.fits' % path2)
        if self.cube_local_min is not None:
            hdu = fits.PrimaryHDU(header=self.cube_local_min.primary_header)
            hdui = fits.ImageHDU(name='DATA',
                                 data=self.cube_local_min.data.filled(fill_value=np.nan),
                                 header=self.cube_local_min.data_header)
            hdul = fits.HDUList([hdu, hdui])
            hdul.writeto('%s/cube_local_min.fits' % path2, overwrite=True)
 #           self.cube_local_min.write('%s/cube_local_min.fits' % path2)

        # step6
        if self.Pval_r is not None:
            np.savetxt('%s/Pval_r.txt' % (path2), self.Pval_r)
        if self.index_pval is not None:
            np.savetxt('%s/index_pval.txt' % (path2), self.index_pval)
        if self.Det_M is not None:
            np.savetxt('%s/Det_M.txt' % (path2), self.Det_M)
        if self.Det_m is not None:
            np.savetxt('%s/Det_min.txt' % (path2), self.Det_m)

        # step7
        if self.Cat0 is not None:
            self.Cat0.write('%s/Cat0.fits' % path2, overwrite=True)
        if self.det_correl_min is not None:
            np.savetxt('%s/zm.txt' % (path2), self.det_correl_min[0])
            np.savetxt('%s/ym.txt' % (path2), self.det_correl_min[1])
            np.savetxt('%s/xm.txt' % (path2), self.det_correl_min[2])
        if self.Pval_r_comp is not None:
            np.savetxt('%s/Pval_r_comp.txt' % (path2), self.Pval_r_comp)
        if self.index_pval_comp is not None:
            np.savetxt('%s/index_pval_comp.txt' % (path2), self.index_pval_comp)
        if self.Det_M_comp is not None:
            np.savetxt('%s/Det_M_comp.txt' % (path2), self.Det_M_comp)
        if self.Det_m_comp is not None:
            np.savetxt('%s/Det_min_comp.txt' % (path2), self.Det_m_comp)

        # step8
        if self.Cat1 is not None:
            self.Cat1.write('%s/Cat1.fits' % path2, overwrite=True)

        # step9
        if self.Cat2 is not None:
            self.Cat2.write('%s/Cat2.fits' % path2, overwrite=True)
        if self.Cat2b is not None:
            self.Cat2b.write('%s/Cat2b.fits' % path2, overwrite=True)

        def save_spectra(spectra, outname):
            hdulist = fits.HDUList([fits.PrimaryHDU()])
            for i in range(len(spectra)):
                hdu = spectra[i].get_data_hdu(name='DATA%d' % i,
                                              savemask='nan')
                hdulist.append(hdu)
                hdu = spectra[i].get_stat_hdu(name='STAT%d' % i)
                if hdu is not None:
                    hdulist.append(hdu)
            write_hdulist_to(hdulist, outname, overwrite=True)

        if self.spectra is not None:
            save_spectra(self.spectra, '%s/spectra.fits' % path2)

        # step 10
        if self.Cat3_lines is not None:
            self.Cat3_lines.write('%s/Cat3_lines.fits' % path2, overwrite=True)
        if self.Cat3_sources is not None:
            self.Cat3_sources.write('%s/Cat3_sources.fits' % path2,
                                    overwrite=True)
        if self.Cat3_spectra is not None:
            save_spectra(self.Cat3_spectra, '%s/Cat3_spectra.fits' % path2)

        self._loginfo("Current session saved in %s" % path2)

    def step01_preprocessing(self, dct_order=10, dct_approx=True):
        """ Preprocessing of data, dct, standardization and noise compensation

        Parameters
        ----------
        dct_order   : int
                      The number of atom to keep for the dct decomposition
        dct_approx : bool
                     if True, the DCT computation is approximated

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
        """
        self._loginfo('Step 01 - Preprocessing, dct order=%d', dct_order)

        self._loginfo('DCT computation')
        self.param['dct_order'] = dct_order
        faint_dct, cont_dct = dct_residual(self.cube_raw, dct_order, self.var,
                                           dct_approx)

        # compute standardized data
        self._loginfo('Data standardizing')
        cube_std = Compute_Standardized_data(faint_dct, self.mask, self.var)
        cont_dct /= np.sqrt(self.var)

        self._loginfo('Std signal saved in self.cube_std and self.ima_std')
        self.cube_std = Cube(data=cube_std, wave=self.wave, wcs=self.wcs,
                             mask=np.ma.nomask, copy=False)
        self._loginfo('DCT continuum saved in self.cont_dct and self.ima_dct')
        self.cont_dct = Cube(data=cont_dct, wave=self.wave, wcs=self.wcs,
                             mask=np.ma.nomask, copy=False)

        self._loginfo('01 Done')

    def step02_areas(self, pfa=.2, minsize=100, maxsize=None):
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

        self.nbAreas    :   int
                            number of areas
        self.areamap : `~mpdaf.obj.Image`
                       The map of areas
        """
        self._loginfo('Step 02 - Areas creation')
        self._loginfo('   - pfa of the test = %0.2f' % pfa)
        self._loginfo('   - side size = %d pixels' % minsize)
        if minsize is None:
            self._loginfo('   - minimum size = None')
        else:
            self._loginfo('   - minimum size = %d pixels**2' % minsize)

        self.param['pfa_areas'] = pfa
        self.param['minsize_areas'] = minsize
        self.param['maxsize_areas'] = maxsize

        nexpmap = (np.sum(~self.mask, axis=0) > 0).astype(np.int)

        NbSubcube = np.maximum(1, int(np.sqrt(np.sum(nexpmap) / (minsize**2))))
        if NbSubcube > 1:
            if maxsize is None:
                maxsize = minsize * 2

            MinSize = minsize**2
            MaxSize = maxsize**2

            self._loginfo('First segmentation of %d^2 square' % NbSubcube)
            self._loginfo('Squares segmentation and fusion')
            square_cut_fus = area_segmentation_square_fusion(
                nexpmap, MinSize, MaxSize, NbSubcube, self.Ny, self.Nx)

            self._loginfo('Sources fusion')
            square_src_fus, src = area_segmentation_sources_fusion(
                self.segmap.data, square_cut_fus, pfa, self.Ny, self.Nx)

            self._loginfo('Convex envelope')
            convex_lab = area_segmentation_convex_fusion(square_src_fus, src)

            self._loginfo('Areas dilation')
            Grown_label = area_growing(convex_lab, nexpmap)

            self._loginfo('Fusion of small area')
            self._loginfo('Minimum Size: %d px' % MinSize)
            self._loginfo('Maximum Size: %d px' % MaxSize)
            areamap = area_segmentation_final(Grown_label, MinSize, MaxSize)

        elif NbSubcube == 1:
            areamap = nexpmap

        self._loginfo('Save the map of areas in self.areamap')

        self.areamap = Image(data=areamap, wcs=self.wcs, dtype=np.int)
        self._loginfo('%d areas generated' % self.nbAreas)
        self.param['nbareas'] = self.nbAreas

        self._loginfo('02 Done')

    def step03_compute_PCA_threshold(self, pfa_test=.01):
        """ Loop on each zone of the data cube and estimate the threshold

        Parameters
        ----------
        pfa_test            :   float
                                Threshold of the test (default=0.01)

        Returns
        -------
        self.testO2  : list of arrays (one per PCA area)
                       Result of the O2 test.
        self.histO2  : lists of arrays (one per PCA area)
                       PCA histogram
        self.binO2   : lists of arrays (one per PCA area)
                       bin for the PCA histogram
        self.thresO2 : list of float
                       For each area, threshold value
        self.meaO2   : list of float
                       Location parameter of the Gaussian fit used to estimate
                       the threshold
        self.stdO2   : list of float
                       Scale parameter of the Gaussian fit used to estimate
                       the threshold
        """
        self._loginfo('Step 03 - PCA threshold computation')
        self._loginfo('   - pfa of the test = %0.2f' % pfa_test)
        self.param['pfa_test'] = pfa_test

        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
        if self.areamap is None:
            raise IOError('Run the step 02 to initialize self.areamap ')

        results = []

        for area_ind in range(1, self.nbAreas + 1):
            # limits of each spatial zone
            ksel = (self.areamap._data == area_ind)

            # Data in this spatio-spectral zone
            cube_temp = self.cube_std._data[:, ksel]

            res = Compute_PCA_threshold(cube_temp, pfa_test)
            results.append(res)
            self._loginfo('Area %d, estimation mean/std/threshold: %f/%f/%f'
                          % (area_ind, res[4], res[5], res[3]))

        (self.testO2, self.histO2, self.binO2, self.thresO2, self.meaO2,
         self.stdO2) = zip(*results)

        self._loginfo('03 Done')

    def step04_compute_greedy_PCA(self, Noise_population=50,
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
        Noise_population    :   float
                                Fraction of spectra used to estimate
                                the background signature
        itermax             :   int
                                Maximum number of iterations
        threshold_list      :   list
                                User given list of threshold (not pfa) to apply
                                on each area, the list is of lenght nbAreas
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
                     The numbers of iterations used by testO2 for each spaxel
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

        self._loginfo('   - Noise_population = %0.2f' % Noise_population)
        self._loginfo('   - List of threshold = ' +
                      " ".join("%.2f" % x for x in thr))
        self._loginfo('   - Max number of iterations = %d' % itermax)

        self.param['threshold_list'] = thr
        self.param['Noise_population'] = Noise_population
        self.param['itermax'] = itermax

        self._loginfo('Compute greedy PCA on each zone')

        faint, mapO2, nstop = Compute_GreedyPCA_area(
            self.nbAreas, self.cube_std._data, self.areamap._data,
            Noise_population, thr, itermax, self.testO2)
        if nstop > 0:
            self._logwarning('The iterations have been reached the limit '
                             'of %d in %d cases', itermax, nstop)

        self._loginfo('Save the faint signal in self.cube_faint')
        self.cube_faint = Cube(data=faint, wave=self.wave, wcs=self.wcs,
                               mask=np.ma.nomask, copy=False)
        self._loginfo('Save the numbers of iterations used by the'
                      ' testO2 for each spaxel in self.mapO2')

        self.mapO2 = Image(data=mapO2, wcs=self.wcs, copy=False)

        self._loginfo('04 Done')

    def step05_compute_TGLR(self, NbSubcube=1, neighboors=26, ncpu=4):
        """Compute the cube of GLR test values.

        The test is done on the cube containing the faint signal
        (``self.cube_faint``) and it uses the PSF and the spectral profile.
        The correlation can be computed per "area"  for low memory system.
        Then a Loop on each zone of ``self.cube_correl`` is performed to
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
        NbSubcube   :   int
                        Number of sub-cubes for the spatial segmentation
                        If NbSubcube>1 the correlation and local maximas and
                        minimas are performed on smaller subcube and combined
                        after. Useful to avoid swapp
        neighboors  :   int
                        Connectivity of contiguous voxels
        ncpu        :   int
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
        self._loginfo('Step 05 - GLR test(NbSubcube=%d' % NbSubcube +
                      ', neighboors=%d)' % neighboors)

        if self.cube_faint is None:
            raise IOError('Run the step 04 to initialize self.cube_faint')

        self.param['neighboors'] = neighboors
        self.param['NbSubcube'] = NbSubcube

        # TGLR computing (normalized correlations)
        self._loginfo('Correlation')
        inty, intx = Spatial_Segmentation(self.Nx, self.Ny, NbSubcube)
        if NbSubcube == 1:
            correl, profile, cm = Correlation_GLR_test(
                self.cube_faint._data, self.var, self.PSF, self.wfields,
                self.profiles, ncpu)
        else:
            correl, profile, cm = Correlation_GLR_test_zone(
                self.cube_faint._data, self.var, self.PSF, self.wfields,
                self.profiles, intx, inty, NbSubcube, ncpu)

        self._loginfo('Save the TGLR value in self.cube_correl')
        correl[self.mask] = 0
        self.cube_correl = Cube(data=correl, wave=self.wave, wcs=self.wcs,
                                mask=np.ma.nomask, copy=False)

        self._loginfo('Save the number of profile associated to the TGLR'
                      ' in self.cube_profile')
        profile[self.mask] = 0
        self.cube_profile = Cube(data=profile, wave=self.wave, wcs=self.wcs,
                                 mask=np.ma.nomask, dtype=int, copy=False)

        self._loginfo('Save the map of maxima in self.maxmap')
        carte_2D_correl = np.amax(self.cube_correl._data, axis=0)
        self.maxmap = Image(data=carte_2D_correl, wcs=self.wcs)

        self._loginfo('Compute p-values of local maximum of correlation values')
        cube_local_max, cube_local_min = Compute_local_max_zone(
            correl, cm, self.mask, intx, inty, NbSubcube, neighboors)
        self._loginfo('Save self.cube_local_max from max correlations')
        self.cube_local_max = Cube(data=cube_local_max, wave=self.wave,
                                   wcs=self.wcs, mask=np.ma.nomask, copy=False)
        self._loginfo('Save self.cube_local_min from min correlations')
        self.cube_local_min = Cube(data=cube_local_min, wave=self.wave,
                                   wcs=self.wcs, mask=np.ma.nomask, copy=False)

        self._loginfo('05 Done')

    def step06_compute_purity_threshold(self, purity=.9, tol_spat=3,
                                        tol_spec=5, spat_size=19,
                                        spect_size=10,
                                        auto=(5, 15, 0.1), threshlist=None):
        """find the threshold  for a given purity

        Parameters
        ----------
        purity : float
                 purity to automatically compute the threshold
        tol_spat : int
                   spatial tolerance for the spatial merging (distance in pixels)
                   TODO en fonction du FWHM
        tol_spec : int
                   spectral tolerance for the spatial merging (distance in pixels)
        spat_size : int
                spatiale size of the spatiale filter
        spect_size : int
                 spectral lenght of the spectral filter
        auto    : tuple (npts1,npts2,pmargin)
                 nb of threshold sample for iteration 1 and 2, margin in purity
                 default (5,15,0.1
        threshlist : list
                 list of thresholds to compute the purity

        Returns
        -------
        self.threshold_correl : float
                                Estimated threshold
        self.Pval_r : array
                      Purity curves
        self.index_pval : array
                          Indexes of the purity curves
        self.Det_M  : List
                      Number of detections in +DATA
        self.Det_m  : List
                      Number of detections in -DATA
        """
        self._loginfo('Step 06 - Compute Purity threshold')

        if self.cube_local_max is None:
            raise IOError('Run the step 05 to initialize ' +
                          'self.cube_local_max and self.cube_local_min')

        self.param['purity'] = purity
        self.param['tol_spat'] = tol_spat
        self.param['tol_spec'] = tol_spec
        self.param['spat_size'] = spat_size
        self.param['spect_size'] = spect_size

        self._loginfo('Estimation of threshold with purity = %.2f' % purity)
        threshold, self.Pval_r, self.index_pval, self.Det_m, self.Det_M = \
            Compute_threshold_purity(purity, self.cube_local_max.data,
                                     self.cube_local_min.data, self.segmap.data,
                                     spat_size, spect_size, tol_spat, tol_spec,
                                     True, True, auto, threshlist)
        self.param['threshold'] = threshold
        self._loginfo('Threshold: %.2f ' % threshold)

        self._loginfo('06 Done')

    def step07_detection(self, threshold=None):
        """Detections on local maxima from max correlation + spatia-spectral
        merging in order to create the first catalog.

        Parameters
        ----------
        threshold : float
                    User threshod if the estimated threshold is not good

        Returns
        -------
        self.Cat0 : astropy.Table
                    First catalog
                    Columns: ID ra dec lbda x0 y0 z0 profile seg_label T_GLR
        self.det_correl_min : (array, array, array)
                              3D positions of detections in correl_min
        """

        self._loginfo('Step 07 - Thresholding and spatio-spectral merging')

        if threshold is not None:
            self.param['threshold'] = threshold

        self.Cat0, self.det_correl_min = Create_local_max_cat(
            self.param['threshold'], self.cube_local_max.data,
            self.cube_local_min.data, self.segmap.data,
            self.param['spat_size'], self.param['spect_size'],
            self.param['tol_spat'], self.param['tol_spec'],
            True, self.cube_profile._data, self.wcs,
            self.wave
        )
        _format_cat(self.Cat0, 0)
        self._loginfo('Save the catalogue in self.Cat0 (%d sources %d lines)',
                      len(np.unique(self.Cat0['ID'])), len(self.Cat0))
        self._loginfo('07 Done')

    def step08_detection_lost(self, purity=None, auto=(5, 15, 0.1),
                              threshlist=None):
        """Detections on local maxima of std cube + spatia-spectral
        merging in order to create an complementary catalog. This catalog is
        merged with the catalog Cat0 in order to create the catalog Cat1

        Parameters
        ----------
        purity : float
                 purity to automatically compute the threshold
                 If None, previous purity is used
        auto     : tuple (npts1,npts2,pmargin)
                 nb of threshold sample for iteration 1 and 2, margin in purity
                 default (5,15,0.1)
        threshlist : list
                 list of thresholds to compute the purity
                 default None
        Returns
        -------
        self.threshold_correl : float
                              Estimated threshold used to detect complementary
                              lines on local maxima of std cube
        self.Pval_r_comp : array
                      Purity curves
        self.index_pval_comp : array
                          Indexes of the purity curves
        self.Det_M_comp  : List
                      Number of detections in +DATA
        self.Det_m_comp  : List
                      Number of detections in -DATA
        self.Cat1 : astropy.Table
                    New catalog
                    Columns: ID ra dec lbda x0 y0 z0 profile seg_label T_GLR
                             STD comp
        """

        self._loginfo('Step 08 - Thresholding and spatio-spectral merging')

        if self.Cat0 is None:
            raise IOError('Run the step 07 to initialize Cat0')

        self._loginfo('Compute local maximum of std cube values')
        inty, intx = Spatial_Segmentation(self.Nx, self.Ny,
                                          self.param['NbSubcube'])
        cube_local_max_faint_dct, cube_local_min_faint_dct = \
            Compute_local_max_zone(self.cube_std.data, self.cube_std.data,
                                   self.mask, intx, inty, self.param['NbSubcube'],
                                   self.param['neighboors'])

        # complementary catalog
        cube_local_max_faint_dct, cube_local_min_faint_dct = \
            CleanCube(cube_local_max_faint_dct, cube_local_min_faint_dct,
                      self.Cat0, self.det_correl_min, self.Nz, self.Nx, self.Ny,
                      self.param['spat_size'], self.param['spect_size'])

        if purity is None:
            purity = self.param['purity']
        self.param['purity2'] = purity

        self._loginfo('Threshold computed with purity = %.1f' % purity)

        self.cube_local_max_faint_dct = cube_local_max_faint_dct
        self.cube_local_min_faint_dct = cube_local_min_faint_dct

        threshold2, self.Pval_r_comp, self.index_pval_comp, self.Det_m_comp, \
            self.Det_M_comp = Compute_threshold_purity(
                purity,
                cube_local_max_faint_dct,
                cube_local_min_faint_dct,
                self.segmap._data,
                self.param['spat_size'],
                self.param['spect_size'],
                self.param['tol_spat'],
                self.param['tol_spec'],
                True, False,
                auto, threshlist)
        self.param['threshold2'] = threshold2
        self._loginfo('Threshold: %.2f ' % threshold2)

        if threshold2 == np.inf:
            self.Cat1 = self.Cat0.copy()
            self.Cat1['comp'] = 0
            self.Cat1['STD'] = 0
        else:
            Catcomp, _ = Create_local_max_cat(threshold2,
                                              cube_local_max_faint_dct,
                                              cube_local_min_faint_dct,
                                              self.segmap._data,
                                              self.param['spat_size'],
                                              self.param['spect_size'],
                                              self.param['tol_spat'],
                                              self.param['tol_spec'],
                                              True,
                                              self.cube_profile._data,
                                              self.wcs, self.wave)
            Catcomp.rename_column('T_GLR', 'STD')
            # merging
            Cat0 = self.Cat0.copy()
            Cat0['comp'] = 0
            Catcomp['comp'] = 1
            Catcomp['ID'] += (Cat0['ID'].max() + 1)
            self.Cat1 = vstack([Cat0, Catcomp])
            _format_cat(self.Cat1, 1)
        ns = len(np.unique(self.Cat1['ID']))
        ds = ns - len(np.unique(self.Cat0['ID']))
        nl = len(self.Cat1)
        dl = nl - len(self.Cat0)
        self._loginfo('Save the catalogue in self.Cat1' +
                      ' (%d [+%s] sources %d [+%d] lines)' % (ns, ds, nl, dl))

        self._loginfo('08 Done')

    def step09_compute_spectra(self, grid_dxy=0):
        """compute the estimated emission line and the optimal coordinates
        for each detected lines in a spatio-spectral grid (each emission line
        is estimated with the deconvolution model :
        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))
        Via PCA LS or denoised PCA LS Method

        Parameters
        ----------
        grid_dxy   : int
                     Maximum spatial shift for the grid

        Returns
        -------
        self.Cat2    : astropy.Table
                       Catalogue of parameters of detected emission lines.
                       Columns: ra dec lbda x0 x y0 y z0 z T_GLR profile
                                residual flux num_line purity
        self.spectra : list of `~mpdaf.obj.Spectrum`
                       Estimated lines
        """
        self._loginfo('Step 09 - Lines estimation (grid_dxy=%d)' % (grid_dxy))
        self.param['grid_dxy'] = grid_dxy

        if self.Cat1 is None:
            raise IOError('Run the step 08 to initialize self.Cat1 catalog')

        self.Cat2, Cat_est_line_raw_T, Cat_est_line_var_T = \
            Estimation_Line(self.Cat1, self.cube_raw, self.var, self.PSF,
                            self.wfields, self.wcs, self.wave, size_grid=grid_dxy,
                            criteria='flux', order_dct=30, horiz_psf=1, horiz=5)

        self._loginfo('Purity estimation')
        self.Cat2 = Purity_Estimation(self.Cat2,
                                      [self.Pval_r, self.Pval_r_comp],
                                      [self.index_pval, self.index_pval_comp])
        _format_cat(self.Cat2, 2)
        self._loginfo('Save the updated catalogue in self.Cat2' +
                      ' (%d lines)' % len(self.Cat2))

        self.spectra = []
        for data, vari in zip(Cat_est_line_raw_T, Cat_est_line_var_T):
            spe = Spectrum(data=data, var=vari, wave=self.wave,
                           mask=np.ma.nomask)
            self.spectra.append(spe)
        self._loginfo('Save the estimated spectrum of each line in ' +
                      'self.spectra')

        self._loginfo('09 Done')

    def step10_clean_results(self, *, merge_lines_z_threshold=5,
                             spectrum_size_fwhm=3):
        """Clean the various results.

        This step does several things to “clean” the results of ORIGIN:

        - The Cat2 line table may contain several lines found at the very same
          x, y, z position in the cube. Only the line with the highest purity
          is kept in the table.
        - Some lines are associated to the same source but are very near
          considering their z positions.  The lines are all marked as merged in
          the brightest line of the group (but are kept in the line table).
        - The FITS file containing the spectra is cleaned to keep only the
          lines from the cleaned line table. The spectrum around each line
          is trimmed around the line position.
        - A table of unique sources is created.

        Attributes added to the ORIGIN object:
        - `Cat3_lines`: clean table of lines;
        - `Cat3_sources`: table of unique sources
        - `Cat3_spectra`: trimmed spectra. For a given <num_line>, the
            spectrum is in `DATA<num_line>` extension and the variance in
            the `STAT<num_line>` extension.

        Parameters
        ----------
        merge_lines_z_threshold: int
            z axis pixel threshold used when merging similar lines.
        spectrum_size_fwhm: float
            The length of the spectrum to keep around each line as a factor of
            the fitted line FWHM.

        """
        self._loginfo('Step10 - Results cleaning')

        if self.Cat2 is None:
            raise IOError('Run the step 09 to initialize self.Cat2')

        self.param['merge_lines_z_threshold'] = merge_lines_z_threshold
        self.param['spectrum_size_fwhm'] = spectrum_size_fwhm

        unique_lines = remove_identical_duplicates(self.Cat2)
        self.Cat3_lines = merge_similar_lines(unique_lines)
        self.Cat3_sources = unique_sources(self.Cat3_lines)

        self._loginfo('Save the unique source catalogue in self.Cat3_sources'
                      ' (%d lines)', len(self.Cat3_sources))
        self._loginfo('Save the cleaned lines in self.Cat3_lines (%d lines)',
                      len(self.Cat3_lines))

        self.Cat3_spectra = trim_spectrum_list(
            self.Cat3_lines, self.spectra, self.FWHM_profiles,
            size_fwhm=spectrum_size_fwhm)

        self._loginfo('Step 10 - Done')

    def step11_create_masks(self, path=None, overwrite=True, mask_size=50,
                            seg_thres_factor=.5):
        """Create source masks and sky masks.

        This step create the mask and sky mask for each source.

        Parameters
        ----------
        path : str
            Path where the masks will be saved.
        overwrite : bool
            Overwrite the folder if it already exists
        mask_size: int
            Widht in pixel for the square masks.
        seg_thres_factor: float
            Factor applied to the detection threshold to get the threshold used
            for mask creation.
        """
        if self.Cat3_lines is None:
            raise IOError('Run the step 10.')

        self._loginfo('Step11 - Mask creation')

        self.param['mask_size'] = mask_size
        self.param['seg_thres_factor'] = seg_thres_factor

        if path is not None and not os.path.exists(path):
            raise IOError("Invalid path: {0}".format(path))

        if path is None:
            out_dir = '%s/%s/masks' % (self.path, self.name)
        else:
            path = os.path.normpath(path)
            out_dir = '%s/%s/masks' % (path, self.name)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        else:
            if overwrite:
                shutil.rmtree(out_dir)
                os.makedirs(out_dir)

        create_masks(
            line_table=self.Cat3_lines,
            source_table=self.Cat3_sources,
            profile_fwhm=self.FWHM_profiles,
            correl_cube=self.cube_correl,
            correl_threshold=self.threshold_correl,
            std_cube=self.cube_std,
            std_threshold=self.threshold_std,
            segmap=self.segmap,
            out_dir=out_dir,
            mask_size=mask_size,
            seg_thres_factor=seg_thres_factor,
            plot_problems=True)

        self._loginfo('Step11 - Mask done')

    def step12_write_sources(self, path=None, overwrite=True, fmt='default',
                             src_vers='0.1', author='undef', ncpu=1):
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
        self._loginfo('Step 12 - Sources creation')
        self._loginfo('Add RA-DEC to the catalogue')
        if self.Cat1 is None:
            raise IOError('Run the step 09 to initialize self.Cat2')

        # path
        if path is not None and not os.path.exists(path):
            raise IOError("Invalid path: {0}".format(path))

        if path is None:
            path_src = '%s/%s/sources' % (self.path, self.name)
            catname = '%s/%s/%s.fits' % (self.path, self.name, self.name)
        else:
            path = os.path.normpath(path)
            path_src = '%s/%s/sources' % (path, self.name)
            catname = '%s/%s/%s.fits' % (path, self.name, self.name)

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
            raise IOError('Run the step 09 to initialize self.spectra')
        nsources = Construct_Object_Catalogue(self.Cat2, self.spectra,
                                              self.cube_correl,
                                              self.wave, self.FWHM_profiles,
                                              path_src, self.name, self.param,
                                              src_vers, author,
                                              self.path, self.maxmap,
                                              self.segmap,
                                              ncpu)

        # create the final catalog
        self._loginfo('Create the final catalog- %d sources' % nsources)
        catF = Catalog.from_path(path_src, fmt='working')
        catF.write(catname, overwrite=overwrite)

        self._loginfo('12 Done')

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

        self.segmap.plot(ax=ax)

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
            bounds = np.linspace(i0, i1 + 1, n + 1) - 0.5
            norm = BoundaryNorm(bounds, n + 1)
            divider = make_axes_locatable(ax)
            cax2 = divider.append_axes("right", size="5%", pad=1)
            plt.colorbar(cax, cax=cax2, cmap=kwargs['cmap'], norm=norm,
                         spacing='proportional', ticks=bounds + 0.5,
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
        ncol      : int
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
        if self.nbAreas is None:
            raise IOError('Run the step 02 to initialize self.nbAreas')

        if fig is None:
            fig = plt.figure()

        if self.nbAreas <= ncol:
            n = 1
            m = self.nbAreas
        else:
            n = self.nbAreas // ncol
            m = ncol
            if (n * m) < self.nbAreas:
                n = n + 1

        for area in range(1, self.nbAreas + 1):
            if area == 1:
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
        if self.nbAreas is None:
            raise IOError('Run the step 02 to initialize self.nbAreas')
        if self.thresO2 is None:
            raise IOError('Run the step 03 to compute the threshold values')
        if ax is None:
            ax = plt.gca()
        ax.plot(np.arange(1, self.nbAreas + 1), self.thresO2, '+')
        med = np.median(self.thresO2)
        diff = np.absolute(self.thresO2 - med)
        mad = np.median(diff)
        if mad != 0:
            ksel = (diff / mad) > cutoff
            if ksel.any():
                ax.plot(np.arange(1, self.nbAreas + 1)[ksel],
                        np.asarray(self.thresO2)[ksel], 'ro')
        ax.set_xlabel('area')
        ax.set_ylabel('Threshold')
        ax.set_title('PCA threshold (med=%.2f, mad= %.2f)' % (med, mad))

    def plot_PCA_threshold(self, area, pfa_test='step03', log10=False,
                           legend=True, xlim=None, ax=None):
        """ Plot the histogram and the threshold for the starting point of the
        PCA

        Parameters
        ----------
        area      : int in [1, nbAreas]
                    Area ID
        pfa_test  : float or str
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
        if self.nbAreas is None:
            raise IOError('Run the step 02 to initialize self.nbAreas')

        if pfa_test == 'step03':
            if 'pfa_test' in self.param:
                pfa_test = self.param['pfa_test']
                hist = self.histO2[area - 1]
                bins = self.binO2[area - 1]
                thre = self.thresO2[area - 1]
                mea = self.meaO2[area - 1]
                std = self.stdO2[area - 1]
            else:
                raise IOError('pfa_test param is None: set a value or run' +
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
        gauss *= hist.max() / gauss.max()

        if log10:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gauss = np.log10(gauss)
                hist = np.log10(hist)

        ax.plot(center, hist, '-k')
        ax.plot(center, hist, '.r')
        ax.plot(center, gauss, '-b', alpha=.5)
        ax.axvline(thre, color='b', lw=2, alpha=.5)
        ax.grid()
        if xlim is None:
            ax.set_xlim((center.min(), center.max()))
        else:
            ax.set_xlim(xlim)
        ax.set_xlabel('frequency')
        ax.set_ylabel('value')
        if legend:
            ax.text(0.1, 0.8, 'zone %d\npfa %.2f\nthreshold %.2f' % (area,
                                                                     pfa_test, thre),
                    transform=ax.transAxes, bbox=dict(facecolor='red', alpha=0.5))
        else:
            ax.text(0.9, 0.9, '%d' % area, transform=ax.transAxes,
                    bbox=dict(facecolor='red', alpha=0.5))

    def plot_mapPCA(self, area=None, iteration=None, ax=None, **kwargs):
        """ Plot at a given iteration (or at the end) the number of times
        a spaxel got cleaned by the PCA

        Parameters
        ----------
        area: int in [1, nbAreas]
                if None draw the full map for all areas
        iteration : int
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
            title += '\n%d iterations' % iteration
        if area is None:
            title += ' (Full map)'
        else:
            mask = np.ones_like(self.mapO2._data, dtype=np.bool)
            mask[self.areamap._data == area] = False
            themap._mask = mask
            title += ' (zone %d)' % area

        if iteration is not None:
            themap[themap._data < iteration] = np.ma.masked

        if ax is None:
            ax = plt.gca()

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'jet'

        themap.plot(title=title, colorbar='v', ax=ax, **kwargs)

#    def plot_segmentation(self, pfa=5e-2, step=6, maxmap=True, ax=None, **kwargs):
#        """ Plot the 2D segmentation map associated to a PFA
#        This function draw the labels of the segmentation map which is computed,
#        not with the same pfa, in :
#            - the step02 to compute the automatic areas splitting for
#            the PCA
#            - the step06 to compute the threshold of the local maxima
#
#        Parameters
#        ----------
#        pfa  : float
#               Pvalue for the test which performs segmentation
#        step : int
#               The Segmentation map as used in this step: (2/6)
#        maxmap : bool
#                 If true, segmentation map is plotted as contours on the maxmap
#        ax   : matplotlib.Axes
#               The Axes instance in which the image is drawn
#        kwargs : matplotlib.artist.Artist
#                 Optional extra keyword/value arguments to be passed to
#                 the ``ax.imshow()`` function
#        """
#        if self.cont_dct is None:
#            raise IOError('Run the step 01 to initialize self.cont_dct')
#        if maxmap and self.maxmap is None:
#            raise IOError('Run the step 05 to initialize self.maxmap')
#
#        if ax is None:
#            ax = plt.gca()
#
#        if step == 2:
#            radius = 2
#            dxy = 2 * radius
#            x = np.linspace(-dxy, dxy, 1 + (dxy) * 2)
#            y = np.linspace(-dxy, dxy, 1 + (dxy) * 2)
#            xv, yv = np.meshgrid(x, y)
#            r = np.sqrt(xv**2 + yv**2)
#            disk = (np.abs(r) <= radius)
#            mask = disk
#        elif step == 6:
#            mask = None
#        else:
#            raise IOError('sept must be equal to 2 or 6')
#
#        map_in = Segmentation(self.segmentation_test.data, pfa, mask=mask)
#
#        if maxmap:
#            self.maxmap[self.maxmap._data == 0] = np.ma.masked
#            self.maxmap.plot(ax=ax, **kwargs)
#            ax.contour(map_in, [0], origin='lower', cmap='Greys')
#        else:
#            ima = Image(data=map_in, wcs=self.wcs)
#            if 'cmap' not in kwargs:
#                kwargs['cmap'] = 'jet'
#            ima.plot(title='Labels of segmentation, pfa: %f' % (pfa), ax=ax,
#                     **kwargs)

    def plot_purity(self, comp=False, ax=None, log10=False):
        """Draw number of sources per threshold computed in step06/step08

        Parameters
        ----------
        comp : bool
               If True, plot purity curves for the complementary lines (step08)
        ax : matplotlib.Axes
             The Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """

        if self.Det_M is None:
            raise IOError('Run the step 06')

        if ax is None:
            ax = plt.gca()

        if comp:
            threshold = self.param['threshold2']
            Pval_r = self.Pval_r_comp
            index_pval = self.index_pval_comp
            purity = self.param['purity2']
            Det_M = self.Det_M_comp
            Det_m = self.Det_m_comp
        else:
            threshold = self.param['threshold']
            Pval_r = self.Pval_r
            index_pval = self.index_pval
            purity = self.param['purity']
            Det_M = self.Det_M
            Det_m = self.Det_m

        ax2 = ax.twinx()
        if log10:
            ax2.semilogy(index_pval, Pval_r, 'y.-', label='purity')
            ax.semilogy(index_pval, Det_M, 'b.-',
                        label='n detections (+DATA)')
            ax.semilogy(index_pval, Det_m, 'g.-',
                        label='n detections (-DATA)')
            ax2.semilogy(threshold, purity, 'xr')

        else:
            ax2.plot(index_pval, Pval_r, 'y.-', label='purity')
            ax.plot(index_pval, Det_M, 'b.-', label='n detections (+DATA)')
            ax.plot(index_pval, Det_m, 'g.-', label='n detections (-DATA)')
            ax2.plot(threshold, purity, 'xr')

        ym, yM = ax.get_ylim()
        ax.plot([threshold, threshold], [ym, yM], 'r', alpha=.25, lw=2,
                label='automatic threshold')

        ax.set_ylim((ym, yM))
        ax.set_xlabel('Threshold')
        ax2.set_ylabel('Purity')
        ax.set_ylabel('Number of detections')
        ax.set_title('threshold %f' % threshold)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc=2)

    def plot_NB(self, src_ind, ax1=None, ax2=None, ax3=None):
        """Plot the narrow bands images

        src_ind : int
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
            ax1 = plt.subplot(1, 3, 1)
            ax2 = plt.subplot(1, 3, 2)
            ax3 = plt.subplot(1, 3, 3)

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
