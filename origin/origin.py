"""
ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes
---------------------------------------------------------

This software was initially developped by Carole Clastres, under the
supervision of David Mary (Lagrange institute, University of Nice), and it was
ported to Python by Laure Piqueras (CRAL). From November 2016 the software is
updated by Antony Schutz.

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL).

"""

import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import warnings
import yaml

from astropy.io import fits
from astropy.table import Table, MaskedColumn
from astropy.utils import lazyproperty
from collections import OrderedDict
from logging.handlers import RotatingFileHandler
from mpdaf.log import setup_logging
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.MUSE import FieldsMap, get_FSF_from_cube_keywords
from mpdaf.tools import write_hdulist_to

from . import steps
from .steps import _format_cat
from .version import __version__

CURDIR = os.path.dirname(os.path.abspath(__file__))


class ORIGIN(steps.LogMixin):
    """ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

    Oriented-object interface to run the ORIGIN software.

    An Origin object is mainly composed by:
    - cube data (raw data and covariance)
    - 1D dictionary of spectral profiles
    - MUSE PSF

    Attributes
    ----------
    path : str
        Path where the ORIGIN data will be stored.
    name : str
        Name of the session and basename for the sources.
    param : dict
        Parameters values.
    cube_raw : array (Nz, Ny, Nx)
        Raw data.
    var : array (Nz, Ny, Nx)
        Variance.
    Nx : int
        Number of columns
    Ny : int
        Number of rows
    Nz : int
        Number of spectral channels
    wcs : `mpdaf.obj.WCS`
        RA-DEC coordinates.
    wave : `mpdaf.obj.WaveCoord`
        Spectral coordinates.
    profiles : list of array
        List of spectral profiles to test
    FWHM_profiles : list
        FWHM of the profiles in pixels.
    wfields : None or list of arrays
        List of weight maps (one per fields in the case of MUSE mosaic)
        None: just one field
    PSF : array (Nz, Nfsf, Nfsf) or list of arrays
        MUSE PSF (one per field)
    FWHM_PSF : float or list of float
        Mean of the fwhm of the PSF in pixel (one per field).
    imawhite : `~mpdaf.obj.Image`
        White image
    segmap : `~mpdaf.obj.Image`
        Segmentation map
    self.cube_std : `~mpdaf.obj.Cube`
        standardized data for PCA. Result of step01.
    self.cont_dct : `~mpdaf.obj.Cube`
        DCT continuum. Result of step01.
    self.ima_std : `~mpdaf.obj.Image`
        Mean of standardized data for PCA along the wavelength axis.
        Result of step01.
    self.ima_dct : `~mpdaf.obj.Image`
        Mean of DCT continuum cube along the wavelength axis.
        Result of step01.
    nbAreas : int
        Number of area (segmentation) for the PCA computation.
        Result of step02.
    areamap : `~mpdaf.obj.Image`
        PCA area. Result of step02.
    testO2 : list of arrays (one per PCA area)
        Result of the O2 test (step03).
    histO2 : list of arrays (one per PCA area)
        PCA histogram (step03).
    binO2 : list of arrays (one per PCA area)
        Bins for the PCA histogram (step03).
    thresO2 : list of float
        For each area, threshold value (step03).
    meaO2 : list of float
        Location parameter of the Gaussian fit used to
        estimate the threshold (step03).
    stdO2 : list of float
        Scale parameter of the Gaussian fit used to
        estimate the threshold (step03).
    cube_faint : `~mpdaf.obj.Cube`
        Projection on the eigenvectors associated to
        the lower eigenvalues of the data cube
        (representing the faint signal). Result of step04.
    mapO2 : `~mpdaf.obj.Image`
        The numbers of iterations used by testO2 for
            each spaxel. Result of step04.
    cube_correl : `~mpdaf.obj.Cube`
        Cube of T_GLR values (step05).
    cube_profile : `~mpdaf.obj.Cube` (type int)
        PSF profile associated to the T_GLR (step05).
    maxmap : `~mpdaf.obj.Image`
        Map of maxima along the wavelength axis (step05).
    cube_local_max : `~mpdaf.obj.Cube`
        Local maxima from max correlation (step05).
    cube_local_min : `~mpdaf.obj.Cube`
        Local maxima from min correlation (step05).
    threshold : float
        Estimated threshold (step06).
    Pval_r : array
        Purity curves (step06).
    index_pval : array
        Indexes of the purity curves (step06).
    Det_M : list
        Number of detections in +DATA (step06).
    Det_m : list
        Number of detections in -DATA  (step06).
    Cat0 : astropy.Table
        Catalog returned by step07
    zm : array
        z-position of the detections from min correlation (step07)
    ym : array
        y-position of the detections from min correlation (step07)
    xm : array
        x-position of the detections from min correlation (step07)
    Pval_r_comp : array
        Purity curves (step08).
    index_pval_comp : array
        Indexes of the purity curves (step08).
    Det_M_comp : list
        Number of detections in +DATA (step08).
    Det_m_comp : list
        Number of detections in -DATA  (step08).
    Cat1 : astropy.Table
        Catalog returned by step08
    spectra : list of `~mpdaf.obj.Spectrum`
        Estimated lines. Result of step09.
    Cat2 : astropy.Table
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

        self.path = path
        self.name = name
        self.outpath = os.path.join(path, name)
        self.param = param or {}
        self.file_handler = None
        os.makedirs(self.outpath, exist_ok=True)

        # stdout & file logger
        setup_logging(name='origin', level=loglevel, color=logcolor,
                      fmt='%(levelname)-05s: %(message)s', stream=sys.stdout)
        self.logger = logging.getLogger('origin')
        self._setup_logfile(self.logger)
        self.param['loglevel'] = loglevel
        self.param['logcolor'] = logcolor

        self._loginfo('Step 00 - Initialization (ORIGIN v%s)', __version__)

        # -----------------------------

        self.steps = OrderedDict()
        for i, cls in enumerate(steps.pipeline, start=1):
            method = cls(self, i, self.param)
            self.steps[method.method_name] = method
            self.__dict__[method.method_name] = method

        # -----------------------------

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
        cube : str
            Cube FITS file name
        segmap : str
            Segmentation map FITS filename
        fieldmap : str
            FITS file containing the field map (mosaic)
        profiles : str
            FITS of spectral profiles
            If None, a default dictionary of 20 profiles is used.
        PSF : str
            Cube FITS filename containing a MUSE PSF per wavelength.
            If None, PSF are computed with a Moffat function
            (13x13 pixels, beta=2.6, fwhm1=0.76, fwhm2=0.66,
            lambda1=4750, lambda2=7000)
        FWHM_PSF : array (Nz)
            FWHM of the PSFs in pixels.
        name : str
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
        folder : str
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

        import pprint
        print('LOAD:')
        pprint.pprint(param)

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
        def load_spectra(filename, *, idlist=None):
            """
            If no idlist is provided, all the extensions are read as
            a succession of DATA / STAT parts or a spectrum.

            If an idlist is provided, the extensions DATA<ID> and STAT<ID> for
            each ID is read and the resulting spectrum is appended to the list.
            """
            spectra = []
            with fits.open(filename) as fspectra:
                if idlist is None:
                    idlist = np.arange(len(fspectra) // 2)

                for i in idlist:
                    spectra.append(Spectrum(hdulist=fspectra,
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
            Cat3_spectra = load_spectra('%s/Cat3_spectra.fits' % folder,
                                        idlist=Cat3_lines['num_line'])
        else:
            Cat3_spectra = None

        if newname is not None:
            name = newname

        return cls(path=path, name=name, param=param,
                   loglevel=param['loglevel'], logcolor=param['logcolor'],
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

    def _setup_logfile(self, logger):
        if self.file_handler is not None:
            # Remove the handlers before adding a new one
            self.file_handler.close()
            logger.handlers.remove(self.file_handler)

        self.logfile = os.path.join(self.outpath, self.name + '.log')
        self.file_handler = RotatingFileHandler(self.logfile, 'a', 1000000, 1)
        self.file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        self.file_handler.setFormatter(formatter)
        logger.addHandler(self.file_handler)

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
        path : str
            Path where the folder (self.name) will be stored.
        erase : bool
            Remove the folder if it exists.

        """
        self._loginfo('Writing...')

        # adapt session if path changes
        if path is not None and path != self.path:
            if not os.path.exists(path):
                raise ValueError("path does not exist: {}".format(path))
            self.path = path
            self.outpath = os.path.join(path, self.name)
            # copy logfile to the new path
            shutil.copy(self.logfile, self.outpath)
            self._setup_logfile(self.logger)

        if erase:
            shutil.rmtree(self.outpath)
        os.makedirs(self.outpath, exist_ok=True)

        # parameters in .yaml
        with open('%s/%s.yaml' % (self.outpath, self.name), 'w') as stream:
            yaml.dump(self.param, stream)

        # PSF
        if isinstance(self.PSF, list):
            for i, psf in enumerate(self.PSF):
                Cube(data=psf, mask=np.ma.nomask).write(
                    '%s' % self.outpath + '/cube_psf_%02d.fits' % i)
        else:
            Cube(data=self.PSF, mask=np.ma.nomask).write(
                '%s' % self.outpath + '/cube_psf.fits')
        if self.wfields is not None:
            for i, wfield in enumerate(self.wfields):
                Image(data=wfield, mask=np.ma.nomask).write(
                    '%s' % self.outpath + '/wfield_%02d.fits' % i)

        if self.ima_white is not None:
            self.ima_white.write('%s/ima_white.fits' % self.outpath)

        for name, step in self.steps.items():
            step.dump(self.outpath)

        # step1
        if self.ima_std is not None:
            self.ima_std.write('%s/ima_std.fits' % self.outpath)
        if self.ima_dct is not None:
            self.ima_dct.write('%s/ima_dct.fits' % self.outpath)

        # step3
        # FIXME: why not saving the 2D arrays?
        if self.nbAreas is not None:
            if self.testO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt('%s/testO2_%d.txt' % (self.outpath, area),
                               self.testO2[area - 1])
            if self.histO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt('%s/histO2_%d.txt' % (self.outpath, area),
                               self.histO2[area - 1])
            if self.binO2 is not None:
                for area in range(1, self.nbAreas + 1):
                    np.savetxt('%s/binO2_%d.txt' % (self.outpath, area),
                               self.binO2[area - 1])

        # step5
        # FIXME: why not cube.write, for NaNs ?
        if self.cube_local_max is not None:
            hdu = fits.PrimaryHDU(header=self.cube_local_max.primary_header)
            hdui = fits.ImageHDU(name='DATA',
                                 data=self.cube_local_max.data.filled(fill_value=np.nan),
                                 header=self.cube_local_max.data_header)
            hdul = fits.HDUList([hdu, hdui])
            hdul.writeto('%s/cube_local_max.fits' % self.outpath,
                         overwrite=True)
#            self.cube_local_max.write('%s/cube_local_max.fits' % self.outpath)
        if self.cube_local_min is not None:
            hdu = fits.PrimaryHDU(header=self.cube_local_min.primary_header)
            hdui = fits.ImageHDU(name='DATA',
                                 data=self.cube_local_min.data.filled(fill_value=np.nan),
                                 header=self.cube_local_min.data_header)
            hdul = fits.HDUList([hdu, hdui])
            hdul.writeto('%s/cube_local_min.fits' % self.outpath,
                         overwrite=True)
            # self.cube_local_min.write('%s/cube_local_min.fits' % self.outpath)

        # step7
        if self.det_correl_min is not None:
            np.savetxt('%s/zm.txt' % (self.outpath), self.det_correl_min[0])
            np.savetxt('%s/ym.txt' % (self.outpath), self.det_correl_min[1])
            np.savetxt('%s/xm.txt' % (self.outpath), self.det_correl_min[2])

        def save_spectra(spectra, outname, *, idlist=None):
            """
            The spectra are saved to a FITS file with two extension per
            spectrum, a DATA<ID> one and a STAT<ID> one. If no idlist is
            provided, the ID is the index of the spectrum in the spectra list.
            If an idlist is provided, the ID in the value of the list at the
            index of the spectrum.

            This is important because the ID in the extension names is the
            num_line identifying the lines.
            """
            if idlist is None:
                idlist = np.arange(len(spectra))

            hdulist = fits.HDUList([fits.PrimaryHDU()])

            for idx, spec_id in enumerate(idlist):
                hdu = spectra[idx].get_data_hdu(name='DATA%d' % spec_id,
                                                savemask='nan')
                hdulist.append(hdu)
                hdu = spectra[idx].get_stat_hdu(name='STAT%d' % spec_id)
                if hdu is not None:
                    hdulist.append(hdu)
            write_hdulist_to(hdulist, outname, overwrite=True)

        if self.spectra is not None:
            save_spectra(self.spectra, '%s/spectra.fits' % self.outpath)

        # step 10
        if self.Cat3_spectra is not None:
            save_spectra(self.Cat3_spectra,
                         '%s/Cat3_spectra.fits' % self.outpath,
                         idlist=self.Cat3_lines['num_line'])

        self._loginfo("Current session saved in %s" % self.outpath)

    def plot_areas(self, ax=None, **kwargs):
        """ Plot the 2D segmentation for PCA from self.step02_areas()
            on the test used to perform this segmentation

        Parameters
        ----------
        ax : matplotlib.Axes
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
            from matplotlib.colors import BoundaryNorm
            from mpl_toolkits.axes_grid1 import make_axes_locatable
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
        log10 : bool
            Draw histogram in logarithmic scale or not
        ncol : int
            Number of colomns in the subplots
        legend : bool
            If true, write pfa and threshold values as legend
        xlim : (float, float)
            Set the data limits for the x-axes
        fig : matplotlib.Figure
            Figure instance in which the image is drawn
        **fig_kw : matplotlib.artist.Artist
            All additional keyword arguments are passed to the figure() call.

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
        cutoff : float
            Median Absolute Deviation cutoff
        ax : matplotlib.Axes
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
        area : int in [1, nbAreas]
            Area ID
        pfa_test : float or str
            PFA of the test (if 'step03', the value set during step03
            is used)
        log10 : bool
            Draw histogram in logarithmic scale or not
        legend : bool
            If true, write pfa and threshold values as legend
        xlim : (float, float)
            Set the data limits for the x-axis
        ax : matplotlib.Axes
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
            from .lib_origin import Compute_PCA_threshold
            testO2, hist, bins, thre, mea, std = Compute_PCA_threshold(
                cube_temp, pfa_test)

        if ax is None:
            ax = plt.gca()

        from scipy import stats
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
        ax : matplotlib.Axes
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
#        pfa : float
#               Pvalue for the test which performs segmentation
#        step : int
#               The Segmentation map as used in this step: (2/6)
#        maxmap : bool
#                 If true, segmentation map is plotted as contours on the maxmap
#        ax : matplotlib.Axes
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
        log10 : bool
            To draw histogram in logarithmic scale or not

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

        Parameters
        ----------
        src_ind : int
            Index of the object in self.Cat0
        ax1 : matplotlib.Axes
            The Axes instance in which the NB image around the source is drawn
        ax2 : matplotlib.Axes
            The Axes instance in which a other NB image for check is drawn
        ax3 : matplotlib.Axes
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
        x : array
            Coordinates along the x-axis of the estimated lines
            in pixels (column).
        y : array
            Coordinates along the y-axis of the estimated lines
            in pixels (column).
        circle : bool
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
        """plot information"""
        with open(self.logfile) as f:
            for line in f:
                if line.find('Done') == -1:
                    print(line, end='')
