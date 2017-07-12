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
from .lib_origin import Spatial_Segmentation, Correlation_GLR_test, \
    Construct_Object_Catalogue, dct_residual, Compute_Standardized_data, \
    O2test,Compute_GreedyPCA_SubCube, init_calibrators, add_calibrator, \
    Compute_local_max_zone, Create_local_max_cat, \
    Estimation_Line, SpatioSpectral_Merging, Segmentation, \
    Spatial_Merging_Circle, Correlation_GLR_test_zone, \
    Compute_thresh_PCA_hist, Compute_threshold, Threshold_pval, \
    Compute_threshold_area, __version__


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
        NbSubcube          : integer
                             Number of sub-cubes for the spatial segmentation 
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
        intx               : array
                             Limits in pixels of the columns for each zone.
        inty               : array
                             Limits in pixels of the rows for each zone.
        cube_faint         : `~mpdaf.obj.Cube`
                             Projection on the eigenvectors associated to the
                             lower eigenvalues of the data cube (representing
                             the faint signal). Result of step02_compute_PCA.
        cube_correl        : `~mpdaf.obj.Cube`
                             Cube of T_GLR values. Result of
                             step03_compute_TGLR. From Max correlations
        cube_correl_min    : `~mpdaf.obj.Cube`
                             Cube of T_GLR values. Result of
                             step03_compute_TGLR. From Min correlations                            
        cube_local_max     : `~mpdaf.obj.Cube`
                             Cube of Local maximam of T_GLR values. Result of
                             step04_compute_Local_max. From Max correlations
        cube_local_min     : `~mpdaf.obj.Cube`
                             Cube of Local maximam of T_GLR values. Result of
                             step04_compute_Local_max. From Min correlations                             
        cube_profile       : `~mpdaf.obj.Cube` (type int)
                             Number of the profile associated to the T_GLR.
                             Result of step03_compute_TGLR.
        cube_pval_correl: `~mpdaf.obj.Cube`
                             Cube of thresholded p-values associated to the
                             local maxima of T_GLR values. 
                             Result of step04_compute_Local_max.                             
        Cat0               : astropy.Table
                             Catalog returned by step04_compute_Local_max
        Cat1               : astropy.Table
                             Catalog returned by step06_compute_spectra.
        spectra            : list of `~mpdaf.obj.Spectrum`
                             Estimated lines. Result of step06_compute_spectra.
        Cat2               : astropy.Table
                             Catalog returned by step07_spatiospectral_merging.
    """
    
    def __init__(self, path, name, filename, NbSubcube, profiles, PSF,
                 FWHM_PSF, intx, inty, cube_faint, mapO2, histO2, freqO2,
                 thresO2, cube_correl, maxmap, cube_profile, Cat0, Cat1, spectra,
                 Cat2, param, cube_std, var, expmap, cube_pval_correl,
                 cube_local_max, cont_dct, segmentation_map, cube_local_min,
                 cube_correl_min):
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
        if NbSubcube is None:
            NbSubcube = max(1, max(self.Nx, self.Ny)//100)
        self.param['nbsubcube'] = NbSubcube
        self.NbSubcube = NbSubcube
        
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
        
        
        # Spatial segmentation
        if intx is None or inty is None:
            self._log_stdout.info('Spatial segmentation')
            self.inty, self.intx = Spatial_Segmentation(self.Nx, self.Ny,
                                                    NbSubcube)
        else:                
            self.inty = inty
            self.intx = intx
        self.param['intx'] = self.intx.tolist()
        self.param['inty'] = self.inty.tolist()
        
        # step1
        self.cube_std = cube_std
        self.cont_dct = cont_dct        
        # step2
        self.cube_faint = cube_faint
        self.mapO2 = mapO2
        self.histO2 = histO2
        self.freqO2 = freqO2
        self.thresO2 = thresO2
        # step3
        self.cube_correl = cube_correl
        self.cube_correl_min = cube_correl_min        
        self.cube_local_max = cube_local_max     
        self.cube_local_min = cube_local_min             
        self.cube_profile = cube_profile
        self.maxmap = maxmap
        # step4
        self.cube_pval_correl = cube_pval_correl
        self.Cat0 = Cat0
        # step5
        self.Cat1 = Cat1
        self.spectra = spectra
        # step6
        self.segmentation_map = segmentation_map
        self.Cat2 = Cat2
        
        self._log_file.info('00 Done')
        
    @classmethod
    def init(cls, cube, NbSubcube=None, profiles=None, 
                 PSF=None, FWHM_PSF=None, name='origin'):
        # NbSubcube None par défaut et sous-cube de 80-100
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
        NbSubcube   : integer
                      Number of sub-cubes for the spatial segmentation
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
        return cls(path='.',  name=name, filename=cube, NbSubcube=NbSubcube,
                   profiles=profiles, PSF=PSF, FWHM_PSF=FWHM_PSF, intx=None,
                   inty=None, cube_faint=None, mapO2=None, histO2=None,
                   freqO2=None, thresO2=None, cube_correl=None, maxmap=None,
                   cube_profile=None, Cat0=None,
                   Cat1=None, spectra=None, Cat2=None, param=None,
                   cube_std=None, var=None, expmap=None,
                   cube_pval_correl=None,cube_local_max=None,cont_dct=None,
                   segmentation_map=None,cube_local_min=None,
                   cube_correl_min=None)
        
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

        intx = np.asarray(param['intx'])
        inty = np.asarray(param['inty'])
        NbSubcube = param['nbsubcube']
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
        if os.path.isfile('%s/cube_faint.fits'%folder):
            cube_faint = Cube('%s/cube_faint.fits'%folder)
        else:
            cube_faint = None
        if os.path.isfile('%s/mapO2_%d_%d.fits'%(folder, NbSubcube-1,
                                                 NbSubcube-1)):
            mapO2 = {}
            for i in range(NbSubcube):
                for j in range(NbSubcube):
                    mapO2[(i,j)] = Image('%s/mapO2_%d_%d.fits'%(folder,
                                                                      i,j))
        else:
            mapO2 = None
        if os.path.isfile('%s/histO2_%d_%d.txt'%(folder, NbSubcube-1,
                                                 NbSubcube-1)):
            histO2 = {}
            for i in range(NbSubcube):
                for j in range(NbSubcube):
                    histO2[(i,j)] = np.loadtxt('%s/histO2_%d_%d.txt'%(folder,
                                                                      i,j))
        else:
            histO2 = None
        if os.path.isfile('%s/freqO2_%d_%d.txt'%(folder, NbSubcube-1,
                                                 NbSubcube-1)):
            freqO2 = {}
            for i in range(NbSubcube):
                for j in range(NbSubcube):
                    freqO2[(i,j)] = np.loadtxt('%s/freqO2_%d_%d.txt'%(folder,
                                                                      i,j))
        else:
            freqO2 = None
#        if os.path.isfile('%s/thresO2_%d_%d.txt'%(folder, NbSubcube-1,
#                                                 NbSubcube-1)):
#            thresO2 = {}
#            for i in range(NbSubcube):
#                for j in range(NbSubcube):
#                    thresO2[(i,j)] = np.loadtxt('%s/thresO2_%d_%d.txt'%(folder,
#                                                                      i,j))
#        else:
#            thresO2 = None            

        if os.path.isfile('%s/thresO2.txt'%(folder)):
            thresO2 = np.loadtxt('%s/thresO2.txt'%(folder)).\
            reshape((NbSubcube, NbSubcube)).astype(np.float)
        else:
            thresO2 = None
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
        if os.path.isfile('%s/cube_pval_correl.fits'%folder):
            cube_pval_correl = Cube('%s/cube_pval_correl.fits'%folder,
                                    mask=np.ma.nomask, dtype=np.float64)            
        else:
            cube_pval_correl = None
        if os.path.isfile('%s/Cat0.fits'%folder):
            Cat0 = Table.read('%s/Cat0.fits'%folder)
        else:
            Cat0 = None
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
        if os.path.isfile('%s/Cat2.fits'%folder):
            Cat2 = Table.read('%s/Cat2.fits'%folder)
        else:
            Cat2 = None
        if os.path.isfile('%s/segmentation_map.fits'%folder):
            segmentation_map = Image('%s/segmentation_map.fits'%folder)
        else:
            segmentation_map = None
            
        if newpath is not None:
            path = newpath
        if newname is not None:
            name = newname
                
        return cls(path=path,  name=name, filename=param['cubename'],
                   NbSubcube=NbSubcube,
                   profiles=param['profiles'], PSF=PSF, FWHM_PSF=FWHM_PSF,
                   intx=intx, inty=inty, cube_std=cube_std, var=var,
                   cube_faint=cube_faint, mapO2=mapO2, histO2=histO2,
                   freqO2=freqO2, thresO2=thresO2, cube_correl=cube_correl,
                   maxmap=maxmap, cube_profile=cube_profile,
                   Cat0=Cat0, Cat1=Cat1,
                   spectra=spectra, Cat2=Cat2, param=param,
                   expmap=expmap, cube_pval_correl=cube_pval_correl,
                   cube_local_max=cube_local_max, cont_dct=cont_dct,
                   segmentation_map=segmentation_map,
                   cube_local_min=cube_local_min, 
                   cube_correl_min=cube_correl_min)
                   
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
        #step2
        if self.histO2 is not None:
            for i in range(self.NbSubcube):
                for j in range(self.NbSubcube):
                    np.savetxt('%s/histO2_%d_%d.txt'%(path2, i,j),
                               self.histO2[(i,j)])
        if self.freqO2 is not None:
            for i in range(self.NbSubcube):
                for j in range(self.NbSubcube):
                    np.savetxt('%s/freqO2_%d_%d.txt'%(path2, i,j),
                               self.freqO2[(i,j)])
#        if self.thresO2 is not None:
#            for i in range(self.NbSubcube):
#                for j in range(self.NbSubcube):
#                    np.savetxt('%s/thresO2_%d_%d.txt'%(path2, i,j),
#                               self.thresO2[(i,j)])                    
        if self.thresO2 is not None:
            np.savetxt('%s/thresO2.txt'%path2, self.thresO2)
        if self.mapO2 is not None:
            for i in range(self.NbSubcube):
                for j in range(self.NbSubcube):
                    self.mapO2[(i,j)].write('%s/mapO2_%d_%d.fits'%(path2, i,j))
        if self.cube_faint is not None:
            self.cube_faint.write('%s/cube_faint.fits'%path2)
        # step3
        if self.cube_correl is not None:
            self.cube_correl.write('%s/cube_correl.fits'%path2)
        if self.cube_correl_min is not None:
            self.cube_correl_min.write('%s/cube_correl_min.fits'%path2)            
        if self.cube_profile is not None:
            self.cube_profile.write('%s/cube_profile.fits'%path2)
        if self.maxmap is not None:
            self.maxmap.write('%s/maxmap.fits'%path2)
        # step4
        if self.cube_local_max is not None:
            self.cube_local_max.write('%s/cube_local_max.fits'%path2)    
        if self.cube_local_min is not None:
            self.cube_local_min.write('%s/cube_local_min.fits'%path2)                                
        if self.cube_pval_correl is not None:
            self.cube_pval_correl.write('%s/cube_pval_correl.fits'%path2)
        if self.Cat0 is not None:
            self.Cat0.write('%s/Cat0.fits'%path2, overwrite=True)
        # step5
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
        # step6
        if self.Cat2 is not None:
            self.Cat2.write('%s/Cat2.fits'%path2, overwrite=True)
        if self.segmentation_map is not None:
            self.segmentation_map.write('%s/segmentation_map.fits'%path2)        
        

    def init_calibrator(self,x=None, y=None, z=None, amp=None ,profil=None,\
                              Cat_cal=None,random=0, save=False,\
                              name='cat_cal.fits'):
        
        """ Initialise calibrators and create catalogue
        
        Parameters
        ----------
        x           :   int or list
                        the x spatiale position of the line (pixel)
        y           :   int or list
                        the y spatiale position of the line (pixel)                   
        z           :   int or list
                        the z spectrale position of the line (pixel)                    
                        
        amp         :   float 
                        if int repeated    
                        amplitude of the line
        
        profil      :   int or list
                        if int repeated
                        the number of the profile associated to the line     
                        
        Cat_cal     :   Table
                        Catalogue of Calibrators from previous use of function
                        useful to a add specific calibrator with random ones
                        string
                        if catcal='add' update of catalogue

        random      :   int
                        number of random line added to the data
                        
        save        :   bool
                        to save the catalogue of calibrator 
                        
        name        :   string
                        name of the catalogue files
                        default name is cat_cal.fits
        Returns
        -------
        self.Cat_calibrator    :    Catalogue
                                    Catalogue of calibrators
        """     
        self._log_file.info('00 - initialization of calibrators')
        self._log_stdout.info('Step 00 - initialization of calibrators')           
        
        nl,ny,nx = self.cube_raw.shape
        nprofil = len(self.profiles)
 
        if Cat_cal=='add':
            try:
                Cat_cal = self.Cat_calibrator
            except:
                self._log_stdout.info('create calibrators catalogue first')
        else:    
            if Cat_cal is not None: 
                self._log_stdout.info('update calibrators catalogue')            
        if random:
            self._log_stdout.info('add %d random calibrators'%random)
        else:
            self._log_stdout.info('add calibrators')        
            
        self.Cat_calibrator = init_calibrators(nl, ny, nx, nprofil, 
                                               x, y, z, amp, profil, random, 
                                               Cat_cal)                        
            
        if save:           
            self.Cat_calibrator.write(name, overwrite=True)
            self._log_stdout.info('Catalogue saved in file: %s'%name)  
        
 
    def add_calibrator(self, name=''):
        """ Initialise calibrators and create catalogue
        
        Parameters
        ----------
        name        :   str
                        name of the catalogue of calibrators file
                        if empty self.Cat_calibrator is used
        Returns
        -------
        self.raw    : array
                      raw data with calibrators
        """             

        self._log_file.info('00 - adding calibrators to data')
        self._log_stdout.info('Step 00 - adding calibrators to data')          
        if name:
            self.Cat_calibrator = Catalog.read(name)
            self._log_stdout.info('Catalogue read from file: %s'%name)                      
        else:
            self._log_stdout.info('Catalogue from self.Cat_calibrator')            

        self.cube_raw = add_calibrator(self.Cat_calibrator, 
                                       self.cube_raw, self.PSF, self.profiles,
                                       self.wfields, self.var)        
        
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
        self._log_stdout.info('Standard data')
        
        # compute standardized data
        cube_std, var = Compute_Standardized_data(faint_dct, self.expmap,
                                                  self.var, newvar)
        
        if newvar:        
            self._log_stdout.info('self.var is computed')   
            self.var = var
        
        self._log_stdout.info('Save the std signal in self.cube_std')        
        self.cube_std = Cube(data=cube_std, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)  
        self.cont_dct = Cube(data=cont_dct, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)
        self._log_file.info('01 Done')

    def step02_compute_greedy_PCA(self, mixing=False,
                              Noise_population=50, threshold_test=.05,
                              itermax=100):
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
                                the fraction of spectra estimated as background
                                
        threshold_test      :   float
                                the threshold of the test (default=1)  
                                
        itermax             :   integer
                                maximum iterations

        Returns
        -------
        self.cube_faint : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the lower
                     eigenvalues of the data cube
                     (representing the faint signal)
        self.mapO2 : dict(`~mpdaf.obj.Image`)
                     For each subcube, he numbers of iterations used by testO2
                     for each spaxel
        self.histO2 : dict(array)
                      For each subcube, histogram
        self.freqO2 : dict(array)
                      For each subcube, frequency
        self.thresO2 : array(NbSubcube, NbSubcube)
                       For each subcube, Treshold value
        """
        self._log_file.info('02 - greedy PCA computation:')
        
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
        
        self._log_file.info('   - Noise_population=%0.2f'%Noise_population)
        self._log_file.info('   - threshold_test=%0.2f'%threshold_test)            
        self.param['Noise_population'] = Noise_population
        self.param['threshold_test'] = threshold_test        
        self._log_stdout.info('Step 02 - greedy PCA computation')                
        self._log_stdout.info('Compute greedy PCA on each zone')  
        
        
        faint, mapO2, self.histO2, self.freqO2, self.thresO2 = \
        Compute_GreedyPCA_SubCube(self.NbSubcube, self.cube_std._data,
                                  self.intx, self.inty, 
                                  Noise_population, threshold_test,itermax)
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
        self.mapO2 = {}
        for numy in range(self.NbSubcube):
            for numx in range(self.NbSubcube):
                self.mapO2[(numx, numy)] = Image(data=mapO2[(numx, numy)],
                                                    wcs=self.wcs)    
        self._log_file.info('02 Done')        
     

    def step03_compute_TGLR(self, area=False):
        """Compute the cube of GLR test values.
        The test is done on the cube containing the faint signal
        (self.cube_faint) and it uses the PSF and the spectral profile.

        Parameters
        ----------                    
        area              :   bool
                              if True, The correlation is performed on smaller 
                              subcube and combined after. Useful to avoid swapp                                                            
                                
        Returns
        -------
        self.cube_correl  : `~mpdaf.obj.Cube`
                            Cube of T_GLR values
        self.cube_profile : `~mpdaf.obj.Cube` (type int)
                             Number of the profile associated to the T_GLR
        self.maxmap       : `~mpdaf.obj.Image`
                             Map of maxima along the wavelength axis
        """
        self._log_file.info('03 GLR test')
        self._log_stdout.info('Step 03 - GLR test')
        if self.cube_faint is None:
            raise IOError('Run the step 02 to initialize self.cube_faint')

        # TGLR computing (normalized correlations)           
        if 'expmap' in self.param: 
            var = self.var/self.expmap
        else:
            var = self.var
            var[self.expmap==0] = np.inf

        if area: 
            correl, profile, cm = Correlation_GLR_test_zone( \
                    self.cube_faint._data, var, self.PSF, self.wfields,
                    self.profiles, self.intx, self.inty, self.NbSubcube)  
            
        else:                          
            correl, profile, cm = Correlation_GLR_test(self.cube_faint._data, 
                                            var, self.PSF, self.wfields,
                                               self.profiles)                                                               
        
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
                       
        self._log_file.info('03 Done')


    def step04_compute_local_max(self, neighboors=26):
        """Loop on each zone of self.cube_correl and compute for each zone:
        
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
        neighboors : integer
                     Connectivity of contiguous voxels
                            
        Returns
        -------
        self.cube_local_max    : `~mpdaf.obj.Cube`
                                 Local maxima from max correlation
        self.cube_local_min    : `~mpdaf.obj.Cube`
                                 Local maxima from minus min correlation                                 
        self.cube_pval_correl  : `~mpdaf.obj.Cube`
                                 Cube of thresholded p-values associated
                                 to the local max of T_GLR values
        """
        self._log_stdout.info('Step 04 - Local maximum and p-values computation')
        self._log_stdout.info('Compute p-values of local maximum of correlation values')
        if self.cube_correl is None:
            raise IOError('Run the step 03 to initialize self.cube_correl')

        cube_local_max, cube_local_min = Compute_local_max_zone(
                                                    self.cube_correl._data,
                                                    self.cube_correl_min._data,
                                                    self.expmap==0,
                                                    self.intx, self.inty,
                                                    self.NbSubcube,
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

    def step05_threshold_pval(self, fidelity=.9, threshold_add=0, area = True):
        """Threshold the Pvalue with the given threshold, if the threshold is
        None the threshold is automaticaly computed from confidence applied
        on local maximam from maximum correlation and local maxima from 
        minus minimum correlation

        Parameters
        ----------
        fidelity : float
                   fidelity to automatically compute the threshold        
        threshold_add : float
                     additional value added to Threshold applied on pvalues.
                            
        Returns
        -------                               
        self.cube_pval_correl  : `~mpdaf.obj.Cube`
                                 Cube of thresholded p-values associated
                                 to the local max of T_GLR values
        self.Cat0 : astropy.Table
                    Catalogue of the referent voxels for each group.
                    Columns: x y z ra dec lbda T_GLR profile pvalC pvalS pvalF
                    Coordinates are in pixels.
        """
        self._log_stdout.info('Step 05 - p-values Thresholding')
        self._log_stdout.info('Threshold the Pvalues')
        self._log_file.info('   computation of threshold with fidelity =%.1f'%fidelity)
        self.param['fidelity'] = fidelity
        self.param['threshold_add'] = threshold_add            
        if self.cube_local_max is None:
            raise IOError('Run the step 04 to initialize self.cube_local_max and self.cube_local_min')
        
        if area: 
            threshold, Pval_M, Pval_m, Pval_r, index_pval, fid_ind, \
            cube_pval_correl, mapThresh = Compute_threshold_area(
                                               fidelity, 
                                               self.cube_local_max.data,
                                               self.cube_local_min.data,
                                               threshold_add, 
                                               self.intx, self.inty,
                                               self.NbSubcube)
            self.mapThresh = mapThresh
        else:
            threshold, Pval_M, Pval_m, Pval_r, index_pval, fid_ind = \
                                               Compute_threshold(
                                               fidelity, 
                                               self.cube_local_max.data,
                                               self.cube_local_min.data)
            
            threshold+=threshold_add
            
            cube_pval_correl = Threshold_pval(self.cube_local_max.data, \
                                              threshold)
        
        
        
        self.param['ThresholdPval'] = threshold
        self.param['Pval_M'] = Pval_M
        self.param['Pval_m'] = Pval_m
        self.param['Pval_r'] = Pval_r
        self.param['index_pval'] = index_pval
        self.param['fid_ind'] = fid_ind
        
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
            
        self.Cat1, Cat_est_line_raw_T, Cat_est_line_var_T = \
        Estimation_Line(self.Cat0, self.cube_raw, var, self.PSF, \
                     self.wfields, self.wcs, self.wave, size_grid = grid_dxy, \
                     criteria = 'flux', order_dct = 30, horiz_psf = 1, \
                     horiz = 5)
            
        self._log_stdout.info('Save the updated catalogue in self.Cat1 (%d lines)'%len(self.Cat1))
        self.spectra = [] 
        for data, vari in zip(Cat_est_line_raw_T, Cat_est_line_var_T): 
            spe = Spectrum(data=data, var=vari, wave=self.wave,
                           mask=np.ma.nomask)
            self.spectra.append(spe)
        self._log_stdout.info('Save the estimated spectrum of each line in self.spectra')
        self._log_file.info('06 Done')       

    def step07_spatiospectral_merging(self, deltaz=20, pfa=5e-2):
        """Construct a catalogue of sources by spatial merging of the
        detected emission lines in a circle with a diameter equal to
        the mean over the wavelengths of the FWHM of the FSF.
        Then, merge the detected emission lines distants in an estimated source 
        area.

        Parameters
        ----------
        deltaz : integer
                 Distance maximum between 2 different lines

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
        self.param['pfa'] = pfa        
        if self.Cat1 is None:
            raise IOError('Run the step 05 to initialize self.Cat1')
        cat = Spatial_Merging_Circle(self.Cat1, fwhm, self.wcs)
        self.Cat2, segmentation_map = SpatioSpectral_Merging(cat, \
                                           self.cube_correl.data, \
                                           self.cont_dct.data, \
                                           self.var, deltaz, pfa)
        self.segmentation_map = Image(data=segmentation_map,
                                    wcs=self.wcs, mask=np.ma.nomask)
        self._log_stdout.info('Save the segmentation map in self.segmentation_map')        
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
        sources : mpdaf.sdetect.SourceList
                  List of sources
        """
        self._log_file.info('08 Sources creation')
        # Add RA-DEC to the catalogue
        self._log_stdout.info('Step 08 - Sources creation')
        self._log_stdout.info('Add RA-DEC to the catalogue')
        if self.Cat2 is None:
            raise IOError('Run the step 05 to initialize self.Cat2')

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
            raise IOError('Run the step 03 to initialize self.cube_correl')
        if self.spectra is None:
            raise IOError('Run the step 06 to initialize self.spectra')
        nsources = Construct_Object_Catalogue(self.Cat2, self.spectra,
                                              self.cube_correl,
                                              self.wave, self.FWHM_profiles,
                                              path_src, self.name, self.param,
                                              src_vers, author,
                                              self.path, self.maxmap,
                                              self.segmentation_map, ncpu)
                                              
        # create the final catalog
        self._log_stdout.info('Create the final catalog')
        catF = Catalog.from_path(path_src, fmt='working')
        catF.write(catname, overwrite=overwrite)
                      
        self._log_file.info('08 Done')

        return nsources

    def plot_segmentation(self, pfa=5e-2, ax=None):
        """ Plot the 2D segmentation map associated to a PFA
        
        Parameters
        ----------
        i: integer in [0, NbSubCube[
           x-coordinate of the zone
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """
        if self.cont_dct is None or self.var is None:
            raise IOError('Run the step 00 to initialize self.cont_dct and self.var')        
            
        if ax is None:
            ax = plt.gca()
            
        map_in = Segmentation(self.cont_dct.data, self.var, pfa)            
        
        ax.imshow(map_in,origin='lower',cmap='jet',interpolation='nearest')
        ax.set_title('Labels of segmentation, pfa: %f' %(pfa))


    def plot_step05(self, i, j, ax=None):
        """Draw number of sources per threshold computed in step05
        """
        if self.cube_faint is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
            
        if ax is None:
            ax = plt.gca()        
        
        threshold = self.param['ThresholdPval'][(i,j)]
        Pval_M = self.param['Pval_M'][(i,j)]
        Pval_m = self.param['Pval_m'][(i,j)]
        Pval_r = self.param['Pval_r'][(i,j)]
        index_pval = self.param['index_pval'][(i,j)]
        fid_ind = self.param['fid_ind'][(i,j)]

        
        ax.semilogy( index_pval, Pval_M, '.-', label = 'from Max Correl' )
        ax.semilogy( index_pval, Pval_m, '.-', label = 'from -Min Correl' )
        ym,yM = ax.get_ylim()
        ax.semilogy( index_pval, Pval_r, '.-', label = 'estimated fidelity' )
        ax.plot([threshold,threshold],[ym,yM],'r', alpha=.25, lw=2 , \
                 label='automatic threshold' )
        ax.plot(threshold, Pval_r[fid_ind],'xr')
        ax.set_ylim((ym,yM))
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Fidelity')
        ax.set_title('zone (%d, %d) - threshold %f' %(i,j,threshold))
        plt.legend()  
        
    def plot_step02(self, i, j, threshold_test=.05, ax=None, log10=True):
        """ Plot the histogram and the threshold for the starting point of the 
        PCA, this version of the plot is to do before doing the PCA
        
        Parameters
        ----------
        i: integer in [0, NbSubCube[
           x-coordinate of the zone
        j: integer in [0, NbSubCube[
           y-coordinate of the zone
        threshold_test :   float
                           the pfa of the test (default=.05) 
                            
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """
        if self.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
            
        if ax is None:
            ax = plt.gca()
    
        # limits of each spatial zone
        x1 = self.intx[i]
        x2 = self.intx[i + 1]
        y2 = self.inty[j]
        y1 = self.inty[j + 1]
        # Data in this spatio-spectral zone

        test = O2test(self.cube_std.data[:, y1:y2, x1:x2])
        
        # automatic threshold computation     
        hist, bins, thre = Compute_thresh_PCA_hist(test, threshold_test)    
    
        if log10:
            hist = np.log10(hist)
        
        center = (bins[:-1] + bins[1:]) / 2
        ax.plot(center, hist,'-k')
        ax.plot(center, hist,'.r')    
        ym,yM = ax.get_ylim()
        ax.plot([thre,thre],[ym,yM],'b',lw=2,alpha=.5)
        ax.grid()
        ax.set_xlim((center.min(),center.max()))
        ax.set_title('zone (%d, %d) - threshold %f' %(i,j,thre))
        
    def plot_PCA(self, i, j, ax=None, log10=True):
        """ Plot the histogram and the threshold for the starting point of the PCA
        
        Parameters
        ----------
        i: integer in [0, NbSubCube[
           x-coordinate of the zone
        j: integer in [0, NbSubCube[
           y-coordinate of the zone
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        log10 : To draw histogram in logarithmic scale or not
        """
        if self.histO2 is None or self.freqO2 is None:
            raise IOError('Run the step 02 to initialize self.histO2 and self.freqO2')
            
        if ax is None:
            ax = plt.gca()
    
        bins = self.freqO2[(i, j)]
        hist = self.histO2[(i, j)]
        thre = self.thresO2[(i, j)]
        if log10:
            hist = np.log10(hist)
        
        center = (bins[:-1] + bins[1:]) / 2
        ax.plot(center, hist,'-k')
        ax.plot(center, hist,'.r')    
        ym,yM = ax.get_ylim()
        ax.plot([thre,thre],[ym,yM],'b',lw=2,alpha=.5)
        ax.grid()
        ax.set_xlim((center.min(),center.max()))
        ax.set_title('zone (%d, %d) - threshold %f' %(i,j,thre))
        
    def plot_mapPCA(self, i, j, ax=None, iteration=None):
        """ Plot the histogram and the threshold for the starting point of the PCA
        
        Parameters
        ----------
        i: integer in [0, NbSubCube[
           x-coordinate of the zone
        j: integer in [0, NbSubCube[
           y-coordinate of the zone
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        iteration : Display the nuisance/bacground pixels at itartion k
        """
        if self.mapO2 is None:
            raise IOError('Run the step 02 to initialize self.mapO2')
            
        if ax is None:
            ax = plt.gca()
    
        if iteration is None:
            mapO2 = self.mapO2[(i, j)].data
        else:
            mapO2 = self.mapO2[(i, j)].data>iteration
            
        cax = ax.imshow(mapO2,origin='lower',cmap='jet',interpolation='nearest')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(cax, cax=cax2)
        ax.set_title('zone (%d, %d)' %(i,j))
        
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
    

    def plot_sources(self, x, y, circle=False, vmin=0, vmax=30, title=None, ax=None):
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
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
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
        self.maxmap.plot(vmin=vmin, vmax=vmax, title=title, ax=ax)
        
    def info(self):
        """ plot information
        """
        currentlog = self._log_file.handlers[0].baseFilename
        with open(currentlog) as f:
            for line in f:
                if line.find('Done') == -1:
                    self._log_stdout.info(line)
        