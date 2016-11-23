"""
ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

This software has been developped by Carole Clastres under the supervision of
David Mary (Lagrange institute, University of Nice) and ported to python by
Laure Piqueras (CRAL).

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL). Please contact
Carole for more info at carole.clastres@univ-lyon1.fr

origin.py contains an oriented-object interface to run the ORIGIN software
"""

from __future__ import absolute_import, division

from astropy.io import fits
from astropy.table import Table
import astropy.units as u
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import shutil
import sys
import yaml

from mpdaf.log import setup_logging, setup_logfile, clear_loggers
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.MUSE import FSF,FieldsMap, get_FSF_from_cube_keywords
from mpdaf.sdetect import Catalog
from .lib_origin import Spatial_Segmentation, \
    Compute_PCA_SubCube, Compute_Number_Eigenvectors_Zone, \
    Compute_Proj_Eigenvector_Zone, Correlation_GLR_test, \
    Compute_pval_correl_zone, Compute_pval_channel_Zone, \
    Compute_pval_final, Compute_Connected_Voxel, \
    Compute_Referent_Voxel, Narrow_Band_Test, \
    Narrow_Band_Threshold, Estimation_Line, \
    Spatial_Merging_Circle, Spectral_Merging, \
    Construct_Object_Catalogue
    
__version__ ='1.0'


class ORIGIN(object):
    """ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

       This software has been developped by Carole Clastres under the
       supervision of David Mary (Lagrange institute, University of Nice).

       The project is funded by the ERC MUSICOS (Roland Bacon, CRAL).
       Please contact Carole for more info at carole.clastres@univ-lyon1.fr

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
        Edge_xmin          : int
                             Minimum limits along the x-axis in pixel
                             of the data cube taken to compute p-values
        Edge_xmax          : int
                             Maximum limits along the x-axis in pixel
                             of the data cube taken to compute p-values
        Edge_ymin          : int
                             Minimum limits along the y-axis in pixel
                             of the data cube taken to compute p-values
        Edge_ymax          : int
                             Maximum limits along the y-axis in pixel
                             of the data cube taken to compute p-values
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
        eig_val            : dict
                             Eigenvalues of each spatio-spectral zone.
                             Result of step01_compute_PCA.
        nbkeep             : array
                             Number of eigenvalues for each zone used to
                             compute the projection. Result of
                             step01_compute_PCA.
        cube_faint         : `~mpdaf.obj.Cube`
                             Projection on the eigenvectors associated to the
                             lower eigenvalues of the data cube (representing
                             the faint signal). Result of step01_compute_PCA.
        cube_cont          : `~mpdaf.obj.Cube`
                             Projection on the eigenvectors associated to the
                             higher eigenvalues of the data cube (representing
                             the continuum). Result of step01_compute_PCA.
        cube_correl        : `~mpdaf.obj.Cube`
                             Cube of T_GLR values. Result of
                             step02_compute_TGLR.
        cube_profile       : `~mpdaf.obj.Cube` (type int)
                             Number of the profile associated to the T_GLR.
                             Result of step02_compute_TGLR.
        cube_pval_correl   : `~mpdaf.obj.Cube`
                             Cube of thresholded p-values associated to the
                             T_GLR values. Result of step03_compute_pvalues.
        scube_pval_channel : `~mpdaf.obj.Cube`
                             Cube of p-values associated to the number of
                             thresholded p-values of the correlations per
                             spectral channel for each zone. Result of
                             step03_compute_pvalues.
        scube_pval_final   : `~mpdaf.obj.Cube`
                             Cube of final thresholded p-values. Result of
                             step03_compute_pvalues.
        Cat0               : astropy.Table
                             Catalog returned by step04_compute_ref_pix
        Cat1               : astropy.Table
                             Catalog returned by step05_compute_NBtests
        Cat1_T1            : astropy.Table
                             Catalog corresponding to the first test of 
                             step06_select_NBtests.
        Cat1_T2            : astropy.Table
                             Catalog corresponding to the second test of 
                             step06_select_NBtests.
        Cat2               : astropy.Table
                             Catalog returned by step07_compute_spectra.
        spectra            : list of `~mpdaf.obj.Spectrum`
                             Estimated lines. Result of step07_compute_spectra.
        Cat3               : astropy.Table
                             Catalog returned by step08_spatial_merging.
        Cat4               : astropy.Table
                             Catalog returned by step09_spectral_merging.
    """
    
    def __init__(self, path, name, filename, NbSubcube, margins, profiles,
                 PSF, FWHM_PSF, intx, inty, cube_faint, cube_cont, cube_correl,
                 cube_profile, cube_pval_correl, cube_pval_channel,
                 cube_pval_final, Cat0, Cat1, Cat1_T1, Cat1_T2, Cat2, spectra,
                 Cat3, Cat4, param, eig_val, nbkeep):
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
        # variance - set to Inf the Nan
        self.var = cub.var
        self.var[np.isnan(self.var)] = np.inf
        # RA-DEC coordinates
        self.wcs = cub.wcs
        # spectral coordinates
        self.wave = cub.wave
        # Dimensions
        self.Nz, self.Ny, self.Nx = cub.shape
        
        # ORIGIN parameters
        self.param['nbsubcube'] = NbSubcube
        self.param['margin'] = margins
        self.NbSubcube = NbSubcube
        self.Edge_xmin = margins[2]
        self.Edge_xmax = self.Nx - margins[3]
        self.Edge_ymin = margins[0]
        self.Edge_ymax = self.Ny - margins[1]
        
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
                    for i in range(1, nfields+1):
                        # Normalization 
                        PSF = PSF / np.sum(PSF, axis=(1, 2))[:, np.newaxis,
                                                             np.newaxis]
                        self.PSF.append(PSF[i] / np.sum(PSF[i], axis=(1, 2))\
                                                    [:, np.newaxis,np.newaxis])
                        # mean of the fwhm of the FSF in pixel
                        self.FWHM_PSF.append(np.mean(fwhm_pix[i]))
                    fmap = FieldsMap(filename, extname='FIELDMAP')
                    # weighted field map
                    self.wfields = fmap.compute_weights()
            else:
                self.param['PSF'] = 'MOFFAT1'
                FSF_model = FSF('MOFFAT1')
                beta = 2.6
                a = 0.97
                b = -4.4e-5
                PSF, fwhm_pix, fwhm_arcsec = \
                    FSF_model.get_FSF_cube(cub, Nfsf, beta=beta, a=a, b=b)
                # Normalization
                self.PSF = PSF / np.sum(PSF, axis=(1, 2))[:, np.newaxis,
                                                                 np.newaxis]
                # mean of the fwhm of the FSF in pixel
                self.FWHM_PSF = np.mean(fwhm_pix)

        else:
            self.param['PSF'] = PSF
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
        self.eig_val = eig_val
        self.nbkeep = nbkeep
        self.cube_faint = cube_faint
        self.cube_cont = cube_cont
        
        # step2
        self.cube_correl = cube_correl
        self.cube_profile = cube_profile
        
        # step3
        self.cube_pval_correl = cube_pval_correl
        self.cube_pval_channel = cube_pval_channel
        self.cube_pval_final = cube_pval_final
        
        # step4
        self.Cat0 = Cat0
        
        # step5
        self.Cat1 = Cat1
        
        # step6
        self.Cat1_T1 = Cat1_T1
        self.Cat1_T2 = Cat1_T2
        
        # step7
        self.Cat2 = Cat2
        self.spectra = spectra
        
        # step8
        self.Cat3 = Cat3
        
        # step9
        self.Cat4 = Cat4
        
        self._log_file.info('00 Done')
        
    @classmethod
    def init(cls, cube, NbSubcube, margins, profiles=None,
                 PSF=None, FWHM_PSF=None, name='origin'):
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
        margin      : (int, int, int, int)
                      Size in pixels of the margins
                      at the bottom, top, left and rigth  of the data cube.
                      (ymin, Ny-ymax, xmin, Nx-xmax)
                      Pixels in margins will not be taken
                      into account to compute p-values.
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
                   margins=margins, profiles=profiles, PSF=PSF,
                   FWHM_PSF=FWHM_PSF, intx=None, inty=None, cube_faint=None,
                   cube_cont=None, cube_correl=None, cube_profile=None,
                   cube_pval_correl=None, cube_pval_channel=None,
                   cube_pval_final=None, Cat0=None, Cat1=None, Cat1_T1=None,
                   Cat1_T2=None, Cat2=None, spectra=None, Cat3=None, Cat4=None,
                   param=None, eig_val=None, nbkeep=None)
        
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

        if os.path.isfile(param['PSF']):
            PSF = param['PSF']
            FWHM_PSF = np.asarray(param['FWHM PSF'])
        else:
            PSF = None
            FWHM_PSF = None
            
        intx = np.asarray(param['intx'])
        inty = np.asarray(param['inty'])
        NbSubcube = param['nbsubcube']
        if os.path.isfile('%s/eigval_%d_%d.txt'%(folder, NbSubcube-1,
                                                 NbSubcube-1)):
            eig_val = {}
            for i in range(NbSubcube):
                for j in range(NbSubcube):
                    eig_val[(i,j)] = np.loadtxt('%s/eigval_%d_%d.txt'%(folder, i,j))
        else:
            eig_val = None
        if os.path.isfile('%s/nbkeep.txt'%(folder)):
            nbkeep = np.loadtxt('%s/nbkeep.txt'%(folder)).\
            reshape((NbSubcube, NbSubcube)).astype(np.int)
        else:
            nbkeep = None
        if os.path.isfile('%s/cube_faint.fits'%folder):
            cube_faint = Cube('%s/cube_faint.fits'%folder)
        else:
            cube_faint = None
        if os.path.isfile('%s/cube_cont.fits'%folder):
            cube_cont = Cube('%s/cube_cont.fits'%folder)
        else:
            cube_cont = None
        if os.path.isfile('%s/cube_correl.fits'%folder):
            cube_correl = Cube('%s/cube_correl.fits'%folder)
        else:
            cube_correl = None
        if os.path.isfile('%s/cube_profile.fits'%folder):
            cube_profile = Cube('%s/cube_profile.fits'%folder)
        else:
            cube_profile = None
        if os.path.isfile('%s/cube_pval_correl.fits'%folder):
            cube_pval_correl = Cube('%s/cube_pval_correl.fits'%folder,
                                    mask=np.ma.nomask, dtype=np.float64)
        else:
            cube_pval_correl = None
        if os.path.isfile('%s/cube_pval_channel.fits'%folder):
            cube_pval_channel = Cube('%s/cube_pval_channel.fits'%folder,
                                     mask=np.ma.nomask, dtype=np.float64)
        else:
            cube_pval_channel = None
        if os.path.isfile('%s/cube_pval_final.fits'%folder):
            cube_pval_final = Cube('%s/cube_pval_final.fits'%folder,
                                   mask=np.ma.nomask, dtype=np.float64)
        else:
            cube_pval_final = None
        if os.path.isfile('%s/Cat0.fits'%folder):
            Cat0 = Table.read('%s/Cat0.fits'%folder)
        else:
            Cat0 = None
        if os.path.isfile('%s/Cat1.fits'%folder):
            Cat1 = Table.read('%s/Cat1.fits'%folder)
        else:
            Cat1 = None
        if os.path.isfile('%s/Cat1_T1.fits'%folder):
            Cat1_T1 = Table.read('%s/Cat1_T1.fits'%folder)
        else:
            Cat1_T1 = None
        if os.path.isfile('%s/Cat1_T2.fits'%folder):
            Cat1_T2 = Table.read('%s/Cat1_T2.fits'%folder)
        else:
            Cat1_T2 = None
        if os.path.isfile('%s/Cat2.fits'%folder):
            Cat2 = Table.read('%s/Cat2.fits'%folder)
        else:
            Cat2 = None
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
        if os.path.isfile('%s/Cat3.fits'%folder):
            Cat3 = Table.read('%s/Cat3.fits'%folder)
        else:
            Cat3 = None
        if os.path.isfile('%s/Cat4.fits'%folder):
            Cat4 = Table.read('%s/Cat4.fits'%folder)
        else:
            Cat4 = None
            
        if newpath is not None:
            path = newpath
        if newname is not None:
            name = newname
                
        return cls(path=path,  name=name, filename=param['cubename'],
                   NbSubcube=NbSubcube, margins=param['margin'],
                   profiles=param['profiles'], PSF=PSF, FWHM_PSF=FWHM_PSF,
                   intx=intx, inty=inty,
                   cube_faint=cube_faint, cube_cont=cube_cont,
                   cube_correl=cube_correl, cube_profile=cube_profile,
                   cube_pval_correl=cube_pval_correl,
                   cube_pval_channel=cube_pval_channel,
                   cube_pval_final=cube_pval_final, Cat0=Cat0, Cat1=Cat1,
                   Cat1_T1=Cat1_T1, Cat1_T2=Cat1_T2, Cat2=Cat2,
                   spectra=spectra, Cat3=Cat3, Cat4=Cat4, param=param,
                   eig_val=eig_val, nbkeep=nbkeep)
                   
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
        
        #step1
        if self.eig_val is not None:
            for i in range(self.NbSubcube):
                for j in range(self.NbSubcube):
                    np.savetxt('%s/eigval_%d_%d.txt'%(path2, i,j),
                               self.eig_val[(i,j)])
        if self.nbkeep is not None:
            np.savetxt('%s/nbkeep.txt'%path2, self.nbkeep)
        if self.cube_faint is not None:
            self.cube_faint.write('%s/cube_faint.fits'%path2)
        if self.cube_cont is not None:
            self.cube_cont.write('%s/cube_cont.fits'%path2)
        # step2
        if self.cube_correl is not None:
            self.cube_correl.write('%s/cube_correl.fits'%path2)
        if self.cube_profile is not None:
            self.cube_profile.write('%s/cube_profile.fits'%path2)
        # step3
        if self.cube_pval_correl is not None:
            self.cube_pval_correl.write('%s/cube_pval_correl.fits'%path2)
        if self.cube_pval_channel is not None:
            self.cube_pval_channel.write('%s/cube_pval_channel.fits'%path2)
        if self.cube_pval_final is not None:
            self.cube_pval_final.write('%s/cube_pval_final.fits'%path2)
        # step4
        if self.Cat0 is not None:
            self.Cat0.write('%s/Cat0.fits'%path2, overwrite=True)
        # step5
        if self.Cat1 is not None:
            self.Cat1.write('%s/Cat1.fits'%path2, overwrite=True)
        # step6
        if self.Cat1_T1 is not None:
            self.Cat1_T1.write('%s/Cat1_T1.fits'%path2, overwrite=True)
        if self.Cat1_T2 is not None:
            self.Cat1_T2.write('%s/Cat1_T2.fits'%path2, overwrite=True)
        # step7
        if self.Cat2 is not None:
            self.Cat2.write('%s/Cat2.fits'%path2, overwrite=True)
        if self.spectra is not None:
            hdulist = fits.HDUList([fits.PrimaryHDU()])
            for i in range(len(self.spectra)):
                hdu = self.spectra[i].get_data_hdu(name='DATA%d'%i,
                                                   savemask='nan')
                hdulist.append(hdu)
                hdu = self.spectra[i].get_stat_hdu(name='STAT%d'%i)
                if hdu is not None:
                    hdulist.append(hdu)
            hdulist.writeto('%s/spectra.fits'%path2, clobber=True)
        # step8
        if self.Cat3 is not None:
            self.Cat3.write('%s/Cat3.fits'%path2, overwrite=True)
        # step9
        if self.Cat4 is not None:
            self.Cat4.write('%s/Cat4.fits'%path2, overwrite=True)
        

    def step01_compute_PCA(self, r0=0.67):
        """ Loop on each zone of the data cube and compute the PCA,
        the number of eigenvectors to keep for the projection
        (with a linear regression and its associated determination
        coefficient) and return the projection of the data
        in the original basis keeping the desired number eigenvalues.

        Parameters
        ----------
        r0          : float
                      Coefficient of determination for projection during PCA

        Returns
        -------
        self.eig_val    : dictionary
                          Eigenvalues of each spatio-spectral zone
        self.nbkeep     : array
                          Number of eigenvalues for each zone used to compute
                          the projection
        self.cube_faint : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the lower
                     eigenvalues of the data cube
                     (representing the faint signal)
        self.cube_cont  : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the higher
                     eigenvalues of the data cube
                     (representing the continuum)
        """
        self._log_file.info('01 - PCA computation r0=%0.2f'%r0)
        self._log_stdout.info('Step 01 - PCA computation')
        self.param['r0PCA'] = r0
        
        # Weigthed data cube
        cube_std = self.cube_raw / np.sqrt(self.var)
        # Compute PCA results
        self._log_stdout.info('Compute the PCA on each zone')
        A, V, self.eig_val, nx, ny, nz = Compute_PCA_SubCube(self.NbSubcube,
                                                        cube_std,
                                                        self.intx, self.inty,
                                                        self.Edge_xmin,
                                                        self.Edge_xmax,
                                                        self.Edge_ymin,
                                                        self.Edge_ymax)

        # Number of eigenvectors for each zone
        # Parameter set to 1 if we want to plot the results
        # Parameters for projection during PCA
        self._log_stdout.info('Compute the number of eigenvectors to keep for the projection')
        list_r0 = np.resize(r0, self.NbSubcube**2)
        self.nbkeep = Compute_Number_Eigenvectors_Zone(self.NbSubcube, list_r0,
                                                  self.eig_val)
        # Adaptive projection of the cube on the eigenvectors
        self._log_stdout.info('Adaptive projection of the cube on the eigenvectors')
        cube_faint, cube_cont = Compute_Proj_Eigenvector_Zone(self.nbkeep,
                                                              self.NbSubcube,
                                                              self.Nx,
                                                              self.Ny,
                                                              self.Nz,
                                                              A, V,
                                                              nx, ny, nz,
                                                              self.inty,
                                                              self.intx)
                                                              
        self._log_stdout.info('Save the faint signal in self.cube_faint')
        self.cube_faint = Cube(data=cube_faint, wave=self.wave, wcs=self.wcs,
                          mask=np.ma.nomask)
        self._log_stdout.info('Save the continuum in self.cube_cont')
        self.cube_cont = Cube(data=cube_cont, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)
        self._log_file.info('01 Done')

    def step02_compute_TGLR(self):
        """Compute the cube of GLR test values.
        The test is done on the cube containing the faint signal
        (self.cube_faint) and it uses the PSF and the spectral profile.

        
        Returns
        -------
        self.cube_correl  : `~mpdaf.obj.Cube`
                            Cube of T_GLR values
        self.cube_profile : `~mpdaf.obj.Cube` (type int)
                             Number of the profile associated to the T_GLR
        """
        self._log_file.info('02 GLR test')
        self._log_stdout.info('Step 02 - GLR test')
        if self.cube_faint is None:
            raise IOError('Run the step 01 to initialize self.cube_faint')
            
        # TGLR computing (normalized correlations)
        correl, profile = Correlation_GLR_test(self.cube_faint._data, self.var,
                                               self.PSF, self.wfields,
                                               self.profiles)
                                               
        self._log_stdout.info('Save the TGLR value in self.cube_correl')
        self.cube_correl = Cube(data=correl, wave=self.wave, wcs=self.wcs,
                      mask=np.ma.nomask)
        self._log_stdout.info('Save the number of profile associated to the TGLR in self.cube_profile')
        self.cube_profile = Cube(data=profile, wave=self.wave, wcs=self.wcs,
                       mask=np.ma.nomask, dtype=int)
        self._log_file.info('Done')

    def step03_compute_pvalues(self, threshold=8):
        """Loop on each zone of self.cube_correl and compute for each zone:

        - the p-values associated to the T_GLR values,
        - the p-values associated to the number of thresholded p-values
          of the correlations per spectral channel,
        - the final p-values which are the thresholded pvalues associated
          to the T_GLR values divided by twice the pvalues associated to the
          number of thresholded p-values of the correlations per spectral
          channel.

        Parameters
        ----------
        threshold : float
                    Threshold applied on pvalues.

        Returns
        -------
        self.cube_pval_correl  : `~mpdaf.obj.Cube`
                                 Cube of thresholded p-values associated
                                 to the T_GLR values
        self.cube_pval_channel : `~mpdaf.obj.Cube`
                                 Cube of p-values associated to the number of
                                 thresholded p-values of the correlations
                                 per spectral channel for each zone
        self.cube_pval_final   : `~mpdaf.obj.Cube`
                                 Cube of final thresholded p-values
        """
        self._log_file.info('03 p-values computation threshold=%.1f'%threshold)
        self._log_stdout.info('Step 03 - p-values computation')
        self._log_stdout.info('Compute p-values of correlation values')
        self.param['ThresholdPval'] = threshold
        if self.cube_correl is None:
            raise IOError('Run the step 02 to initialize self.cube_correl')

        cube_pval_correl = Compute_pval_correl_zone(self.cube_correl._data,
                                                    self.intx, self.inty,
                                                    self.NbSubcube,
                                                    self.Edge_xmin,
                                                    self.Edge_xmax,
                                                    self.Edge_ymin,
                                                    self.Edge_ymax,
                                                    threshold)
        self._log_stdout.info('Save the result in self.cube_pval_correl')
        self.cube_pval_correl = Cube(data=cube_pval_correl, wave=self.wave,
                                     wcs=self.wcs, mask=np.ma.nomask)

        # p-values of spectral channel
        # Estimated mean for p-values distribution related
        # to the Rayleigh criterium
        self._log_stdout.info('Compute p-values of spectral channel')
        try:
            mean_est = self.FWHM_PSF**2
            self.param['meanestPvalChan'] = np.asscalar(mean_est)
        except:
            mean_est = [FWHM_PSF**2 for FWHM_PSF in self.FWHM_PSF]
            self.param['meanestPvalChan'] = mean_est.tolist()
        cube_pval_channel = Compute_pval_channel_Zone(cube_pval_correl,
                                                      self.intx, self.inty,
                                                      self.NbSubcube,
                                                      mean_est, self.wfields)
        self._log_stdout.info('Save the result in self.cube_pval_channel')
        self.cube_pval_channel = Cube(data=cube_pval_channel, wave=self.wave,
                                      wcs=self.wcs, mask=np.ma.nomask)

        # Final p-values
        self._log_stdout.info('Compute final p-values')
        cube_pval_final = Compute_pval_final(cube_pval_correl,
                                             cube_pval_channel,
                                             threshold)
        self._log_stdout.info('Save the result in self.cube_pval_final')
        self.cube_pval_final = Cube(data=cube_pval_final, wave=self.wave,
                                    wcs=self.wcs, mask=np.ma.nomask)
        self._log_file.info('03 Done')

    def step04_compute_ref_pix(self, neighboors=26):
        """compute the groups of connected voxels with a flood-fill algorithm
        on the cube of final thresholded p-values. Then compute referent
        voxel of each group of connected voxels using the voxel with the
        higher T_GLR value.

        Parameters
        ----------
        neighboors        : integer
                            Connectivity of contiguous voxels

        Returns
        -------
        self.Cat0 : astropy.Table
                    Catalogue of the referent voxels for each group.
                    Columns: x y z ra dec lbda T_GLR profile pvalC pvalS pvalF
                    Coordinates are in pixels.
        """
        self._log_file.info('04 compute referent pixels neighboors=%d'%neighboors)
        self._log_stdout.info('Step 04 - referent pixels')
        
        # connected voxel
        self._log_stdout.info('Compute connected voxels')
        self.param['neighboors'] = neighboors
        labeled_cube, Ngp = Compute_Connected_Voxel(self.cube_pval_final._data,
                                                    neighboors)
        self._log_stdout.info('%d connected voxels detected' % Ngp)
        # Referent pixel
        self._log_stdout.info('Compute referent pixels')
        if self.cube_correl is None or self.cube_profile is None:
            raise IOError('Run the step 02 to initialize self.cube_correl and self.cube_profile')
        if self.cube_pval_correl is None or self.cube_pval_channel is None \
                                         or self.cube_pval_final is None:
            raise IOError('Run the step 03 to initialize self.cube_pval_* cubes')
            
        self.Cat0 = Compute_Referent_Voxel(self.cube_correl._data,
                                           self.cube_profile._data,
                                           self.cube_pval_correl._data,
                                           self.cube_pval_channel._data,
                                           self.cube_pval_final._data, Ngp,
                                           labeled_cube, self.wcs, self.wave)
        self._log_stdout.info('Save a first version of the catalogue of emission lines in self.Cat0')
        self._log_file.info('04 Done')

    def step05_compute_NBtests(self, nb_ranges=3):
        """compute the 2 narrow band tests for each detected emission line.

        Parameters
        ----------
        nb_ranges   : integer
                      Number of the spectral ranges skipped to compute the
                      controle cube

        Returns
        -------
        self.Cat1 : astropy.Table
               Catalogue of parameters of detected emission lines.
               Columns: x y z ra dec lbda T_GLR profile pvalC pvalS pvalF T1 T2
        """
        self._log_file.info('05 NB tests nb_ranges=%d'%nb_ranges)
        self._log_stdout.info('Step 05 - NB tests')
        self.param['NBranges'] = nb_ranges
        if self.Cat0 is None:
            raise IOError('Run the step 04 to initialize self.Cat0')
        self.Cat1 = Narrow_Band_Test(self.Cat0, self.cube_raw, self.profiles,
                                self.PSF, self.wfields, nb_ranges, self.wcs)
        self._log_stdout.info('Save the updated catalogue in self.Cat1')
        self._log_file.info('05 Done')

    def step06_select_NBtests(self, thresh_T1=0.2, thresh_T2=2):
        """select emission lines according to the 2 narrow band tests.

        Parameters
        ----------
        thresh_T1 : float
                    Threshold for the test 1
        thresh_T2 : float
                    Threshold for the test 2

        Returns
        -------
        self.Cat1_T1 : astropy.Table
                       Catalogue of parameters of detected emission lines
                       selected with the test 1
        self.Cat1_T2 : astropy.Table
                       Catalogue of parameters of detected emission lines
                       selected with the test 2

        Columns of the catalogues :
        x y z ra dec lbda T_GLR profile pvalC pvalS pvalF T1 T2
        """
        self._log_file.info('06 Selection according to NB tests thresh_T1=%.1f thresh_T2=%.1f'%(thresh_T1, thresh_T2))
        self._log_stdout.info('Step 06 - Selection according to NB tests')
        self.param['threshT1'] = thresh_T1
        self.param['threshT2'] = thresh_T2
        if self.Cat1 is None:
            raise IOError('Run the step 05 to initialize self.Cat1')
        # Thresholded narrow bands tests
        self.Cat1_T1, self.Cat1_T2 = Narrow_Band_Threshold(self.Cat1,
                                                           thresh_T1,
                                                           thresh_T2)
        self._log_stdout.info('%d emission lines selected with the test 1' % len(self.Cat1_T1))
        self._log_stdout.info('Save the corresponding catalogue in self.Cat1_T1')
        self._log_stdout.info('%d emission lines selected with the test 2' % len(self.Cat1_T2))
        self._log_stdout.info('Save the corresponding catalogue in self.Cat1_T2')
        self._log_file.info('06 Done')

    def step07_compute_spectra(self, T=2, grid_dxy=0, grid_dz=0):
        """compute the estimated emission line and the optimal coordinates
        for each detected lines in a spatio-spectral grid (each emission line
        is estimated with the deconvolution model :
        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))

        Parameters
        ----------
        T          : 1 or 2
                     if T=1, self.Cat1_T1 is used as input
                     if T=2, self.Cat1_T2 is used as input
        grid_dxy   : integer
                     Maximum spatial shift for the grid
        grid_dz    : integer
                     Maximum spectral shift for the grid

        Returns
        -------
        self.Cat2    : astropy.Table
                       Catalogue of parameters of detected emission lines.
                       Columns: x y z ra dec lbda T_GLR profile pvalC pvalS
                                pvalF T1 T2
                       residual flux num_line
        self.spectra : list of `~mpdaf.obj.Spectrum`
                       Estimated lines
        """
        self._log_file.info('07 Lines estimation T=%d grid_dxy=%d grid_dz=%d'%(T, grid_dxy, grid_dz))
        self._log_stdout.info('Step 07 - Lines estimation')
        self.param['NBtest'] = 'T%d'%T
        self.param['grid_dxy'] = grid_dxy
        self.param['grid_dz'] = grid_dz
        if T==1:
            Cat1_T = self.Cat1_T1
        elif T==2:
            Cat1_T = self.Cat1_T2
        else:
            raise IOError('Invalid parameter T')
        if self.cube_faint is None:
            raise IOError('Run the step 01 to initialize self.cube_faint')
        if self.cube_profile is None:
            raise IOError('Run the step 02 to initialize self.cube_profile')
        if Cat1_T is None:
            raise IOError('Run the step 06 to initialize self.Cat1_T* catalogs')
        self.Cat2, Cat_est_line_raw_T, Cat_est_line_std_T = \
            Estimation_Line(Cat1_T, self.cube_profile._data, self.Nx, self.Ny,
                            self.Nz, self.var, self.cube_faint._data, grid_dxy,
                            grid_dz, self.PSF, self.wfields, self.profiles,
                            self.wcs, self.wave)
        self._log_stdout.info('Save the updated catalogue in self.Cat2')
        self.spectra = []
        for data, std in zip(Cat_est_line_raw_T, Cat_est_line_std_T):
            spe = Spectrum(data=data, var=std**2, wave=self.wave,
                           mask=np.ma.nomask)
            self.spectra.append(spe)
        self._log_stdout.info('Save the estimated spectrum of each line in self.spectra')
        self._log_file.info('07 Done')

    def step08_spatial_merging(self):
        """Construct a catalogue of sources by spatial merging of the
        detected emission lines in a circle with a diameter equal to
        the mean over the wavelengths of the FWHM of the FSF.

        Returns
        -------
        self.Cat3 : astropy.Table
                    Columns: ID x_circle y_circle ra_circle dec_circle
                    x_centroid y_centroid ra_centroid dec_centroid nb_lines x y
                    z ra dec lbda T_GLR profile pvalC pvalS pvalF T1 T2
                    residual flux num_line
        """
        self._log_file.info('08 Spatial merging')
        self._log_stdout.info('Step 08 - Spatial merging')
        if self.wfields is None:
            fwhm = self.FWHM_PSF
        else:
            fwhm = np.max(np.array(self.FWHM_PSF)) # to be improved !!
        if self.Cat2 is None:
            raise IOError('Run the step 07 to initialize self.Cat2')
        self.Cat3 = Spatial_Merging_Circle(self.Cat2, fwhm, self.wcs)
        self._log_stdout.info('Save the updated catalogue in self.Cat3')
        self._log_file.info('08 Done')

    def step09_spectral_merging(self, deltaz=1):
        """Merge the detected emission lines distants to less than deltaz
           spectral channel in each group.

        Parameters
        ----------
        deltaz : integer
                 Distance maximum between 2 different lines

        Returns
        -------
        self.Cat4 : astropy.Table
                    Catalogue
                    Columns: ID x_circle y_circle ra_circle dec_circle
                    x_centroid y_centroid ra_centroid dec_centroid nb_lines x y
                    z ra dec lbda T_GLR profile pvalC pvalS pvalF T1 T2
                    residual flux num_line
        """
        self._log_file.info('09 spectral merging deltaz=%d'%deltaz)
        self._log_stdout.info('Step 09 - Spectral merging')
        self.param['deltaz'] = deltaz
        if self.Cat3 is None:
            raise IOError('Run the step 08 to initialize self.Cat3')
        Cat_est_line_raw = [spe._data for spe in self.spectra]
        self.Cat4 = Spectral_Merging(self.Cat3, Cat_est_line_raw, deltaz)
        self._log_stdout.info('Save the updated catalogue in self.Cat4')
        self._log_file.info('09 Done')

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
        sources : mpdaf.sdetect.SourceList
                  List of sources
        """
        self._log_file.info('10 Sources creation')
        # Add RA-DEC to the catalogue
        self._log_stdout.info('Step 10 - Sources creation')
        self._log_stdout.info('Add RA-DEC to the catalogue')
        if self.Cat4 is None:
            raise IOError('Run the step 10 to initialize self.Cat4')

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
            raise IOError('Run the step 02 to initialize self.cube_correl')
        if self.spectra is None:
            raise IOError('Run the step 07 to initialize self.spectra')
        nsources = Construct_Object_Catalogue(self.Cat4, self.spectra,
                                              self.cube_correl._data,
                                              self.wave, self.FWHM_profiles,
                                              path_src, self.name, self.param,
                                              src_vers, author, ncpu)
                                              
        # create the final catalog
        self._log_stdout.info('Create the final catalog')
        catF = Catalog.from_path(path_src)
        catF.write(catname)
                      
        self._log_file.info('10 Done')

        return nsources
        
    def plot_PCA(self, i, j, ax=None):
        """ Plot the eigenvalues and the separation point
        
        Parameters
        ----------
        i: integer in [0, NbSubCube[
           x-coordinate of the zone
        j: integer in [0, NbSubCube[
           y-coordinate of the zone
        ax : matplotlib.Axes
                the Axes instance in which the image is drawn
        """
        if self.eig_val is None or self.nbkeep is None:
            raise IOError('Run the step 01 to initialize self.eig_val and selb.nbkeep')
            
        if ax is None:
            ax = plt.gca()
        
        lambdat = self.eig_val[(i, j)]
        nbt = self.nbkeep[i, j]
        ax.semilogy(lambdat)
        ax.semilogy(nbt, lambdat[nbt], 'r+')
        plt.title('zone (%d, %d)' %(i,j))
        
    def plot_NB(self, i, ax1=None, ax2=None, ax3=None):
        """Plot the narrow bands images
        
        i : integer
            index of the object in self.Cat1
        ax1 : matplotlib.Axes
              The Axes instance in which the NB image
              around the source is drawn
        ax2 : matplotlib.Axes
              The Axes instance in which a other NB image for check is drawn
        ax3 : matplotlib.Axes
              The Axes instance in which the difference is drawn
        """
        if self.Cat1 is None:
            raise IOError('Run the step 05 to initialize self.Cat1')
            
        if ax1 is None and ax2 is None and ax3 is None:
            ax1 = plt.subplot(1,3,1)
            ax2 = plt.subplot(1,3,2)
            ax3 = plt.subplot(1,3,3)
            
        # Coordinates of the source
        x0 = self.Cat1[i]['x']
        y0 = self.Cat1[i]['y']
        z0 = self.Cat1[i]['z']
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
        num_prof = self.Cat1[i]['profile']
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
        nb_ranges = self.param['NBranges']
        if (z0 + longz + nb_ranges * long0) < self.cube_raw.shape[0]:
            intz1c = intz1 + nb_ranges * long0
            intz2c = intz2 + nb_ranges * long0
        else:
            intz1c = intz1 - nb_ranges * long0
            intz2c = intz2 - nb_ranges * long0
        cube_controle_plot = self.cube_raw[intz1c:intz2c, y01:y02, x01:x02]
        # (1/sqrt(2)) * difference of the 2 sububes
        diff_cube_plot = (1. / np.sqrt(2)) * (cube_test_plot - cube_controle_plot)
        # tests
        T1 = self.Cat1[i]['T1']
        T2 = self.Cat1[i]['T2']
        
        if ax1 is not None:
            ax1.plot(x00, y00, 'm+')
            ima_test_plot = Image(data=cube_test_plot.sum(axis=0), wcs=wcs)
            title = 'cube test - (%d,%d)\n' % (x0, y0) + \
                    'T1=%.3f T2=%.3f\n' % (T1,T2) + \
                    'lambda=%d int=[%d,%d[' % (z0, intz1, intz2)
            ima_test_plot.plot(colorbar='v', title=title, ax=ax1)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)

        if ax2 is not None:
            ax2.plot(x00, y00, 'm+')
            ima_controle_plot = Image(data=cube_controle_plot.sum(axis=0), wcs=wcs)
            title = 'check - (%d,%d)\n' % (x0, y0) + \
                        'T1=%.3f T2=%.3f\n' % (T1, T2) + \
                        'int=[%d,%d[' % (intz1c, intz2c)
            ima_controle_plot.plot(colorbar='v', title=title, ax=ax2)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)

        if ax3 is not None:
            ax3.plot(x00, y00, 'm+')
            ima_diff_plot = Image(data=diff_cube_plot.sum(axis=0), wcs=wcs)
            title = 'Difference narrow band - (%d,%d)\n' % (x0, y0) + \
                    'T1=%.3f T2=%.3f\n' % (T1, T2) + \
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
            
        carte_2D_correl = np.amax(self.cube_correl._data, axis=0)
        carte_2D_correl_ = Image(data=carte_2D_correl, wcs=self.wcs)

        if ax is None:
            ax = plt.gca()

        ax.plot(x, y, 'k+')
        if circle:
            for px, py in zip(x, y):
                c = plt.Circle((px, py), np.round(fwhm / 2), color='k',
                               fill=False)
                ax.add_artist(c)
        carte_2D_correl_.plot(vmin=vmin, vmax=vmax, title=title, ax=ax)
        
    def info(self):
        """ plot information
        """
        currentlog = self._log_file.handlers[0].baseFilename
        with open(currentlog) as f:
            for line in f:
                if line.find('Done') == -1:
                    self._log_stdout.info(line)
        