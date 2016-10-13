"""Copyright 2010-2016 CNRS/CRAL

This file is part of MPDAF.

MPDAF is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version

MPDAF is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with MPDAF.  If not, see <http://www.gnu.org/licenses/>.


ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes

This software has been developped by Carole Clastres under the supervision of
David Mary (Lagrange institute, University of Nice) and ported to python by
Laure Piqueras (CRAL).

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL). Please contact
Carole for more info at carole.clastres@univ-lyon1.fr

origin.py contains an oriented-object interface to run the ORIGIN software
"""

from __future__ import absolute_import, division

import astropy.units as u
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
from scipy.io import loadmat
import shutil

from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.MUSE import FSF,FieldsMap 
from .lib_origin import Compute_PSF, Spatial_Segmentation, \
    Compute_PCA_SubCube, Compute_Number_Eigenvectors_Zone, \
    Compute_Proj_Eigenvector_Zone, Correlation_GLR_test, \
    Compute_pval_correl_zone, Compute_pval_channel_Zone, \
    Compute_pval_final, Compute_Connected_Voxel, \
    Compute_Referent_Voxel, Narrow_Band_Test, \
    Narrow_Band_Threshold, Estimation_Line, \
    Spatial_Merging_Circle, Spectral_Merging, \
    Add_radec_to_Cat, Construct_Object_Catalogue


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
        filename      : string
                        Cube FITS file name.
        cube_raw      : array (Nz, Ny, Nx)
                        Raw data.
        var           : array (Nz, Ny, Nx)
                        Covariance.
        Nx            : integer
                        Number of columns
        Ny            : integer
                        Number of rows
        Nz            : int
                        Number of spectral channels
        wcs           : `mpdaf.obj.WCS`
                        RA-DEC coordinates.
        wave          : `mpdaf.obj.WaveCoord`
                        Spectral coordinates.
        intx          : array
                        Limits in pixels of the columns for each zone
        inty          : array
                        Limits in pixels of the rows for each zone
        Edge_xmin     : int
                        Minimum limits along the x-axis in pixel
                        of the data cube taken to compute p-values
        Edge_xmax     : int
                        Maximum limits along the x-axis in pixel
                        of the data cube taken to compute p-values
        Edge_ymin     : int
                        Minimum limits along the y-axis in pixel
                        of the data cube taken to compute p-values
        Edge_ymax     : int
                        Maximum limits along the y-axis in pixel
                        of the data cube taken to compute p-values
        profiles      : array
                        Dictionary of spectral profiles to test
        FWHM_profiles : array
                        FWHM of the profiles in pixels.
        PSF           : array (Nz, Nfsf, Nfsf)
                        MUSE PSF
        FWHM_PSF      : float
                        Mean of the fwhm of the PSF in pixel
    """

    def __init__(self, cube, NbSubcube, margins, profiles=None,
                 FWHM_profiles=None, PSF=None, FWHM_PSF=None):
        """Create a ORIGIN object.

        An Origin object is composed by:
        - cube data (raw data and covariance)
        - 1D dictionary of spectral profiles
        - MUSE PSF
        - parameters used to segment the cube in different zones.


        Parameters
        ----------
        cube        : string (Cube FITS file name) or cube object
        NbSubcube   : integer
                      Number of sub-cubes for the spatial segmentation
        margin      : (int, int, int, int)
                      Size in pixels of the margins
                      at the bottom, top, left and rigth  of the data cube.
                      (ymin, Ny-ymax, xmin, Nx-xmax)
                      Pixels in margins will not be taken
                      into account to compute p-values.
        profiles    : array (Size_profile, N_profile)
                      Dictionary of spectral profiles
                      If None, a default dictionary of 20 profiles is used.
        FWHM_profiles : array (N_profile)
                        FWHM of the profiles in pixels.
        PSF         : string
                      Cube FITS filename containing a MUSE PSF per wavelength.
                      If None, PSF are computed with a Moffat function
                      (13x13 pixels, beta=2.6, fwhm1=0.76, fwhm2=0.66,
                      lambda1=4750, lambda2=7000)
        FWHM_PSF    : array (Nz)
                      FWHM of the PSFs in pixels.
        """
        self._logger = logging.getLogger('mpdaf')
        self._logger.info('ORIGIN - Read the Data Cube')
        # create parameters dictionary
        self.param = {}
        self.param['cubename'] = cube
        self.param['nbsubcube'] = NbSubcube
        self.param['margin'] = margins
        self.param['PSF'] = PSF
        if type(cube) is Cube:
            self.filename = cube.filename
            cub = cube
        else:
            # Read cube
            self.filename = cube
            cub = Cube(self.filename)
        # Raw data cube
        # Set to 0 the Nan
        self.cube_raw = cub.data.filled(fill_value=0)
        # variance
        self.var = cub.var
        # RA-DEC coordinates
        self.wcs = cub.wcs
        # spectral coordinates
        self.wave = cub.wave

        # Dimensions
        self.Nz, self.Ny, self.Nx = cub.shape

        # Set to Inf the Nana
        self.var[np.isnan(self.var)] = np.inf

        self.NbSubcube = NbSubcube

        self.Edge_xmin = margins[2]
        self.Edge_xmax = self.Nx - margins[3]
        self.Edge_ymin = margins[0]
        self.Edge_ymax = self.Ny - margins[1]

        # Dictionary of spectral profile
        if profiles is None or FWHM_profiles is None:
            self._logger.info('ORIGIN - Load dictionary of spectral profile')
            DIR = os.path.dirname(__file__)
            self.profiles = loadmat(DIR + '/Dico_FWHM_2_12.mat')['Dico']
            self.FWHM_profiles = np.linspace(2, 12, 20)  # pixels
        else:
            self.profiles = profiles
            self.FWHM_profiles = FWHM_profiles

        # FSF cube(s)
        self._logger.info('ORIGIN - Compute PSF')
        step_arcsec = self.wcs.get_step(unit=u.arcsec)[0]
        self.wfields = None
        if PSF is None or FWHM_PSF is None:
            Nfsf=13
            try:
                FSF_mode = cub.primary_header['FSFMODE']
                if FSF_mode != 'MOFFAT1':
                    raise IOError('This method is coded only for FSFMODE=MOFFAT1')
            except:
                FSF_mode = None
                
            if FSF_mode is None:
                self.PSF, fwhm_pix, fwhm_arcsec = \
                Compute_PSF(self.wave, self.Nz, Nfsf=Nfsf, beta=2.6,
                            fwhm1=0.76, fwhm2=0.66, lambda1=4750,
                            lambda2=7000, step_arcsec=step_arcsec)
                # mean of the fwhm of the FSF in pixel
                self.FWHM_PSF = np.mean(fwhm_arcsec) / step_arcsec
            else:
                self.param['PSF'] = FSF_mode
                nfields = cub.primary_header['NFIELDS']
                FSF_model = FSF(FSF_mode)
                if nfields == 1: # just one FSF
                    nf = 0
                    beta = cub.primary_header['FSF%02dBET'%nf]
                    a = cub.primary_header['FSF%02dFWA'%nf]
                    b = cub.primary_header['FSF%02dFWB'%nf]
                    self.PSF, fwhm_pix, fwhm_arcsec = \
                    FSF_model.get_FSF_cube(cub, Nfsf, beta=beta, a=a, b=b)
                    # Normalization
                    self.PSF = self.PSF / np.sum(self.PSF, axis=(1, 2))\
                    [:, np.newaxis, np.newaxis]
                    # mean of the fwhm of the FSF in pixel
                    self.FWHM_PSF = np.mean(fwhm_arcsec) / step_arcsec
                else:
                    self.PSF = []
                    self.FWHM_PSF = []
                    for i in range(1, nfields+1):
                        beta = cub.primary_header['FSF%02dBET'%i]
                        a = cub.primary_header['FSF%02dFWA'%i]
                        b = cub.primary_header['FSF%02dFWB'%i]
                        PSF, fwhm_pix, fwhm_arcsec = \
                        FSF_model.get_FSF_cube(cub, Nfsf, beta=beta,
                                               a=a, b=b)
                        # Normalization (???)
                        PSF = PSF / np.sum(PSF, axis=(1, 2))[:, np.newaxis,
                                                             np.newaxis]
                        self.PSF.append(PSF)
                        # mean of the fwhm of the FSF in pixel
                        self.FWHM_PSF.append(np.mean(fwhm_arcsec) / step_arcsec)
                    fmap = FieldsMap(self.filename, extname='FIELDMAP')
                    self.wfields = fmap.compute_weights()

        else:
            cubePSF = Cube(PSF)
            if cubePSF.shape[1] != cubePSF.shape[2]:
                raise IOError('PSF must be a square image.')
            if not cubePSF.shape[1]%2:
                raise IOError('The spatial size of the PSF must be odd.')
            if cubePSF.shape[0] != self.Nz:
                raise IOError('PSF and data cube have not the same dimensions along the spectral axis.')
            if not np.isclose(cubePSF.wcs.get_step(unit=u.arcsec)[0], step_arcsec):
                raise IOError('PSF and data cube have not the same pixel sizes.')

            self.PSF = cubePSF._data
            # mean of the fwhm of the FSF in pixel
            self.FWHM_PSF = np.mean(FWHM_PSF)
            
        del cub

        # Spatial segmentation
        self._logger.info('ORIGIN - Spatial segmentation')
        self.inty, self.intx = Spatial_Segmentation(self.Nx, self.Ny,
                                                    NbSubcube)

    def compute_PCA(self, r0=0.67, fig=None):
        """ Loop on each zone of the data cube and compute the PCA,
        the number of eigenvectors to keep for the projection
        (with a linear regression and its associated determination
        coefficient) and return the projection of the data
        in the original basis keeping the desired number eigenvalues.

        Parameters
        ----------
        r0          : float
                      Coefficient of determination for projection during PCA
        fig : figure instance
                      If not None, plot the eigenvalues and the separation point.

        Returns
        -------
        cube_faint : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the lower
                     eigenvalues of the data cube
                     (representing the faint signal)
        cube_cont  : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the higher
                     eigenvalues of the data cube
                     (representing the continuum)
        """
        # save paaremeters values in object
        self.param['r0PCA'] = r0
        # Weigthed data cube
        cube_std = self.cube_raw / np.sqrt(self.var)
        # Compute PCA results
        self._logger.info('ORIGIN - Compute the PCA on each zone')
        A, V, eig_val, nx, ny, nz = Compute_PCA_SubCube(self.NbSubcube,
                                                        cube_std,
                                                        self.intx, self.inty,
                                                        self.Edge_xmin,
                                                        self.Edge_xmax,
                                                        self.Edge_ymin,
                                                        self.Edge_ymax)

        # Number of eigenvectors for each zone
        # Parameter set to 1 if we want to plot the results
        # Parameters for projection during PCA
        self._logger.info('ORIGIN - Compute the number of eigenvectors to keep for the projection')
        list_r0 = np.resize(r0, self.NbSubcube**2)
        nbkeep = Compute_Number_Eigenvectors_Zone(self.NbSubcube, list_r0, eig_val, fig)
        # Adaptive projection of the cube on the eigenvectors
        self._logger.info('ORIGIN - Adaptive projection of the cube on the eigenvectors')
        cube_faint, cube_cont = Compute_Proj_Eigenvector_Zone(nbkeep,
                                                              self.NbSubcube,
                                                              self.Nx,
                                                              self.Ny,
                                                              self.Nz,
                                                              A, V,
                                                              nx, ny, nz,
                                                              self.inty, self.intx)
        cube_faint = Cube(data=cube_faint, wave=self.wave, wcs=self.wcs,
                          mask=np.ma.nomask)
        cube_cont = Cube(data=cube_cont, wave=self.wave, wcs=self.wcs,
                         mask=np.ma.nomask)
        return cube_faint, cube_cont

    def compute_TGLR(self, cube_faint):
        """Compute the cube of GLR test values obtained with the given
        PSF and dictionary of spectral profile.

        Parameters
        ----------
        cube_faint : mpdaf.obj.cube
                     data cube on test

        Returns
        -------
        correl  : `~mpdaf.obj.Cube`
                  cube of T_GLR values
        profile : `~mpdaf.obj.Cube` (type int)
                  Number of the profile associated to the T_GLR
                  profile = Cube('profile.fits', dtype=int)
        """
        # TGLR computing (normalized correlations)
        self._logger.info('ORIGIN - Compute the GLR test')
        correl, profile = Correlation_GLR_test(cube_faint._data, self.var,
                                               self.PSF, self.wfields,
                                               self.profiles)
        correl = Cube(data=correl, wave=self.wave, wcs=self.wcs,
                      mask=np.ma.nomask)
        profile = Cube(data=profile, wave=self.wave, wcs=self.wcs,
                       mask=np.ma.nomask, dtype=int)
        return correl, profile

    def compute_pvalues(self, correl, threshold=8):
        """Loop on each zone of the data cube and compute for each zone:

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
        cube_pval_correl  : `~mpdaf.obj.Cube`
                            Cube of thresholded p-values associated
                            to the T_GLR values
        cube_pval_channel : `~mpdaf.obj.Cube`
                            Cube of p-values associated to the number of
                            thresholded p-values of the correlations
                            per spectral channel for each zone
        cube_pval_final   : `~mpdaf.obj.Cube`
                            Cube of final thresholded p-values
        """
        # p-values of correlation values
        self._logger.info('ORIGIN - Compute p-values of correlation values')
        self.param['ThresholdPval'] = threshold
        cube_pval_correl = Compute_pval_correl_zone(correl._data, self.intx,
                                                    self.inty, self.NbSubcube,
                                                    self.Edge_xmin,
                                                    self.Edge_xmax,
                                                    self.Edge_ymin,
                                                    self.Edge_ymax,
                                                    threshold)
        # p-values of spectral channel
        # Estimated mean for p-values distribution related
        # to the Rayleigh criterium
        self._logger.info('ORIGIN - Compute p-values of spectral channel')
        try:
            mean_est = self.FWHM_PSF**2
        except:
            mean_est = [FWHM_PSF**2 for FWHM_PSF in self.FWHM_PSF]
        self.param['meanestPvalChan'] = mean_est
        cube_pval_channel = Compute_pval_channel_Zone(cube_pval_correl,
                                                      self.intx, self.inty,
                                                      self.NbSubcube,
                                                      mean_est, self.wfields)

        # Final p-values
        self._logger.info('ORIGIN - Compute final p-values')
        cube_pval_final = Compute_pval_final(cube_pval_correl, cube_pval_channel,
                                             threshold)

        cube_pval_correl = Cube(data=cube_pval_correl, wave=self.wave, wcs=self.wcs,
                                mask=np.ma.nomask)
        cube_pval_channel = Cube(data=cube_pval_channel, wave=self.wave, wcs=self.wcs,
                                 mask=np.ma.nomask)
        cube_pval_final = Cube(data=cube_pval_final, wave=self.wave, wcs=self.wcs,
                               mask=np.ma.nomask)

        return cube_pval_correl, cube_pval_channel, cube_pval_final

    def compute_ref_pix(self, correl, profile, cube_pval_correl,
                        cube_pval_channel, cube_pval_final,
                        neighboors=26):
        """compute the groups of connected voxels with a flood-fill algorithm
        on the cube of final thresholded p-values. Then compute referent
        voxel of each group of connected voxels using the voxel with the
        higher T_GLR value.

        Parameters
        ----------
        correl            : `~mpdaf.obj.Cube`
                            Cube of T_GLR values
        profile           : `~mpdaf.obj.Cube` (type int)
                            Number of the profile associated to the T_GLR
        cube_pval_correl  : `~mpdaf.obj.Cube`
                           Cube of thresholded p-values associated
                           to the T_GLR values
        cube_pval_channel : `~mpdaf.obj.Cube`
                            Cube of spectral p-values
        cube_pval_final   : `~mpdaf.obj.Cube`
                            Cube of final thresholded p-values
        neighboors        : integer
                            Connectivity of contiguous voxels

        Returns
        -------
        Cat0 : astropy.Table
               Catalogue of the referent voxels for each group.
               Coordinates are in pixels.
               Columns of Cat_ref : x y z T_GLR profile pvalC pvalS pvalF
        """
        # connected voxels
        self._logger.info('ORIGIN - Compute connected voxels')
        self.param['neighboors'] = neighboors
        labeled_cube, Ngp = Compute_Connected_Voxel(cube_pval_final._data, neighboors)
        self._logger.info('ORIGIN - %d connected voxels detected' % Ngp)
        # Referent pixel
        self._logger.info('ORIGIN - Compute referent pixels')
        Cat0 = Compute_Referent_Voxel(correl._data, profile._data, cube_pval_correl._data,
                                      cube_pval_channel._data, cube_pval_final._data, Ngp,
                                      labeled_cube)
        return Cat0

    def compute_NBtests(self, Cat0, nb_ranges=3, plot_narrow=False):
        """compute the 2 narrow band tests for each detected emission line.

        Parameters
        ----------
        Cat0        : astropy.Table
                      Catalogue of parameters of detected emission lines.
                      Columns of the Catalogue Cat0 :
                      x y z T_GLR profile pvalC pvalS pvalF
        nb_ranges   : integer
                      Number of the spectral ranges skipped to compute the
                      controle cube
        plot_narrow : boolean
                      If True, plot the narrow bands images

        Returns
        -------
        Cat1 : astropy.Table
               Catalogue of parameters of detected emission lines.
               Columns of the Catalogue Cat1 :
               x y z T_GLR profile pvalC pvalS pvalF T1 T2
        """
        # Parameter set to 1 if we want to plot the results and associated folder
        plot_narrow = False
        self._logger.info('ORIGIN - Compute narrow band tests')
        self.param['NBranges'] = nb_ranges
        Cat1 = Narrow_Band_Test(Cat0, self.cube_raw, self.profiles,
                                self.PSF, self.wfields, nb_ranges,
                                plot_narrow, self.wcs)
        return Cat1

    def select_NBtests(self, Cat1, thresh_T1=0.2, thresh_T2=2):
        """select emission lines according to the 2 narrow band tests.

        Parameters
        ----------
        Cat1      : astropy.Table
                    Catalogue of detected emission lines.
        thresh_T1 : float
                    Threshold for the test 1
        thresh_T2 : float
                    Threshold for the test 2

        Returns
        -------
        Cat1_T1 : astropy.Table
                  Catalogue of parameters of detected emission lines selected with
                  the test 1
        Cat1_T2 : astropy.Table
                  Catalogue of parameters of detected emission lines selected with
                  the test 2

        Columns of the catalogues :
        x y z T_GLR profile pvalC pvalS pvalF T1 T2
        """
        self.param['threshT1'] = thresh_T1
        self.param['threshT2'] = thresh_T2
        # Thresholded narrow bands tests
        Cat1_T1, Cat1_T2 = Narrow_Band_Threshold(Cat1, thresh_T1, thresh_T2)
        self._logger.info('ORIGIN - %d emission lines selected with the test 1' % len(Cat1_T1))
        self._logger.info('ORIGIN - %d emission lines selected with the test 2' % len(Cat1_T2))
        return Cat1_T1, Cat1_T2

    def estimate_line(self, Cat1_T, profile, cube_faint,
                      grid_dxy=0, grid_dz=0):
        """compute the estimated emission line and the optimal coordinates
        for each detected lines in a spatio-spectral grid (each emission line
        is estimated with the deconvolution model :
        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))

        Parameters
        ----------
        Cat1_T     : astropy.Table
                     Catalogue of parameters of detected emission lines selected
                     with a narrow band test.
                     Columns of the Catalogue Cat1_T:
                     x y z T_GLR profile pvalC pvalS pvalF T1 T2
        profile    : `~mpdaf.obj.Cube`
                     Number of the profile associated to the T_GLR
        cube_faint : `~mpdaf.obj.Cube`
                     Projection on the eigenvectors associated to the lower
                     eigenvalues
        grid_dxy   : integer
                     Maximum spatial shift for the grid
        grid_dz    : integer
                     Maximum spectral shift for the grid

        Returns
        -------
        Cat2_T           : astropy.Table
                           Catalogue of parameters of detected emission lines.
                           Columns of the Catalogue Cat2:
                           x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual
                           flux num_line
        Cat_est_line : list of `~mpdaf.obj.Spectrum`
                        Estimated lines
        """
        self._logger.info('ORIGIN - Lines estimation')
        self.param['grid_dxy'] = grid_dxy
        self.param['grid_dz'] = grid_dz
        Cat2_T, Cat_est_line_raw_T, Cat_est_line_std_T = \
            Estimation_Line(Cat1_T, profile._data, self.Nx, self.Ny, self.Nz, self.var, cube_faint._data,
                            grid_dxy, grid_dz, self.PSF, self.wfields, self.profiles)
        Cat_est_line = []
        for data, std in zip(Cat_est_line_raw_T, Cat_est_line_std_T):
            spe = Spectrum(data=data, var=std**2, wave=self.wave, mask=np.ma.nomask)
            Cat_est_line.append(spe)
        return Cat2_T, Cat_est_line

    def merge_spatialy(self, Cat2_T):
        """Construct a catalogue of sources by spatial merging of the
        detected emission lines in a circle with a diameter equal to
        the mean over the wavelengths of the FWHM of the FSF.

        Parameters
        ----------
        Cat2_T   : astropy.Table
                   catalogue
                   Columns of Cat2_T:
                   x y z T_GLR profile pvalC pvalS pvalF T1 T2
                   residual flux num_line

        Returns
        -------
        Cat3 : astropy.Table
               Columns of Cat3:
               ID x_circle y_circle x_centroid y_centroid nb_lines
               x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual flux num_line
        """
        self._logger.info('ORIGIN - Spatial merging')
        if self.wfields is None:
            fwhm = self.FWHM_PSF
        else:
            fwhm = np.max(np.array(self.FWHM_PSF)) # to be improved !!
        Cat3 = Spatial_Merging_Circle(Cat2_T, fwhm)
        return Cat3

    def merge_spectraly(self, Cat3, Cat_est_line, deltaz=1):
        """Merge the detected emission lines distants to less than deltaz
           spectral channel in each group.

        Parameters
        ----------
        Cat3         : astropy.Table
                       Catalogue of detected emission lines
                       Columns of Cat:
                       ID x_circle y_circle x_centroid y_centroid nb_lines
                       x y z T_GLR profile pvalC pvalS pvalF T1 T2
                       residual flux num_line
        Cat_est_line : list of `~mpdaf.obj.Spectrum`
                       List of estimated lines
        deltaz       : integer
                       Distance maximum between 2 different lines

        Returns
        -------
        Cat4 : astropy.Table
               Catalogue
               Columns of Cat4:
               ID x_circle y_circle x_centroid y_centroid nb_lines
               x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual flux num_line
        """
        self._logger.info('ORIGIN - Spectral merging')
        self.param['deltaz'] = deltaz
        Cat_est_line_raw = [spe._data for spe in Cat_est_line]
        Cat4 = Spectral_Merging(Cat3, Cat_est_line_raw, deltaz)
        return Cat4

    def write_sources(self, Cat4, Cat_est_line, correl, name='origin', path='.', overwrite=True, fmt='default', src_vers='0.1', author='undef',ncpu=1):
        """add corresponding RA/DEC to each referent pixel of each group and
        write the final sources.


        Parameters
        ----------
        Cat4             : astropy.Table
                           Catalogue of the detected emission lines:
                           ID x_circle y_circle x_centroid y_centroid
                           nb_lines x y z T_GLR profile pvalC pvalS pvalF
                           T1 T2 residual flux num_line
        Cat_est_line : list of `~mpdaf.obj.Spectrum`
                           List of estimated lines
        correl           : `~mpdaf.obj.Cube`
                           Cube of T_GLR values
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
        # Add RA-DEC to the catalogue
        self._logger.info('ORIGIN - Add RA-DEC to the catalogue')
        CatF_radec = Add_radec_to_Cat(Cat4, self.wcs)

        # path
        if not os.path.exists(path):
            raise IOError("Invalid path: {0}".format(path))

        path = os.path.normpath(path)

        path2 = path + '/' + name
        if not os.path.exists(path2):
            os.makedirs(path2)
        else:
            if overwrite:
                shutil.rmtree(path2)
                os.makedirs(path2)

        # list of source objects
        self._logger.info('ORIGIN - Create the list of sources')
        nsources = Construct_Object_Catalogue(CatF_radec, Cat_est_line,
                                              correl._data, self.wave,
                                              self.filename, self.FWHM_profiles,
                                              path2, name, self.param,
                                              src_vers, author,ncpu)

        return nsources

    def plot(self, correl, x, y, circle=False, vmin=0, vmax=30, title=None, ax=None):
        """Plot detected emission lines on the 2D map of maximum of the T_GLR
        values over the spectral channels.

        Parameters
        ----------
        correl : `~mpdaf.obj.Cube`
                 Cube of T_GLR values
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
        if self.wfields is None:
            fwhm = self.FWHM_PSF
        else:
            fwhm = np.max(np.array(self.FWHM_PSF))
            
        carte_2D_correl = np.amax(correl._data, axis=0)
        carte_2D_correl_ = Image(data=carte_2D_correl, wcs=self.wcs)

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.plot(x, y, 'k+')
        if circle:
            for px, py in zip(x, y):
                c = plt.Circle((px, py), np.round(fwhm / 2), color='k',
                               fill=False)
                ax.add_artist(c)
        carte_2D_correl_.plot(vmin=vmin, vmax=vmax, title=title, ax=ax)
        
        
    def merge_extended_objects(self, incat, ima, fwhm=0.7, threshold=2.0, kernel_size=3, minsize=5.0):
        self._logger.info('ORIGIN - Extended Objects Merging')
        
        from photutils import detect_sources, source_properties, properties_table
        from astropy.convolution import Gaussian2DKernel
        from astropy.stats import gaussian_fwhm_to_sigma
        
        self._logger.info('Creating segmentation image with threshold {}'.format(threshold))
        nsource_init = len(np.unique(incat['ID']))
        sigma = gaussian_fwhm_to_sigma * fwhm/0.2
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()
        segm = ima.clone()
        segobj = detect_sources(ima.data, threshold, npixels=5, filter_kernel=kernel)
        segm.data = segobj.data
        props = source_properties(ima.data, segm.data)
        tab = properties_table(props)
        self._logger.info('{} detected objects'.format(len(tab)))
        
        stab = tab[tab['area']>minsize**2]
        stab.sort('area')
        stab.reverse()
        self._logger.info('Selected {} sources with linear size > {} spaxels'.format(len(stab),minsize))
        
        k,l = (np.round(incat['y_circle']).astype(int), np.round(incat['x_circle']).astype(int))
        vals = segm.data[k,l]
        
        incat['OLD_ID'] = incat['ID']
        
        for k in stab['id']:
            mask = vals==k
            srclist = incat[mask]
            if len(srclist) > 0:
                #self._logger.debug('merging {} sources for id {}'.format(len(srclist),k))  
                iden = srclist['ID'].min()
                clipflux = np.clip(srclist['flux'],0,np.infty)
                if clipflux.sum() == 0: 
                    self._logger.warning('all line flux are <=0, skipped')
                    continue
                xc = np.average(srclist['x_centroid'], weights=clipflux)
                yc = np.average(srclist['y_centroid'], weights=clipflux) 
                incat['ID'][mask] = iden
                incat['x_centroid'][mask] = xc
                incat['y_centroid'][mask] = yc  
        self._logger.debug('Recreate IDs for the catalog')
        current_ids = np.unique(incat['ID'])
        current_ids.sort()
        for k,iden in enumerate(current_ids):
            mask = incat['ID'] == iden
            incat['ID'][mask] = k + 1
        nsource_final = len(np.unique(incat['ID']))   
        nmerged = nsource_init - nsource_final
        self._logger.info('{}/{} sources merged in catalog [{}]'.format(nmerged,nsource_init,nsource_final))
            
        return incat, segm
        