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

lib_origin.py contains the methods that compose the ORIGIN software
"""

from __future__ import absolute_import, division

import astropy.units as u
import logging
import numpy as np
import os.path
import sys
import time

from astropy.table import Table, Column, join
from astropy.utils.console import ProgressBar
from joblib import Parallel, delayed
from scipy import signal, stats, special
from scipy.ndimage import measurements, morphology
from scipy.spatial import KDTree
from six.moves import range, zip

from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.sdetect import Source

__version__ = 'ORIGIN_18122015_02'

def Spatial_Segmentation(Nx, Ny, NbSubcube):
    """Function to compute the limits in pixels for each zone.
    Each zone is computed from the left to the right and the top to the bottom
    First pixel of the first zone has coordinates : (row,col) = (Nx,1).

    Parameters
    ----------
    Nx        : integer
                Number of columns
    Ny        : integer
                Number of rows
    NbSubcube : integer
                Number of subcubes for the spatial segmentation

    Returns
    -------
    intx, inty : integer, integer
                  limits in pixels of the columns/rows for each zone

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # Segmentation of the rows vector in Nbsubcube parts from the right to the
    # left
    inty = np.linspace(Ny, 0, NbSubcube + 1, dtype=np.int)
    # Segmentation of the columns vector in Nbsubcube parts from the left to
    # the right
    intx = np.linspace(0, Nx, NbSubcube + 1, dtype=np.int)
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return inty, intx


def Compute_PCA_SubCube(NbSubcube, cube_std, intx, inty, Edge_xmin, Edge_xmax,
                        Edge_ymin, Edge_ymax):
    """Function to compute the PCA on each zone of a data cube.

    Parameters
    ----------
    NbSubcube : integer
                Number of subcubes for the spatial segmentation
    cube_std  : array
                Cube data weighted by the standard deviation
    intx      : integer
                limits in pixels of the columns for each zone
    inty      : integer
                limits in pixels of the rows for each zone
    Edge_xmin : int
                Minimum limits along the x-axis in pixel
                of the data taken to compute p-values
    Edge_xmax : int
                Maximum limits along the x-axis in pixel
                of the data taken to compute p-values
    Edge_ymin : int
                Minimum limits along the y-axis in pixel
                of the data taken to compute p-values
    Edge_ymax : int
                Maximum limits along the y-axis in pixel
                of the data taken to compute p-values

    Returns
    -------
    A       : dict
              Projection of the data on the eigenvectors basis
    V       : dict
              Eigenvectors basis
    eig_val : dict
              Eigenvalues computed for each spatio-spectral zone
    nx      : array
              Number of columns for each spatio-spectral zone
    ny      : array
              Number of rows for each spatio-spectral zone
    nz      : array
              Number of spectral channels for each spatio-spectral zone

    Date  : Dec,7 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # Initialization
    nx = np.empty((NbSubcube, NbSubcube), dtype=np.int)
    ny = np.empty((NbSubcube, NbSubcube), dtype=np.int)
    nz = np.empty((NbSubcube, NbSubcube), dtype=np.int)
    eig_val = {}
    V = {}
    A = {}

    # Spatial segmentation
    with ProgressBar(NbSubcube**2) as bar:
        for numy in range(NbSubcube):
            for numx in range(NbSubcube):
                bar.update()
                # limits of each spatial zone
                x1 = intx[numx]
                x2 = intx[numx + 1]
                y2 = inty[numy]
                y1 = inty[numy + 1]
                # Data in this spatio-spectral zone
                cube_temp = cube_std[:, y1:y2, x1:x2]

                # Edges are excluded for PCA computing
                x1 = max(x1, Edge_xmin + 1)
                x2 = min(x2, Edge_xmax)
                y1 = max(y1, Edge_ymin + 1)
                y2 = min(y2, Edge_ymax)
                cube_temp_edge = cube_std[:, y1:y2, x1:x2]

                # Dimensions of each subcube of each spatio-spectral zone
                nx[numx, numy] = cube_temp.shape[2]
                ny[numx, numy] = cube_temp.shape[1]
                nz[numx, numy] = cube_temp.shape[0]

                # PCA on each subcube
                A_c, V_c, lambda_c = Compute_PCA_edge(cube_temp, cube_temp_edge)
                # eigenvalues for each spatio-spectral zone
                eig_val[(numx, numy)] = lambda_c
                # Eigenvectors basis for each spatio-spectral zone
                V[(numx, numy)] = V_c
                # Projection of the data on the eigenvectors basis
                # for each spatio-spectral zone
                A[(numx, numy)] = A_c

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return A, V, eig_val, nx, ny, nz


def Compute_PCA_edge(cube, cube_edge):
    """Function to compute the PCA the spectra of a data cube by excluding
    the undesired spectra.

    Parameters
    ----------
    cube      : array
                cube data weighted by the standard deviation
    cube_edge : array
                cube data weighted by the standard deviation without the
                undesired spectra (ie those on the edges).

    Returns
    -------
    A       : array
              Projection of the data cube on the eigenvectors basis
    eig_vec : array
              Eigenvectors basis corrsponding to the eigenvalues
    eig_val : array
              Eigenvalues computed for each spatio-spectral zone

    Date  : Dec,3 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # data cube converted to dictionary of spectra
    cube_v = cube.reshape(cube.shape[0], cube.shape[1] * cube.shape[2])
    # data cube without undesired spectra converted to dictionary of spectra
    cube_ve = cube_edge.reshape(cube_edge.shape[0],
                                cube_edge.shape[1] * cube_edge.shape[2])
    # Spectral covariance of the desired spectra
    C = np.cov(cube_ve)
    # Eigenvalues (ascending order) and Eigenvectors basis
    eig_val, eig_vec = np.linalg.eigh(C)
    # Projection of the data cube on the eigenvectors basis
    A = eig_vec.T.dot(cube_v)
    return A, eig_vec, eig_val


def Compute_Number_Eigenvectors_Zone(NbSubcube, list_r0, eig_val):
    """Function to compute the number of eigenvectors to keep for the
    projection for each zone by calling the function
    Compute_Number_Eigenvectors.

    Parameters
    ----------
    NbSubcube   : float
                  Number of subcube in the spatial segementation
    list_r0     : array
                  List of the determination coefficient for each zone
    eig_val     : dict
                  eigenvalues of each spatio-spectral zone
    fig : figure instance
                  if not None, plot the eigenvalues and the separation
                  point

    Returns
    -------
    nbkeep : array
             number of eigenvalues for each zone used to compute the projection

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # Initialization
    nbkeep = np.empty((NbSubcube, NbSubcube), dtype=np.int)
    zone = 0
    for numy in range(NbSubcube):
        for numx in range(NbSubcube):

            # Eigenvalues for this zone
            lambdat = eig_val[(numx, numy)]

            # Number of eigenvalues per zone
            nbkeep[numx, numy] = Compute_Number_Eigenvectors(lambdat,
                                                             list_r0[zone])
            zone = zone + 1

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return nbkeep


def Compute_Number_Eigenvectors(eig_val, r0):
    """Function to compute the number of eigenvectors to keep for the
    projection with a linear regression and its associated determination
    coefficient

    Parameters
    ----------
    eig_val : array
              eigenvalues of each zone
    r0      : float
              Determination coefficient value set by the user

    Returns
    -------
    nbkeep : float
             number of eigenvalues used to compute the projection

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # Initialization
    nl = eig_val.shape[0]
    coeffr = np.zeros(nl - 4)

    # Start with the 5 first eigenvalues for the linear regression
    for r in range(5, nl + 1):
        Y = np.log(eig_val[:r] + 0j)
        # Y[np.isnan(Y)] = 0
        X = np.array([np.ones(r), eig_val[:r]])
        beta = np.linalg.lstsq(X.T, Y)[0]
        Yest = np.dot(X.T, beta)
        # Determination coefficient
        Y = np.real(Y)
        Yest = np.real(Yest)
        coeffr[r - 5] = 1 - (np.sum((Y - Yest)**2) /
                             np.sum((Y - np.mean(Y))**2))

    # Find the coefficient closer of r0
    rt = 4 + np.where(coeffr >= r0)[0]
    if rt.shape[0] == 0:
        return 0
    else:
        return rt[-1]


def Compute_Proj_Eigenvector_Zone(nbkeep, NbSubcube, Nx, Ny, Nz, A, V,
                                  nx, ny, nz, inty, intx):
    """Function to compute the projection on the selected eigenvectors of the
    data cube in the original basis by calling the function
    Compute_Proj_Eigenvector.

    Parameters
    ----------
    nbkeep    : array
                number of eigenvalues for each zone used to compute the
                projection
    NbSubcube : int
                Number of subcube in the spatial segementation
    Nx        : int
                Size of the cube along the x-axis
    Ny        : int
                Size of the cube along the z-axis
    Nz        : int
                Size of the cube along the spectral axis
    A         : dict
                Projection of the data on the eigenvectors basis
    V         : dict
                Eigenvectors basis
    nx        : array
                Number of columns for each spatio-spectral zone
    ny        : array
                Number of rows for each spatio-spectral zone
    nz        : array
                Number of spectral channels for each spatio-spectral zone
    intx      : array
                limits in pixels of the columns for each zone
    inty      : array
                limits in pixels of the rows for each zone

    Returns
    -------
    cube_faint : array
                 Projection on the eigenvectors associated to the lower
                 eigenvalues of the data cube (reprensenting the faint signal)
    cube_cont  : array
                 Projection on the eigenvectors associated to the higher
                 eigenvalues of the data cube (representing the continuum)

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # initialization
    cube_faint = np.zeros((Nz, Ny, Nx))
    cube_cont = np.zeros((Nz, Ny, Nx))

    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone
            x1 = intx[numx]
            x2 = intx[numx + 1]
            y2 = inty[numy]
            y1 = inty[numy + 1]
            At = A[(numx, numy)]
            Vt = V[(numx, numy)]
            r = nbkeep[numx, numy]
            cube_proj_faint_v, cube_proj_cont_v = \
                Compute_Proj_Eigenvector(At, Vt, r)
            # Resize the subcube
            cube_faint[:, y1:y2, x1:x2] = \
                cube_proj_faint_v.reshape((nz[numx, numy],
                                           ny[numx, numy],
                                           nx[numx, numy]))
            cube_cont[:, y1:y2, x1:x2] = \
                cube_proj_cont_v.reshape((nz[numx, numy],
                                          ny[numx, numy],
                                          nx[numx, numy]))
#             cube_faint[cube_faint==np.NaN] = 0
#             cube_cont[cube_cont==np.NaN] = 0
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return cube_faint, cube_cont


def Compute_Proj_Eigenvector(A, V, r):
    """Function to compute the projection of the data in the original basis
    keepping the desired number eigenvalues.

    Parameters
    ----------
    A : array
        Projection of the data on the eigenvectors basis
    V : array
        Eigenvectors basis
    r : float
        Number of eigenvalues to keep for the projection

    Returns
    -------
    cube_proj_low_v  : array
                       Projection on the eigenvectors associated to the lower
                       eigenvalues of the spectra.
    cube_proj_high_v : array
                       Projection on the eigenvectors associated to the higher
                       eigenvalues of the spectra.

    Date  : Dec,7 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # initialization
    cube_proj_low_v = np.dot(V[:, :r + 1], A[:r + 1, :])
    cube_proj_high_v = np.dot(V[:, r + 1:], A[r + 1:, :])
    return cube_proj_low_v, cube_proj_high_v


def Correlation_GLR_test(cube, sigma, PSF_Moffat, weights, Dico):
    """Function to compute the cube of GLR test values obtained with the given
    PSF and dictionary of spectral profile.

    Parameters
    ----------
    cube       : array
                 data cube on test
    sigma      : array
                 MUSE covariance
    PSF_Moffat : list of arrays
                 FSF for each field of this data cube
    weights    : list of array
                 Weight maps of each field
    Dico       : array
                 Dictionary of spectral profiles to test

    Returns
    -------
    correl  : array
              cube of T_GLR values
    profile : array
              Number of the profile associated to the T_GLR

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # data cube weighted by the MUSE covariance
    cube_var = cube / np.sqrt(sigma)
    # Inverse of the MUSE covariance
    inv_var = 1. / sigma

    # Dimensions of the data
    shape = cube_var.shape
    Nz = cube_var.shape[0]
    Ny = cube_var.shape[1]
    Nx = cube_var.shape[2]

    cube_fsf = np.empty(shape)
    norm_fsf = np.empty(shape)
    if weights is None: # one FSF
        # Spatial convolution of the weighted data with the zero-mean FSF
        logger.info('Step 1/4 Spatial convolution of the weighted data with the '
                'zero-mean FSF')
        PSF_Moffat_m = PSF_Moffat \
            - np.mean(PSF_Moffat, axis=(1, 2))[:, np.newaxis, np.newaxis]
        for i in ProgressBar(list(range(Nz))):
            cube_fsf[i, :, :] = signal.fftconvolve(cube_var[i, :, :],
                                                   PSF_Moffat_m[i, :, :][::-1, ::-1],
                                                   mode='same')
        del cube_var
        fsf_square = PSF_Moffat_m**2
        del PSF_Moffat_m
        # Spatial part of the norm of the 3D atom
        logger.info('Step 2/4 Computing Spatial part of the norm of the 3D atoms')
        for i in ProgressBar(list(range(Nz))):
            norm_fsf[i, :, :] = signal.fftconvolve(inv_var[i, :, :],
                                                   fsf_square[i, :, :][::-1, ::-1],
                                                   mode='same')
        del fsf_square, inv_var
    else: # several FSF
        # Spatial convolution of the weighted data with the zero-mean FSF
        logger.info('Step 1/4 Spatial convolution of the weighted data with the '
                'zero-mean FSF')
        nfields = len(PSF_Moffat)
        PSF_Moffat_m = []
        for n in range(nfields):
            PSF_Moffat_m.append(PSF_Moffat[n] \
            - np.mean(PSF_Moffat[n], axis=(1, 2))[:, np.newaxis, np.newaxis])
        # build a weighting map per PSF and convolve
        cube_fsf = np.empty(shape)
        for i in ProgressBar(list(range(Nz))):
            cube_fsf[i, :, :] = 0
            for n in range(nfields):
                cube_fsf[i, :, :] = cube_fsf[i, :, :] \
                        + signal.fftconvolve(weights[n]*cube_var[i, :, :],
                                             PSF_Moffat_m[n][i, :, :][::-1, ::-1],
                                            mode='same')
        del cube_var
        fsf_square = []
        for n in range(nfields):
            fsf_square.append(PSF_Moffat_m[n]**2)
        del PSF_Moffat_m
        # Spatial part of the norm of the 3D atom
        logger.info('Step 2/4 Computing Spatial part of the norm of the 3D atoms')
        for i in ProgressBar(list(range(Nz))):
            norm_fsf[i, :, :] = 0
            for n in range(nfields):
                norm_fsf[i, :, :] = norm_fsf[i, :, :] \
                + signal.fftconvolve(weights[n]*inv_var[i, :, :],
                                    fsf_square[n][i, :, :][::-1, ::-1],
                                    mode='same')

    # First cube of correlation values
    # initialization with the first profile
    profile = np.zeros(shape, dtype=np.int)

    # First spectral profile
    k0 = 0
    d_j = Dico[k0]
    # zero-mean spectral profile
    d_j = d_j - np.mean(d_j)
    # Compute the square of the spectral profile
    profile_square = d_j**2

    ygrid, xgrid = np.mgrid[0:Ny, 0:Nx]
    xgrid = xgrid.flatten()
    ygrid = ygrid.flatten()

    cube_profile = np.empty(shape)
    norm_profile = np.empty(shape)

    logger.info('Step 3/4 Spectral convolution of the weighted datacube')
    with ProgressBar(Nx * Ny) as bar:
        for y in range(Ny):
            for x in range(Nx):
                bar.update()
                # Spectral convolution of the weighted data cube spreaded
                # by the FSF and the spectral profile : correlation between the
                # data and the 3D atom
                cube_profile[:,y,x] = signal.fftconvolve(cube_fsf[:,y,x], d_j,
                                                          mode = 'same')
                # Spectral convolution between the spatial part of the norm of the
                # 3D atom and the spectral profile : The norm of the 3D atom
                norm_profile[:,y,x] = signal.fftconvolve(norm_fsf[:,y,x],
                                                          profile_square,
                                                          mode = 'same')

    # Set to the infinity the norm equal to 0
    norm_profile[norm_profile <= 0] = np.inf
    # T_GLR values with constraint  : cube_profile>0
    GLR = np.zeros((Nz, Ny, Nx, 2))
    GLR[:, :, :, 0] = cube_profile / np.sqrt(norm_profile)

    logger.info('Step 4/4 Computing second cube of correlation values')

    for k in ProgressBar(list(range(1, len(Dico)))):
        # Second cube of correlation values
        d_j = Dico[k]
        d_j = d_j - np.mean(d_j)
        profile_square = d_j**2

        i = 0
        for y in range(Ny):
            for x in range(Nx):
                cube_profile[:,y,x] = signal.fftconvolve(cube_fsf[:,y,x], d_j,
                                                         mode = 'same')
                norm_profile[:,y,x] = signal.fftconvolve(norm_fsf[:,y,x],
                                                         profile_square,
                                                         mode = 'same')

        norm_profile[norm_profile <= 0] = np.inf
        GLR[:, :, :, 1] = cube_profile / np.sqrt(norm_profile)

        # maximum over the fourth dimension
        PROFILE_MAX = np.argmax(GLR, axis=3)
        correl = np.amax(GLR, axis=3)
        # Number of corresponding real profile
        profile[PROFILE_MAX == 1] = k
        # Set the first cube of correlation values correspond
        # to the maximum of the two previous ones
        GLR[:, :, :, 0] = correl

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return correl, profile


def Compute_pval_correl_zone(correl, intx, inty, NbSubcube, Edge_xmin,
                             Edge_xmax, Edge_ymin, Edge_ymax, threshold):
    """Function to compute the p-values associated to the
    T_GLR values for each zone

    Parameters
    ----------
    correl    : array
                cube of T_GLR values (correlations)
    intx      : array
                limits in pixels of the columns for each zone
    inty      : array
                limits in pixels of the rows for each zone
    NbSubcube : int
                Number of subcube in the spatial segementation
    Edge_xmin : int
                Minimum limits along the x-axis in pixel
                of the data taken to compute p-values
    Edge_xmax : int
                Maximum limits along the x-axis in pixel
                of the data taken to compute p-values
    Edge_ymin : int
                Minimum limits along the y-axis in pixel
                of the data taken to compute p-values
    Edge_ymax : int
                Maximum limits along the y-axis in pixel
                of the data taken to compute p-values
    threshold : float
                The threshold applied to the p-values cube

    Returns
    -------
    cube_pval_correl : array
                       cube of thresholded p-values associated
                       to the T_GLR values

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # initialization
    cube_pval_correl = np.ones(correl.shape)

    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone
            x1 = intx[numx]
            x2 = intx[numx + 1]
            y2 = inty[numy]
            y1 = inty[numy + 1]

            # Edges are excluded for computing parameters of the
            # distribution of the T_GLR (mean and std)
            x1 = max(x1, Edge_xmin + 1)
            x2 = min(x2, Edge_xmax)
            y1 = max(y1, Edge_ymin + 1)
            y2 = min(y2, Edge_ymax)
            correl_temp_edge = correl[:, y1:y2, x1:x2]

            # Cube of pvalues for each zone
            cube_pval_correl_temp = Compute_pval_correl(correl_temp_edge)
            cube_pval_correl[:, y1:y2, x1:x2] = cube_pval_correl_temp

    # Threshold the pvalues
    threshold_log = 10**(-threshold)
    cube_pval_correl[cube_pval_correl >= threshold_log] = 1

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return cube_pval_correl


def Compute_pval_correl(correl_temp_edge):
    """Function to compute distribution of the T_GLR values with
    hypothesis : T_GLR are distributed according a normal distribution

    Parameters
    ----------
    correl_temp_edge : T_GLR values with edges excluded

    Returns
    -------
    cube_pval_correl : array
                       p-values asssociated to the T_GLR values

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    moy_est = np.mean(correl_temp_edge)
    std_est = np.std(correl_temp_edge)
    # hypothesis : T_GLR are distributed according a normal distribution
    rv = stats.norm(loc=moy_est, scale=std_est)
    cube_pval_correl = 1 - rv.cdf(correl_temp_edge)

    return cube_pval_correl


def Compute_pval_channel_Zone(cube_pval_correl, intx, inty, NbSubcube,
                              mean_est, weights):
    """Function to compute the p-values associated to the number of
    thresholded p-values of the correlations per spectral channel for
    each zone by calling the function Compute_pval_channel

    Parameters
    ----------
    cube_pval_correl : array
                       cube of thresholded p-values associated
                       to the T_GLR values
    intx             : array
                       limits in pixels of the columns for each zone
    inty             : array
                       limits in pixels of the rows for each zone
    NbSubcube        : int
                       Number of subcube in the spatial segmentation
    mean_est         : float
                       Estimated mean of the distribution

    Returns
    -------
    cube_pval_channel : array
                        cube of p-values associated to the number of
                        thresholded p-values of the correlations per spectral
                        channel for each zone

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # initialization
    cube_pval_channel = np.zeros(cube_pval_correl.shape)
    cube_pval_correl_threshold = cube_pval_correl.copy()

    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone
            x1 = intx[numx]
            x2 = intx[numx + 1]
            y2 = inty[numy]
            y1 = inty[numy + 1]

            if weights is None:
                m = mean_est
            else:
                w = np.array([np.sum(weights[n][y1:y2, x1:x2]) for n in range(len(weights))])
                w /= np.sum(w)
                m = np.sum(w*np.array(mean_est))

            X = cube_pval_correl_threshold[:, y1:y2, x1:x2]

            # How many thresholded pvalues in each spectral channel
            n_lambda = np.sum(np.array(X != 1, dtype=np.int), axis=(1, 2))
            # pvalues computed for each spectral channel
            pval_channel_temp = Compute_pval_channel(X, n_lambda, m)
            # cube of p-values
            cube_pval_channel[:, y1:y2, x1:x2] = pval_channel_temp[:, np.newaxis,
                                                                   np.newaxis]
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return cube_pval_channel


def Compute_pval_channel(X, n_lambda, mean_est):
    """Function to compute the p-values associated to the
    number of thresholded p-values of the correlations per spectral channel

    Parameters
    ----------
    X        : array
               number of thresholded p-values associated to the T_GLR values
               per spectral channel
    n_lambda : int
               How many thresholded pvalues in each spectral channel
    mean_est : float
               Estimated mean of the distribution given by the FWHM of the FSF.

    Returns
    -------
    cube_pval_channel : array
                        cube of p-values for each spectral channel

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # initialization
    N = np.sum(np.array(X != 1, dtype=np.int))
    # Estimation of p parameter with the mean of the distribution set by the
    # FSF size
    p_est = mean_est / N
    # Hypothesis : Binomial distribution for each channel
    pval_channel = special.bdtr(N - 1, N, p_est) - \
        special.bdtr(n_lambda, N, p_est)
    pval_channel[pval_channel <= 0] = 0
    return pval_channel


def Compute_pval_final(cube_pval_correl, cube_pval_channel, threshold, sky):

    """Function to compute the final p-values which are the thresholded
    pvalues associated to the T_GLR values divided by twice the pvalues
    associated to the number of thresholded p-values of the correlations
    per spectral channel for each zone

    The pvalues equals to zero correspond to the values flag to zero because
    they are higher than the threshold

    Parameters
    ----------
    cube_pval_correl  : array
                        cube of thresholded p-values associated
                        to the T_GLR values
    cube_pval_channel : array
                        cube of p-values
    threshold         : float
                        The threshold applied to the p-values cube
    sky               : Bool
                        enable or disable the channel pvalue to compute the
                        final pvalue in the normalization process.

    Returns
    -------
    cube_pval_final : array
                      cube of final thresholded p-values

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    Date  : Nov,23 2016
    Modifed: Antony Schutz
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # probability : Pr(line|not nuisance) = Pr(line)/Pr(not nuisance)
    ksel_correl = (cube_pval_correl==0)
    if sky:
        ksel_channel = (cube_pval_channel==0)
    # Set the pvalues equals to zero to an arbitrary very low value, but not
    # zero
    cube_pval_correl[ksel_correl] = np.spacing(1)**6
    if sky:
        cube_pval_channel[ksel_channel] = np.spacing(1)**6
    probafinale = cube_pval_correl
    if sky:
        probafinale /= cube_pval_channel

    # # this is not used after
    # cube_pval_correl[ksel_correl] = 0
    # cube_pval_channel[ksel_channel] = 0

    # pvalue = probability/2
    cube_pval_final = probafinale / 2
    # Set the nan to 1
    cube_pval_final[np.isnan(cube_pval_final)] = 1
    # Threshold the p-values
    threshold_log = 10**(-threshold)
    cube_pval_final = cube_pval_final * (cube_pval_final < threshold_log)
    # The pvalues equals to zero correspond to the values flag to zero because
    # they are higher than the threshold so actually they have to be set to 1
    cube_pval_final[cube_pval_final == 0] = 1
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return cube_pval_final


def Compute_Connected_Voxel(cube_pval_final, neighboors):
    """Function to compute the groups of connected voxels with a
    flood-fill algorithm.

    Parameters
    ----------
    cube_pval_final : array
                      cube of final thresholded p-values
    neighboors      : int
                      Number of connected components

    Returns
    -------
    labeled_cube : array
                   An integer array where each unique feature in
                   cube_pval_final has a unique label.
    Ngp          : integer
                   Number of groups

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
#     threshold_log = 10**(-threshold)
#     # The p-values higher than the thresholded are previously set to 1, here we
#     # set them to 0 because we want to merge in group pvalues thresholded.
#     cube_pval_final = cube_pval_final*(cube_pval_final<threshold_log)
    cube_pval_final[cube_pval_final == 1] = 0

    # connected components
    conn = (neighboors + 1)**(1 / 3.)
    s = morphology.generate_binary_structure(3, conn)
    labeled_cube, Ngp = measurements.label(cube_pval_final, structure=s)
    # Maximum number of voxels in one group
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return labeled_cube, Ngp


def Compute_Referent_Voxel(correl, profile, cube_pval_correl,
                           cube_pval_channel, cube_pval_final, Ngp,
                           labeled_cube, wcs, wave):
    """Function to compute refrerent voxel of each group of connected voxels
    using the voxel with the higher T_GLR value.

    Parameters
    ----------
    correl            : array
                        cube of T_GLR values
    profile           : array
                        Number of the profile associated to the T_GLR
    cube_pval_correl  : array
                        cube of thresholded p-values associated
                        to the T_GLR values
    cube_pval_channel : array
                        cube of p-values
    cube_pval_final   : array
                        cube of final thresholded p-values
    Ngp               : int
                        Number of groups
    labeled_cube      : array
                        An integer array where each unique feature in
                        cube_pval_final has a unique label.
    wcs               : `mpdaf.obj.WCS`
                         RA-DEC coordinates.
    wave              : `mpdaf.obj.WaveCoord`
                         Spectral coordinates.

    Returns
    -------
    Cat_ref : astropy.Table
              Catalogue of the referent voxels coordinates for each group
              Columns of Cat_ref : x y z ra dec lba,
                                   T_GLR profile pvalC pvalS pvalF

    Date  : Dec,16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    grp = measurements.find_objects(labeled_cube)
    argmax = [np.argmax(correl[grp[i]]) for i in range(Ngp)]
    correl_max = np.array([np.ravel(correl[grp[i]])[argmax[i]]
                           for i in range(Ngp)])
    z, y, x = np.meshgrid(list(range(correl.shape[0])),
                          list(range(correl.shape[1])),
                          list(range(correl.shape[2])), indexing='ij')
    zpixRef = np.array([np.ravel(z[grp[i]])[argmax[i]] for i in range(Ngp)])
    ypixRef = np.array([np.ravel(y[grp[i]])[argmax[i]] for i in range(Ngp)])
    xpixRef = np.array([np.ravel(x[grp[i]])[argmax[i]] for i in range(Ngp)])
    profile_max = profile[zpixRef, ypixRef, xpixRef]
    pvalC = cube_pval_correl[zpixRef, ypixRef, xpixRef]
    pvalS = cube_pval_channel[zpixRef, ypixRef, xpixRef]
    pvalF = cube_pval_final[zpixRef, ypixRef, xpixRef]
    # add real coordinates
    pixcrd = [[p, q] for p, q in zip(ypixRef, xpixRef)]
    skycrd = wcs.pix2sky(pixcrd)
    ra = skycrd[:, 1]
    dec = skycrd[:, 0]
    lbda = wave.coord(zpixRef)
    # Catalogue of referent pixels
    Cat_ref = Table([xpixRef, ypixRef, zpixRef, ra, dec, lbda, correl_max,
                     profile_max, pvalC, pvalS, pvalF],
                    names=('x', 'y', 'z', 'ra', 'dec', 'lbda', 'T_GLR',
                           'profile', 'pvalC', 'pvalS', 'pvalF'))
    # Catalogue sorted along the Z axis
    Cat_ref.sort('z')
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return Cat_ref


def Narrow_Band_Test(Cat0, cube_raw, Dico, PSF_Moffat, weights,
                     nb_ranges, wcs):
    """Function to compute the 2 narrow band tests for each detected
    emission line

    Parameters
    ----------
    Cat0        : astropy.Table
                  Catalogue of parameters of detected emission lines:
                  Columns of the Catalogue Cat0 :
                  x y z T_GLR profile pvalC pvalS pvalF
    cube_raw    : array
                  Raw data cube
    Dico        : array
                  Dictionary of spectral profiles to test
    PSF_Moffat  : array or list of arrays
                  FSF for this data cube
    nb_ranges   : integer
                  Number of skipped intervals for computing control cube
    wcs         : `mpdaf.obj.WCS`
                  Spatial coordinates

    Returns
    -------
    Cat1 : astropy.Table
           Catalogue of parameters of detected emission lines:
           Columns of the Catalogue Cat1 :
           x y z ra dec lbda T_GLR profile pvalC pvalS pvalF T1 T2

    Date : Dec,16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # Initialization
    T1 = []
    T2 = []

    for i in range(len(Cat0)):
        # Coordinates of the voxel
        x0 = Cat0[i]['x']
        y0 = Cat0[i]['y']
        z0 = Cat0[i]['z']
        # FSF
        if weights is None:
            FSF = PSF_Moffat
        else:
            FSF = np.sum(np.array([weights[n][y0,x0]*PSF_Moffat[n] for n in range(len(weights))]), axis=0)

        # spectral profile
        num_prof = Cat0[i]['profile']
        profil0 = Dico[num_prof]
        # length of the spectral profile
        profil1 = profil0[profil0 > 1e-13]
        long0 = profil1.shape[0]
        # half-length of the spectral profile
        longz = long0 // 2
        # spectral range
        intz1 = max(0, z0 - longz)
        intz2 = min(cube_raw.shape[0], z0 + longz + 1)
        # Subcube on test
        longxy = FSF.shape[1] // 2
        inty1 = max(0, y0 - longxy)
        inty2 = min(cube_raw.shape[1], y0 + longxy + 1)
        intx1 = max(0, x0 - longxy)
        intx2 = min(cube_raw.shape[2], x0 + longxy + 1)
        cube_test = cube_raw[intz1:intz2, inty1:inty2, intx1:intx2]

        # controle cube
        if (z0 + longz + nb_ranges * long0) < cube_raw.shape[0]:
            intz1c = intz1 + nb_ranges * long0
            intz2c = intz2 + nb_ranges * long0
        else:
            intz1c = intz1 - nb_ranges * long0
            intz2c = intz2 - nb_ranges * long0
        cube_controle = cube_raw[intz1c:intz2c, inty1:inty2, intx1:intx2]

        # (1/sqrt(2)) * difference of the 2 sububes
        diff_cube = (1. / np.sqrt(2)) * (cube_test - cube_controle)

        # Test 1
        s1 = np.ones_like(cube_test)
        s1 = s1 / np.sqrt(np.sum(s1**2))
        T1.append(np.inner(diff_cube.flatten(), s1.flatten()))

        # Test 2
        atom = np.zeros((long0, FSF.shape[1], FSF.shape[2]))

        # Construction of the 3D atom corresponding to the spectral profile
        z1 = max(0, z0 - longz)
        z2 = min(cube_raw.shape[0], long0 + z0 - longz)
        atom[z1 - z0 + longz:z2 - z0 + longz, :, :] = profil1[z1 - z0 + longz:z2 - z0 + longz,
                                                              np.newaxis, np.newaxis] \
            * FSF[z1:z2, :, :]

        # Normalization
        atom = atom / np.sqrt(np.sum(atom**2))
        # Edges
        # The minimal coordinates corresponding to the spatio-spectral range
        # of the data cube
        x1 = np.abs(min(0, x0 - longxy))
        y1 = np.abs(min(0, y0 - longxy))
        z1 = np.abs(min(0, z0 - longz))

        # Part of the atom corresponding to the spatio-spectral range of the
        # data cube
        s2 = atom[z1:z1 + intz2 - intz1, y1:y1 + inty2 - inty1, x1:x1 + intx2 - intx1]

        # Test 2
        T2.append(np.inner(diff_cube.flatten(), s2.flatten()))

    col_t1 = Column(name='T1', data=T1)
    col_t2 = Column(name='T2', data=T2)
    Cat1 = Cat0.copy()
    Cat1.add_columns([col_t1, col_t2])
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return Cat1


def Narrow_Band_Threshold(Cat1, thresh_T1, thresh_T2):
    """Function to compute the 2 narrow band tests for each detected
    emission line

    Parameters
    ----------
    Cat1      : astropy.Table
                Catalogue of parameters of detected emission lines:
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

    Columns of the Catalogues :
        [col line; row line; spectral channel line; ra; dec; T_GLR line ;
        spectral profile ; pval T_GLR;  pval channel;  final pval ; T1 ; T2]

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # Catalogue with the rows corresponding to the lines with test values
    # greater than the given threshold
    Cat1_T1 = Cat1[Cat1['T1'] > thresh_T1]
    Cat1_T2 = Cat1[Cat1['T2'] > thresh_T2]
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return Cat1_T1, Cat1_T2


def Estimation_Line(Cat1_T, profile, Nx, Ny, Nz, sigma, cube_faint,
                    grid_dxy, grid_dz, PSF_Moffat, weights, Dico, wcs, wave):
    """Function to compute the estimated emission line and the optimal
    coordinates for each detected lines in a spatio-spectral grid.

    Parameters
    ----------
    Cat1_T     : astropy.Table
                 Catalogue of parameters of detected emission lines selected
                 with a narrow band test.
                 Columns of the Catalogue Cat1_T:
                 x y z T_GLR profile pvalC pvalS pvalF T1 T2
    profile    : array
                 Number of the profile associated to the T_GLR
    Nx         : int
                 Size of the cube along the x-axis
    Ny         : int
                 Size of the cube along the z-axis
    Nz         : int
                 Size of the cube along the spectral axis
    sigma      : array
                 MUSE covariance
    cube_faint : array
                 Projection on the eigenvectors associated to the lower
                 eigenvalues
    grid_dxy   : integer
                 Maximum spatial shift for the grid
    grid_dz    : integer
                 Maximum spectral shift for the grid
    PSF_Moffat : array
                 FSF for this data cube
    Dico       : array
                 Dictionary of spectral profiles to test
    wcs        : `mpdaf.obj.WCS`
                  RA-DEC coordinates.
    wave       : `mpdaf.obj.WaveCoord`
                 Spectral coordinates.

    Returns
    -------
    Cat2             : astropy.Table
                       Catalogue of parameters of detected emission lines.
                       Columns of the Catalogue Cat2:
                       x y z ra dec lbda, T_GLR profile pvalC pvalS pvalF
                       T1 T2 residual flux num_line
    Cat_est_line_raw : list of arrays
                       Estimated lines in data space
    Cat_est_line_std : list of arrays
                       Estimated lines in SNR space

    Date  : Dec, 16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # Initialization
    Cat2_x = []
    Cat2_y = []
    Cat2_z = []
    Cat2_res_min = []
    Cat2_flux = []
    Cat_est_line_raw = []
    Cat_est_line_std = []
    if weights is None:
        longxy = PSF_Moffat.shape[1] // 2
    else:
        longxy = PSF_Moffat[0].shape[1] // 2

    # Spatio-spectral grid
    grid_x1 = np.maximum(0, Cat1_T['x'] - grid_dxy)
    grid_x2 = np.minimum(Nx, Cat1_T['x'] + grid_dxy + 1)
    grid_y1 = np.maximum(0, Cat1_T['y'] - grid_dxy)
    grid_y2 = np.minimum(Ny, Cat1_T['y'] + grid_dxy + 1)
    grid_z1 = np.maximum(0, Cat1_T['z'] - grid_dz)
    grid_z2 = np.minimum(Nz, Cat1_T['z'] + grid_dz + 1)

    ngrid = (grid_x2 - grid_x1) * (grid_y2 - grid_y1) * (grid_z2 - grid_z1)

    # Pad subcube in case of the 3D atom is out of the cube data
    cube_faint_pad = np.zeros((cube_faint.shape[0],
                               cube_faint.shape[1] + 2 * grid_dxy + 2 * longxy,
                               cube_faint.shape[2] + 2 * grid_dxy + 2 * longxy))
    cube_faint_pad[0: cube_faint.shape[0],
                   grid_dxy + longxy: cube_faint.shape[1] + grid_dxy + longxy,
                   grid_dxy + longxy: cube_faint.shape[2] + grid_dxy + longxy] \
        = cube_faint

    # Loop on emission lines detected
    nit = len(Cat1_T)
    col_del = []
    for it in ProgressBar(range(nit)):
        # initialization
        line_est_raw = np.zeros((ngrid[it], Nz))
        line_est_std = np.zeros((ngrid[it], Nz))
        residual = np.zeros(ngrid[it])
        flux = np.zeros(ngrid[it])

        # Estimation of a line on each voxel of the grid
        z_f, y_f, x_f = np.meshgrid(list(range(grid_z1[it], grid_z2[it])),
                                    list(range(grid_y1[it], grid_y2[it])),
                                    list(range(grid_x1[it], grid_x2[it])),
                                    indexing='ij')
        z_f = z_f.ravel()
        y_f = y_f.ravel()
        x_f = x_f.ravel()

        # size of the 3D atom along the spatial axes
        inty1 = np.maximum(0, y_f - longxy)
        inty2 = np.minimum(Ny, y_f + longxy + 1)
        intx1 = np.maximum(0, x_f - longxy)
        intx2 = np.minimum(Nx, x_f + longxy + 1)

        x1 = x_f + grid_dxy
        x2 = x_f + 2 * longxy + grid_dxy + 1
        y1 = y_f + grid_dxy
        y2 = y_f + 2 * longxy + grid_dxy + 1

        xmin = np.abs(np.minimum(0, x_f - longxy))
        ymin = np.abs(np.minimum(0, y_f - longxy))

        for n in range(x_f.shape[0]):
            s = (slice(None, None, 1),
                 slice(inty1[n], inty2[n], 1),
                 slice(intx1[n], intx2[n], 1))
            spad = (slice(None, None, 1),
                    slice(y1[n], y2[n], 1),
                    slice(x1[n], x2[n], 1))

            #FSF
            if weights is None:
                FSF = PSF_Moffat
            else:
                FSF = np.sum(np.array([weights[k][y_f[n], x_f[n]]*PSF_Moffat[k] for k in range(len(weights))]), axis=0)

            f, res, lraw, lstd = Compute_Estim_Grid(x_f[n], y_f[n], z_f[n],
                                                    grid_dxy, profile, Nx, Ny,
                                                    Nz, sigma[:, y_f[n], x_f[n]],
                                                    sigma[s], cube_faint[s],
                                                    cube_faint_pad[spad],
                                                    FSF, longxy, Dico,
                                                    xmin[n], ymin[n])

            flux[n] = f
            residual[n] = res
            line_est_raw[n, :] = lraw
            line_est_std[n, :] = lstd

        # Take the estimated line with the minimum absolute value of the residual
        ind_n = np.argmin(np.abs(residual))
        if not np.isnan(flux[ind_n]) and not np.ma.isMaskedArray(flux[ind_n]):
            Cat2_x.append(x_f[ind_n])
            Cat2_y.append(y_f[ind_n])
            Cat2_z.append(z_f[ind_n])
            Cat2_res_min.append(np.abs(residual[ind_n]))
            Cat2_flux.append(flux[ind_n])
            Cat_est_line_raw.append(line_est_raw[ind_n, :])
            Cat_est_line_std.append(line_est_std[ind_n, :])
        else:
            col_del.append(it)
    sys.stdout.write("\n")

    Cat2 = Cat1_T.copy()
    Cat2.remove_rows(col_del)
    Cat2['x'] = Cat2_x
    Cat2['y'] = Cat2_y
    Cat2['z'] = Cat2_z
    # add real coordinates
    pixcrd = [[p, q] for p, q in zip(Cat2_y, Cat2_x)]
    skycrd = wcs.pix2sky(pixcrd)
    ra = skycrd[:, 1]
    dec = skycrd[:, 0]
    lbda = wave.coord(Cat2_z)
    Cat2['ra'] = ra
    Cat2['dec'] = dec
    Cat2['lbda'] = lbda
    #
    col_res = Column(name='residual', data=Cat2_res_min)
    col_flux = Column(name='flux', data=Cat2_flux)
    col_num = Column(name='num_line', data=np.arange(len(Cat2)))
    Cat2.add_columns([col_res, col_flux, col_num])

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return Cat2, Cat_est_line_raw, Cat_est_line_std


def Compute_Estim_Grid(x0, y0, z0, grid_dxy, profile, Nx, Ny, Nz,
                       sigmat, sigma_t, cube_faint_t, cube_faint_pad,
                       PSF_Moffat, longxy, Dico, xmin, ymin):
    """Function to compute the estimated emission line for each coordinate
    with the deconvolution model :
    subcube = FSF*line -> line_est = subcube*fsf/(fsf^2)

    Parameters
    ----------
    x0 : integer
         Column of the voxel to compute the estimated line
    y0 : integer
         Row of the voxel to compute the estimated line
    z0 : integer
         Spectral channel of the voxel to compute the estimated line
    grid_dxy : integer
               Maximum spatial shift for the grid
    profile  : array
               Number of the profile associated to the T_GLR
    Nx         : int
                 Size of the cube along the x-axis
    Ny         : int
                 Size of the cube along the z-axis
    Nz         : int
                 Size of the cube along the spectral axis
    sigmat     : array
                 MUSE covariance for the pixel (x0,y0)
    sigma_t      : array
                 MUSE covariance
    cube_faint_t : array
                 Projection on the eigenvectors associated to the lower
                 eigenvalues
    cube_faint_pad : array
                     Pad subcube in case of the 3D atom is out of the cube data
    PSF_Moffat : array
                 FSF for this data cube
    longxy     : float
                 mid-size of the PSF
    Dico       : array
                 Dictionary of spectral profiles to test
    xmin       : int
                 Edge of the cube
    ymin       : int
                 Edge of the cube

    Returns
    -------
    res          : float
                   Residual for this line estimation
    flux         : float
                   Flux of the estimated line in the data space
    line_est_raw : array
                   Estimated line in the data space
    line_est     : array
                   Estimated line in the SNR space


    Date  : Dec, 11 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # spectral profile
    num_prof = profile[z0, y0, x0]
    profil0 = Dico[num_prof]
    profil1 = profil0[profil0 > 1e-20]
    long0 = profil1.shape[0]
    longz = long0 // 2

    # size of the 3D atom along the spectral axis
    intz1 = max(0, z0 - longz)
    intz2 = min(Nz, z0 + longz + 1)

    # Initialization
    line_est = np.zeros(Nz)

    # Deconvolution
    line_est[intz1:intz2] = np.sum((PSF_Moffat[intz1:intz2, :, :] *
                                    cube_faint_pad[intz1:intz2, :, :]),
                                   axis=(1, 2)) \
        / np.sum((PSF_Moffat[intz1:intz2, :, :] *
                  PSF_Moffat[intz1:intz2, :, :]), axis=(1, 2))

    # Estimated line in data space
    line_est_raw = line_est * np.sqrt(sigmat)

    # Atome 3D corresponding to the estimated line
    atom_est = np.zeros((long0, PSF_Moffat.shape[1], PSF_Moffat.shape[2]))
    z1 = max(0, z0 - longz)
    z2 = min(Nz, long0 + z0 - longz)
    atom_est[z1 - z0 + longz:z2 - z0 + longz, :, :] = \
        line_est_raw[z1:z2, np.newaxis, np.newaxis] * PSF_Moffat[z1:z2, :, :]

    z1 = np.abs(min(0, z0 - longz))

    # Atom cut at the edges of the cube
    atom_est_cut = atom_est[z1:z1 + intz2 - intz1,
                            ymin:ymin + cube_faint_t.shape[1],
                            xmin:xmin + cube_faint_t.shape[2]]
    # Estimated 3D atom in SNR space
    atom_est_std = atom_est_cut / np.sqrt(sigma_t[intz1:intz2, :, :])
    # Norm of the 3D atom
    norm2_atom1 = np.inner(atom_est_std.flatten(), atom_est_std.flatten())
    # Estimated amplitude of the 3D atom
    alpha_est = np.inner(cube_faint_t[intz1:intz2, :, :].flatten(),
                         atom_est_std.flatten()) / norm2_atom1
    # Estimated detected emitters
    atom_alpha_est = alpha_est * atom_est_std
    # Residual of the estimation
    res = np.ma.sum(np.ma.masked_invalid((cube_faint_t[intz1:intz2, :, :] - atom_alpha_est)**2, copy=False))
    # Flux
    flux = np.ma.sum(np.ma.masked_invalid(line_est_raw, copy=False))

    return flux, res, line_est_raw, line_est


def Spatial_Merging_Circle(Cat0, fwhm_fsf, wcs):
    """Construct a catalogue of sources by spatial merging of the detected
    emission lines in a circle with a diameter equal to the mean over the
    wavelengths of the FWHM of the FSF

    Parameters
    ----------
    Cat0     : astropy.Table
               catalogue
               Columns of Cat0:
               x y z ra dec lbda T_GLR profile pvalC pvalS pvalF T1 T2
               residual flux num_line
    fwhm_fsf : float
               The mean over the wavelengths of the FWHM of the FSF
    wcs      : `mpdaf.obj.WCS`
               RA-DEC coordinates.

    Returns
    -------
    CatF : astropy.Table
           Columns of CatF:
           ID x_circle y_circle ra_circle dec_circle x_centroid y_centroid
           ra_centroid dec_centroid nb_lines x y z ra dec lbda T_GLR profile
           pvalC pvalS pvalF T1 T2 residual flux num_line
    """
    logger = logging.getLogger('origin')
    t0 = time.time()

    colF = []
    colF_id = []
    colF_x = []
    colF_y = []
    colF_xc = []
    colF_yc = []
    colF_nlines = []

    points = np.empty((len(Cat0), 2))
    points[:, 0] = Cat0['x'].data
    points[:, 1] = Cat0['y'].data

    col_tglr = Cat0['T_GLR'].data
    col_id = np.arange(len(Cat0))

    t = KDTree(points)
    r = t.query_ball_tree(t, fwhm_fsf / 2)
    r = [list(x) for x in set(tuple(x) for x in r)]

    centroid = np.array([np.sum(col_tglr[r[i]][:, np.newaxis] * points[r[i]], axis=0) / np.sum(col_tglr[r[i]]) for i in range(len(r))])
    unique_centroid = np.array(list(set(tuple(p) for p in centroid)))

    t_centroid = KDTree(unique_centroid)
    r = t_centroid.query_ball_tree(t, fwhm_fsf / 2)

#    while True:
#        ncentroid = len(unique_centroid)
#        t_centroid = KDTree(unique_centroid)
#        r = t_centroid.query_ball_tree(t, np.round(fwhm_fsf/2))
#        centroid = np.array([np.sum(col_tglr[r[i]][:,np.newaxis] * points[r[i]], axis=0) / np.sum(col_tglr[r[i]]) for i in range(len(r))])
#        uniq = np.array(list(set(tuple(p) for p in centroid)))
#        if len(uniq) >= ncentroid:
#            break
#        else:
#            unique_centroid = uniq

    sorted_lists = sorted(zip(r, unique_centroid), key=lambda t: len(t[0]),
                          reverse=True)
    r = [p[0] for p in sorted_lists]
    unique_centroid = [p[1] for p in sorted_lists]

    used_lines = []

    for i in range(len(r)):
        # Number of lines for this source
        lines = [l for l in r[i] if col_id[l] not in used_lines]
        if len(lines) > 0:
            # Number of this source
            num_source = i + 1

            used_lines += lines
            nb_lines = len(lines)
            # To fulfill each line of the catalogue
            n_S = np.resize(num_source, nb_lines)
            # Coordinates of the center of the circle
            x_c = np.resize(unique_centroid[i][0], nb_lines)
            y_c = np.resize(unique_centroid[i][1], nb_lines)
            # Centroid weighted by the T_GLR of voxels in each group
            centroid = np.sum(col_tglr[lines][:, np.newaxis] * points[lines],
                              axis=0) / np.sum(col_tglr[lines])
            # To fulfill each line of the catalogue
            x_centroid = np.resize(centroid[0], nb_lines)
            y_centroid = np.resize(centroid[1], nb_lines)
            # Number of lines for this source
            nb_lines = np.resize(int(nb_lines), nb_lines)
            # New catalogue of detected emission lines merged in sources
            colF.append(col_id[lines])
            colF_id.append(n_S)
            colF_x.append(x_c)
            colF_y.append(y_c)
            colF_xc.append(x_centroid)
            colF_yc.append(y_centroid)
            colF_nlines.append(nb_lines)

    CatF = Cat0[np.concatenate(colF)].copy()
    col_id = Column(name='ID', data=np.concatenate(colF_id))
    colF_x = np.concatenate(colF_x)
    col_x = Column(name='x_circle', data=colF_x)
    colF_y = np.concatenate(colF_y)
    col_y = Column(name='y_circle', data=colF_y)
    colF_xc = np.concatenate(colF_xc)
    col_xc = Column(name='x_centroid', data=colF_xc)
    colF_yc = np.concatenate(colF_yc)
    col_yc = Column(name='y_centroid', data=colF_yc)
    col_nlines = Column(name='nb_lines', data=np.concatenate(colF_nlines))
    # add real coordinates
    pixcrd = [[p, q] for p, q in zip(colF_y, colF_x)]
    skycrd = wcs.pix2sky(pixcrd)
    col_ra = Column(name='ra_circle', data=skycrd[:, 1])
    col_dec = Column(name='dec_circle', data=skycrd[:, 0])
    pixcrd = [[p, q] for p, q in zip(colF_yc, colF_xc)]
    skycrd = wcs.pix2sky(pixcrd)
    col_rac = Column(name='ra_centroid', data=skycrd[:, 1])
    col_decc = Column(name='dec_centroid', data=skycrd[:, 0])
    CatF.add_columns([col_id, col_x, col_y, col_ra, col_dec, col_xc, col_yc,
                      col_rac, col_decc, col_nlines],
                     indexes=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    nid = len(np.unique(CatF['ID']))
    logger.info('%d sources identified in catalog after spatial merging', nid)
    logger.debug('%s executed in %1.1fs' % (whoami(), time.time() - t0))

    return CatF


def Spectral_Merging(Cat, Cat_est_line_raw, deltaz=1):
    """Merge the detected emission lines distants to less than deltaz
    spectral channel in each group

    Parameters
    ---------
    Cat          : astropy.Table
                   Catalogue of detected emission lines
                   Columns of Cat:
                   ID x_circle y_circle ra_circle dec_circle
                   x_centroid y_centroid ra_centroid, dec_centroid nb_lines
                   x y z ra dec lbda T_GLR profile pvalC pvalS pvalF T1 T2
                   residual flux num_line
    Cat_est_line : list of array
                   Catalogue of estimated lines
    deltaz       : integer
                   Distance maximum between 2 different lines

    Returns
    -------
    CatF : astropy.Table
           Catalogue
           Columns of CatF:
           ID x_circle y_circle ra_circle dec_circle x_centroid y_centroid
           ra_centroid dec_centroid nb_lines x y z ra dec lbda T_GLR profile
           pvalC pvalS pvalF T1 T2 residual flux num_line

    Date  : Dec,16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # Initialization
    CatF = Table()

    # Loop on the group
    for i in ProgressBar(np.unique(Cat['ID'])):
        # Catalogue of the lines in this group
        E = Cat[Cat['ID'] == i]
        # Sort along the maximum of the estimated lines
        z = [np.argmax(Cat_est_line_raw[k]) for k in E['num_line']]
        # Add the spectral channel of the maximum
        # Sort along the spectral channel of the maximum
        col_zp = Column(name='z2', data=z)
        Ez = E.copy()
        Ez.add_column(col_zp)
        Ez.sort('z2')

        ksel = np.where(Ez[1:]['z2'] - Ez[:-1]['z2'] > deltaz)
        indF = []
        for ind in np.split(np.arange(len(Ez)), ksel[0] + 1):
            if len(ind) == 1:
                # if the 2 lines are not close and not in the catlaogue yet
                indF.append(ind[0])
            else:
                # Keep the estimated line with the highest flux
                indF.append(np.where(Ez['flux'] == max(Ez[ind]['flux']))[0][0])

        CatF_temp = Table(Ez[indF])
        nb_line = len(CatF_temp)

        # Set the new number of lines for each group
        CatF_temp['nb_lines'] = np.resize(int(nb_line), len(CatF_temp))
        if len(CatF) == 0:
            CatF = CatF_temp
        else:
            CatF = join(CatF, CatF_temp, join_type='outer')

    CatF.remove_columns(['z2'])
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return CatF


def Construct_Object(k, ktot, uflux, unone, cols, units, desc, fmt, step_wave,
                     origin, filename, maxmap, correl, fwhm_profiles,
                     param, path, name, i, ra, dec, x_centroid,
                     y_centroid, wave_pix, GLR, num_profil, pvalC, pvalS,
                     pvalF, T1, T2, nb_lines, Cat_est_line_data,
                     Cat_est_line_var, y, x, flux, src_vers, author):
    """Function to create the final source

    Parameters
    ----------
    """

    logger = logging.getLogger('origin')
    logger.info('{}/{} source ID {}'.format(k+1,ktot,i))
    cube = Cube(filename)
    cubevers = cube.primary_header.get('CUBE_V', '')
    origin.append(cubevers)

    maxmap = Image(data=maxmap, wcs=cube.wcs)

    src = Source.from_data(i, ra, dec, origin)
    src.add_attr('x', x_centroid, desc='x position in pixel',
                 unit=u.pix, fmt='d')
    src.add_attr('y', y_centroid, desc='y position in pixel',
                 unit=u.pix, fmt='d')

    src.add_white_image(cube)
    src.add_cube(cube, 'MUSE_CUBE')
    src.add_image(maxmap, 'MAXMAP')
    src.add_attr('SRC_V', src_vers, desc='Source version')

    src.add_history('Source created with Origin', author)

    w = cube.wave.coord(wave_pix, unit=u.angstrom)
    names = np.array(['%04d'%w[j] for j in range(nb_lines)])
    if np.unique(names).shape != names.shape:
        names = names.astype(np.int)
        while(not ((names[1:]-names[:-1]) == 0).all()):
            names[1:][(names[:-1]-names[1:]) == 0] += 1
        names = names.astype(np.str)

    correl_ = Cube(data=correl, wcs=cube.wcs, wave=cube.wave, mask=cube._mask, copy=False)

    for j in range(nb_lines):
        sp_est = Spectrum(data=Cat_est_line_data[j, :],
                          wave=cube.wave)
        ksel = np.where(sp_est._data != 0)
        z1 = ksel[0][0]
        z2 = ksel[0][-1] + 1
        # Estimated line
        sp = sp_est[z1:z2]
        # Wavelength in angstrom of estimated line
        #wave_ang = wave.coord(ksel[0], unit=u.angstrom)
        # T_GLR centered around this line
        c = correl[z1:z2, y[j], x[j]]
        # FWHM in arcsec of the profile
        profile_num = num_profil[j]
        profil_FWHM = step_wave * fwhm_profiles[profile_num]
        #profile_dico = Dico[profile_num]
        fl = flux[j]
        vals = [w[j], profil_FWHM, fl, GLR[j], pvalC[j], pvalS[j],
                    pvalF[j], T1[j], T2[j], profile_num]
        src.add_line(cols, vals, units, desc, fmt)
        src.spectra['LINE_{:s}'.format(names[j])] = sp
        sp = Spectrum(wave=cube.wave[z1:z2], data=c)
        src.spectra['CORR_{:s}'.format(names[j])] = sp
        src.add_narrow_band_image_lbdaobs(cube,
                                        'NB_LINE_{:s}'.format(names[j]),
                                        w[j], width=2 * profil_FWHM,
                                        is_sum=True, subtract_off=True)
        src.add_narrow_band_image_lbdaobs(correl_,
                                        'NB_CORR_{:s}'.format(names[j]),
                                        w[j], width=2 * profil_FWHM,
                                        is_sum=True, subtract_off=True)

        if 'ThresholdPval' in param.keys():
            src.OP_THRES = (param['ThresholdPval'], 'Orig Threshold Pval')
        if 'deltaz' in param.keys():
            src.OP_DZ = (param['deltaz'], 'Orig deltaz')
        if 'r0PCA' in param.keys():
            src.OP_R0 = (param['r0PCA'], 'Orig PCA R0')
        if 'threshT1' in param.keys():
            src.OP_T1 = (param['threshT1'], 'Orig T1 threshold')
        if 'threshT2' in param.keys():
            src.OP_T2 = (param['threshT2'], 'Orig T2 threshold')
        if 'neighboors' in param.keys():
            src.OP_NG = (param['neighboors'], 'Orig Neighboors')
        if 'meanestPvalChan' in param.keys():
            try:
                src.OP_MP = (param['meanestPvalChan'], 'Orig Meanest PvalChan')
            except:
                for i in range(len(param['meanestPvalChan'])):
                    src.header['OP_MP%02d'%(i+1)] = param['meanestPvalChan'][i]
        if 'nbsubcube' in param.keys():
            src.OP_NS = (param['nbsubcube'], 'Orig nb of subcubes')
        if 'grid_dxy' in param.keys():
            src.OP_DXY = (param['grid_dxy'], 'Orig Grid Nxy')
        if 'grid_dz' in param.keys():
            src.OP_DZ = (param['grid_dz'], 'Orig Grid Nz')
        if 'PSF' in param.keys():
            src.OP_FSF = (param['PSF'], 'Orig FSF cube')
        src.write('%s/%s-%05d.fits' % (path, name, src.ID))


def Construct_Object_Catalogue(Cat, Cat_est_line, correl, wave, fwhm_profiles,
                               path, name, param, src_vers, author, ncpu=1):
    """Function to create the final catalogue of sources with their parameters

    Parameters
    ----------
    Cat              : Catalogue of parameters of detected emission lines:
                       ID x_circle y_circle x_centroid y_centroid nb_lines
                       x y z T_GLR profile pvalC pvalS pvalF T1 T2 residual
                       flux num_line RA DEC
    Cat_est_line     : list of spectra
                       Catalogue of estimated lines
    correl            : array
                        Cube of T_GLR values
    wave              : `mpdaf.obj.WaveCoord`
                        Spectral coordinates
    fwhm_profiles     : array
                        List of fwhm values (in pixels) of the input spectra profiles (DICO).


    Date  : Dec, 16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    uflux = u.erg / (u.s * u.cm**2)
    unone = u.dimensionless_unscaled
    cols = ['LBDA_ORI', 'FWHM_ORI', 'FLUX_ORI', 'GLR', 'PVALC', 'PVALS',
            'PVALF', 'T1', 'T2', 'PROF']
    units = [u.Angstrom, u.Angstrom, uflux, unone, unone, unone, unone, unone,
             unone, unone]
    desc = None
    fmt = ['.2f', '.2f', '.1f', '.1f', '.1e', '.1e', '.1e', '.1f', '.1f', 'd']

    step_wave = wave.get_step(unit=u.angstrom)
    filename = param['cubename']
    origin = ['ORIGIN', __version__, os.path.basename(filename)]

    maxmap = np.amax(correl, axis=0)

    sources_arglist = []

    for i in np.unique(Cat['ID']):
        # Source = group
        E = Cat[Cat['ID'] == i]
        ra = E['ra_centroid'][0]
        dec = E['dec_centroid'][0]
        x_centroid = E['x_centroid'][0]
        y_centroid = E['y_centroid'][0]
        # Lines of this group
        wave_pix = E['z']
        GLR = E['T_GLR']
        num_profil = E['profile']
        pvalC = E['pvalC']
        pvalS = E['pvalS']
        pvalF = E['pvalF']
        T1 = E['T1']
        T2 = E['T2']
        # Number of lines in this group
        nb_lines = E['nb_lines'][0]
        Cat_est_line_data = np.empty((nb_lines, wave.shape))
        Cat_est_line_var = np.empty((nb_lines, wave.shape))
        for j in range(nb_lines):
            Cat_est_line_data[j,:] = Cat_est_line[E['num_line'][j]]._data
            Cat_est_line_var[j,:] = Cat_est_line[E['num_line'][j]]._var
        y = E['y']
        x = E['x']
        flux = E['flux']

        source_arglist = (i, ra, dec, x_centroid,
                     y_centroid, wave_pix, GLR, num_profil, pvalC, pvalS,
                     pvalF, T1, T2, nb_lines, Cat_est_line_data,
                     Cat_est_line_var, y, x, flux, src_vers, author)
        sources_arglist.append(source_arglist)

    if ncpu > 1:
        # run in parallel
        errmsg = Parallel(n_jobs=ncpu, max_nbytes=1e6)(
            delayed(Construct_Object)(k, len(sources_arglist), uflux, unone, cols, units, desc,
                                      fmt, step_wave, origin, filename,
                                      maxmap, correl, fwhm_profiles,
                                      param, path, name, *source_arglist)
            for k,source_arglist in enumerate(sources_arglist))
        # print error messages if any
        for msg in errmsg:
            if msg is None: continue
            logger.error(msg)
    else:
        for k,source_arglist in enumerate(sources_arglist):
            msg = Construct_Object(k, len(sources_arglist), uflux, unone, cols, units, desc,
                                      fmt, step_wave, origin, filename,
                                      maxmap, correl, fwhm_profiles,
                                      param, path, name, *source_arglist)
            if msg is not None:
                logger.error(msg)

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return len(np.unique(Cat['ID']))


def whoami():
    return sys._getframe(1).f_code.co_name
