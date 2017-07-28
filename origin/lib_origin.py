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
Laure Piqueras (CRAL). From November 2016 the software is updated by Antony
Schutz under the supervision of David Mary

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL).
Please contact Carole for more info at carole.clastres@univ-lyon1.fr
Please contact Antony for more info at antonyschutz@gmail.com

lib_origin.py contains the methods that compose the ORIGIN software
"""
from __future__ import absolute_import, division

import matplotlib.pyplot as plt

import astropy.units as u
import logging
import numpy as np
import os.path
import sys
import time

from astropy.table import Table, Column
from astropy.utils.console import ProgressBar
from joblib import Parallel, delayed
from scipy import signal, stats
from scipy.ndimage import measurements, filters
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial import KDTree
from scipy.sparse.linalg import svds
from six.moves import range, zip

from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.sdetect import Source

__version__ = 'beta3'

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


def DCTMAT(dct_o):
    """Function to compute the DCT Matrix or order dct_o.

    Parameters
    ----------
    dct_o   :   integer
                order of the dct (spectral length)

    Returns
    -------
    dct_m   :   array
                DCT Matrix
    """
    cc = np.arange(dct_o)
    cc = np.repeat(cc[None,:], dct_o, axis=0)
    dct_m = np.sqrt(2 / dct_o) \
        * np.cos(np.pi * ((2 * cc) + 1) * cc.T / (2 * dct_o))
    dct_m[0,:] = dct_m[0,:] / np.sqrt(2)
    return dct_m


def dct_residual(w_raw, order):
    """Function to compute the residual of the DCT on raw data.

    Parameters
    ----------
    RAW     :   array
                the RAW data

    order   :   integer
                The number of atom to keep for the dct decomposition

    Returns
    -------
    Faint     : array
                residual from the dct decomposition

    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """
    nl = w_raw.shape[0]
    D0 = DCTMAT(nl)
    D0 = D0[:, 0:order+1]
    A = np.dot(D0, D0.T)

    cont = np.tensordot(A, w_raw, axes=(0,0))
    Faint = w_raw - cont

    return Faint, cont


def Compute_Standardized_data(cube_dct ,expmap, var, newvar):
    """Function to compute the standardized data.

    Parameters
    ----------
    cube_dct:   array
                output of dct_residual

    expmap  :   array
                exposure map

    var     : array
              variance array

    newvar  : boolean
              if true, variance is re-estimated

    Returns
    -------
    STD     :   array
                standardized data cube from cube dct

    VAR     :   array
                cube of variance

    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """
    nl,ny,nx = cube_dct.shape

    mask = expmap==0

    cube_dct[mask] = np.nan

    mean_lambda = np.nanmean(cube_dct, axis=(1,2))
    mean_lambda = mean_lambda[:, np.newaxis, np.newaxis]* np.ones((nl,ny,nx))
    if newvar:
        var = np.nanvar(cube_dct, axis=(1,2))
        var = var[:, np.newaxis, np.newaxis] * np.ones((nl,ny,nx))
        var[mask] = np.inf
    else:
        print('expmap unknown')
        var[mask] = np.inf

    STD = (cube_dct - mean_lambda) / np.sqrt(var)
    STD[mask] = 0

    return STD, var


def Compute_GreedyPCA_SubCube(NbSubcube, cube_std, intx, inty,
                              Noise_population, threshold_test,itermax):
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
    nuisance_test   :   function used to estimate the nuisance degree of a
                        spectrum (default is O2_test)
    Noise_population:   proportion of estimated noise part used to
                        define the background spectra
    threshold       :   threshold of nuisance_test
    itermax         :   max iteration

    Returns
    -------
    cube_faint : array
                Faint greedy decomposition od STD Cube

    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    cube_faint = np.zeros(cube_std.shape)
    mapO2 = {} #2D
    histO2 = {} #1D
    frecO2 = {} #1D
    thresO2 = np.empty((NbSubcube, NbSubcube)) #scalaire
    # Spatial segmentation
    with ProgressBar(NbSubcube**2) as bar:
        for numy in range(NbSubcube):
            for numx in range(NbSubcube):
                # limits of each spatial zone
                x1 = intx[numx]
                x2 = intx[numx + 1]
                y2 = inty[numy]
                y1 = inty[numy + 1]
                # Data in this spatio-spectral zone
                cube_temp = cube_std[:, y1:y2, x1:x2]
                # greedy PCA on each subcube
                cube_faint[:, y1:y2, x1:x2], mO2, hO2, fO2, tO2 = \
                Compute_GreedyPCA( cube_temp, Noise_population,
                                  threshold_test,itermax)
                mapO2[(numx, numy)] = mO2
                histO2[(numx, numy)] = hO2
                frecO2[(numx, numy)] = fO2
                thresO2[(numx, numy)] = tO2
                bar.update()

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return cube_faint, mapO2, histO2, frecO2, thresO2


def O2test(Cube_in):
    """Function to compute the test on data. The test estimate the background
    part and nuisance part of the data by mean of second order test:
    Testing mean and variance at same time of spectra

    Parameters
    ----------
    Cube_in :   array
                  The 3D cube data to test


    Returns
    -------
    test    :   array
                2D result of the test

    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """
    return np.mean( Cube_in**2 ,axis=0)


def Compute_thresh_PCA_hist(test, threshold_test): 
    """Function to compute greedy svd.
    Parameters
    ----------
    test :   array
             2D data from the O2 test
    threshold_test      :   float
                            the pfa of the test (default=.05)

    Returns
    -------
    histO2  :   histogram value of the test
    frecO2  :   frequencies of the histogram
    thresO2 :   automatic threshold for the O2 test

    Date  : July, 06 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """

    test_v = np.ravel(test)
    c = test_v[test_v>0]
    histO2, frecO2 = np.histogram(c, bins='fd', normed=True)
    ind = np.argmax(histO2)
    mod = frecO2[ind]
    ind2 = np.argmin(( histO2[ind]/2 - histO2[:ind] )**2)
    fwhm = mod - frecO2[ind2]
    sigma = fwhm/np.sqrt(2*np.log(2))

    thresO2 = mod - sigma*stats.norm.ppf(threshold_test)

    return histO2, frecO2, thresO2


def Compute_GreedyPCA(cube_in, Noise_population, threshold_test\
                      ,itermax):
    """Function to compute greedy svd. thanks to the test (test_fun) and
    according to a defined threshold (threshold_test) the cube is segmented
    in nuisance and background part. A part of the background part
    (1/Noise_population %) is used to compute a mean background, a signature.
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
    Cube_in :   array
                The 3D cube data clean

    test_fun:   function
                the test to be performed on data

    Noise_population    :   float
                            the fraction of spectra estimated as background

    threshold_test      :   float
                            the pfa of the test (default=.05)
    itermax             : max iterations
    Returns
    -------
    faint    :  array
                cleaned cube

    mapO2   :   array
                2D MAP filled with the number of iteration per spectra

    histO2  :   histogram value of the test
    frecO2  :   frequencies of the histogram
    thresO2 :   automatic threshold for the O2 test

    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """

    faint = cube_in.copy()
    nl,ny,nx = cube_in.shape
    test = O2test(faint)

    # automatic threshold computation
    histO2, frecO2, thresO2 = Compute_thresh_PCA_hist(test, threshold_test)

    # nuisance part
    py,px = np.where(test>thresO2)

    npix = len(py)

    mapO2 = np.zeros((ny, nx))

    with ProgressBar(npix) as bar:
        # greedy loop based on test
        tmp=0
        while True:
            tmp+=1
            mapO2[py,px] += 1
            if len(py)==0:
                break
            if tmp>itermax:
                print('Warning iterations stopped at %d' %(tmp))
                break
            # vector data
            test_v = np.ravel(test)
            test_v = test_v[test_v>0]
            bckv = np.reshape(faint,(nl,ny*nx))
            nind = np.where(test_v<=thresO2)[0]
            sortind = np.argsort(test_v[nind])
            # at least one spectra is used to perform the test
            l = 1 + int( len(nind) / Noise_population)
            # background estimation
            b = np.mean(bckv[:,nind[sortind[:l]]],axis=1)

            # cube segmentation
            x_red = faint[:,py,px]

            # orthogonal projection with background
            x_red -= np.dot( np.dot(b[:,None],b[None,:]) , x_red )
            x_red /= np.nansum(b**2)

            # remove spectral mean from residual data
            mean_in_pca = np.mean( x_red , axis = 1)
            x_red_nomean = x_red.copy()
            x_red_nomean -= np.repeat(mean_in_pca[:,np.newaxis], \
                                      x_red.shape[1], axis=1)

            # sparse svd if nb spectrum > 1 else normal svd
            if x_red.shape[1]==1:
                break
                # if PCA will not converge or if giant pint source will exists
                # in faint PCA the reason will be here, in later case
                # add condition while calculating the "mean_in_pca"
                # deactivate the substraction of the mean.
                # This will make the vector whish is above threshold
                # equal to the background. For now we prefer to keep it, to
                # stop iteration earlier in order to keep residual sources
                # with the hypothesis that this spectrum is slightly above
                # the threshold (what we observe in data)
                U,s,V = np.linalg.svd( x_red_nomean , full_matrices=False)
            else:
                U,s,V = svds( x_red_nomean , k=1)

            # orthogonal projection
            xest = np.dot( np.dot(U,np.transpose(U)), \
                          np.reshape(faint,(nl,ny*nx)))
            faint -= np.reshape(xest,(nl,ny,nx))

            # test
            test = O2test(faint)

            # nuisance part
            py,px = np.where(test>thresO2)
            bar.update(npix-len(py))

    return faint, mapO2, histO2, frecO2, thresO2


def init_calibrators(nl, ny, nx, nprofil, x, y, z, amp, profil, random, \
                     Cat_cal):
    """Function to add calibrator at position z,y,x with amplitude amp
    following the profiles profilid


    Parameters
    ----------
    nl,ny,nx    :   int
                    samples size of data in spectral and spatiales dimensions
    nprofil     :   int
                    number of spectrale profil

    x           :   int or list
                    the x spatiale position of the line (pixel)
    y           :   int or list
                    the y spatiale position of the line (pixel)
    z           :   int or list
                    the z spectrale position of the line (pixel)

    amp         :   float
                    if x is array or list and if amp is int, amp is repeated
                    amplitude of the line

    profilid    :   int or list
                    if x is array or list and if profilid is int,
                    profilid is repeated
                    the number (id) of the profile associated to the line

    random      :   int
                    number of random line added to the data

    Cat_cal     :  catalogue
                   Catalogue with calibrators parameters
                   string
                   if Cat_cal='add', update of the catalogue

    Returns
    -------
    Cat_cal     :  catalogue
                   Catalogue with calibrators parameters

    Date  : Mar, 30 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """

    logger = logging.getLogger('origin')

    if random==0 and x=='' and Cat_cal=='' :
        msg = 'no parameters given: initialize x, y, z, amp,'\
                     +'profilid or random or give a catalogue'
        logger.error(msg)
    if type(Cat_cal)==Table:
      # if input is a catalogue only
        _x = Cat_cal['x']
        _y = Cat_cal['y']
        _z = Cat_cal['z']
        _amp = Cat_cal['amp']
        _profil = Cat_cal['profil']
    else:
        _x = []
        _y = []
        _z = []
        _amp = []
        _profil = []

    # full random initialisation
    if random:
        _amp += list(np.random.rand(random) * 5)
        _x += list(np.array(np.random.rand(random) * nx, dtype=int))
        _y += list(np.array(np.random.rand(random) * ny, dtype=int))
        _z += list(np.array(np.random.rand(random) * nl, dtype=int))
        _profil += list(np.array( np.random.rand(random) * nprofil, dtype=int))

    if x is not None:
        if type(x)==int: # x is int and Table need list
            _x.append(x)
            _y.append(y)
            _z.append(z)
            _amp.append(amp)
            _profil.append(profil)

        else:
            # if position are given but not all amplitude or all profil ID
            if type(amp)==int:
                amp = np.resize(amp,len(x))
            if type(profil)==int:
                profil = np.resize(profil,len(x))
            _amp += list(amp)
            _x += list(x)
            _y += list(y)
            _z += list(z)
            _profil += list(profil)

    # creation of (new) catalogue
    Cat_ref = Table([_x, _y, _z, _amp, _profil ],
                        names=('x', 'y', 'Z', 'amp', 'profil'))

    return Cat_ref


def add_calibrator(Cat_cal, raw, PSF, profiles, weights, var):
    """Function to add calibrator to raw data


    Parameters
    ----------
    Cat_cal     :   catalogue
                    Catalogue with calibrators parameters

    raw         :   array
                    The 3D RAW data

    PSF         :   array
                    The PSF from orig.PSF

    profiles    :   array
                    The profiles from orig.profiles


    Returns
    -------
    Cube_out    :  array
                   raw data with calibrators

    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()

    nl,ny,nx = raw.shape
    print(nl,ny,nx)
    Cube_out = raw.copy()

    x = Cat_cal['x']
    y = Cat_cal['y']
    z = Cat_cal['Z']
    print(z[0],y[0],x[0])
    amp = Cat_cal['amp']
    profil = Cat_cal['profil']


    if type(x) != int :
        for n in range(len(x)):
            Cube_test = np.zeros((nl,ny,nx))
            Cube_test[z[n],y[n],x[n]] = 1
            pp = profiles[profil[n]]
            Cube_test[:, y[n],x[n]] = \
            signal.fftconvolve(Cube_test[:,y[n],x[n]], pp, mode='same')
            fmin = np.maximum(0, z[n]-2*len(pp))
            fmax = np.minimum(nl,z[n]+2*len(pp))
            for psf in range(len(PSF)):
                if np.sum(weights[psf])>0:
                    for i in range(fmin,fmax):
                        Cube_test[i, :, :] += signal.fftconvolve(weights[psf]*Cube_test[i, :, :],PSF[psf][i, :, :], mode='same')
            Cube_test = Cube_test / Cube_test.max() * amp[n]
            Cube_out += Cube_test
    else:
        Cube_test = np.zeros((nl,ny,nx))
        Cube_test[z,y,x] = 1
        pp = profiles[profil]
        Cube_test[:, y,x] = \
        signal.fftconvolve(Cube_test[:,y,x], pp, mode='same')
        for i in range(z-2*len(pp),z+2*len(pp)):
            Cube_test[i, :, :] = signal.fftconvolve(Cube_test[i, :, :],
                                PSF[i, :, :], mode='same')
        Cube_test = Cube_test / Cube_test.max() * amp
        Cube_out += Cube_test


    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))

    return Cube_out
    
    
def Correlation_GLR_test_zone(cube, sigma, PSF_Moffat, weights, Dico, \
                              intx, inty, NbSubcube):
    """Function to compute the cube of GLR test values per zone
    obtained with the given PSF and dictionary of spectral profile.

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
    intx      : array
                limits in pixels of the columns for each zone
    inty      : array
                limits in pixels of the rows for each zone
    NbSubcube : int
                Number of subcube in the spatial segmentation

    Returns
    -------
    correl  : array
              cube of T_GLR values
    profile : array
              Number of the profile associated to the T_GLR

    Date  : Jul, 4 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """
    logger = logging.getLogger('origin')
    # initialization
    # size psf
    if weights is None:
        sizpsf = PSF_Moffat.shape[1]
    else:
        sizpsf = PSF_Moffat[0].shape[1]
    longxy = int(sizpsf // 2)

    Nl,Ny,Nx = cube.shape

    correl = np.zeros((Nl,Ny,Nx))
    correl_min = np.zeros((Nl,Ny,Nx))
    profile = np.zeros((Nl,Ny,Nx))

    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            logger.info('Area %d,%d / (%d,%d)' %(numy,numx,NbSubcube,NbSubcube))
            # limits of each spatial zone
            x1 = np.maximum(0,intx[numx] - longxy)
            x2 = np.minimum(intx[numx + 1]+longxy,Nx)
            y1 = np.maximum(0,inty[numy+1 ] - longxy)
            y2 = np.minimum(inty[numy ]+longxy,Ny)

            x11 = intx[numx]-x1
            y11 = inty[numy+1]-y1
            x22 = intx[numx+1]-x1
            y22 = inty[numy]-y1

            mini_cube = cube[:,y1:y2,x1:x2]
            mini_sigma = sigma[:,y1:y2,x1:x2]

            c,p,cm = Correlation_GLR_test(mini_cube, mini_sigma, PSF_Moffat, \
                                          weights, Dico)

            correl[:,inty[numy+1]:inty[numy],intx[numx]:intx[numx+1]] = \
            c[:,y11:y22,x11:x22]

            profile[:,inty[numy+1]:inty[numy],intx[numx]:intx[numx+1]] = \
            p[:,y11:y22,x11:x22]

            correl_min[:,inty[numy+1]:inty[numy],intx[numx]:intx[numx+1]] = \
            cm[:,y11:y22,x11:x22]

    return correl, profile, correl_min


def Correlation_GLR_test(cube, sigma, PSF_Moffat, weights, Dico):
    # Antony optimiser
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
              cube of T_GLR values of maximum correlation
    profile : array
              Number of the profile associated to the T_GLR
    correl_min : array
                 cube of T_GLR values of minimum correlation

    Date  : July, 6 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
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
        logger.info('Step 1/3 Spatial convolution of the weighted data with the '
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
        logger.info('Step 2/3 Computing Spatial part of the norm of the 3D atoms')
        for i in ProgressBar(list(range(Nz))):
            norm_fsf[i, :, :] = signal.fftconvolve(inv_var[i, :, :],
                                                   fsf_square[i, :, :][::-1, ::-1],
                                                   mode='same')
        del fsf_square, inv_var
    else: # several FSF
        # Spatial convolution of the weighted data with the zero-mean FSF
        logger.info('Step 1/3 Spatial convolution of the weighted data with the '
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
        logger.info('Step 2/3 Computing Spatial part of the norm of the 3D atoms')
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

#    # First spectral profile
#    k0 = 0
#    d_j = Dico[k0]
#    # zero-mean spectral profile
#    d_j = d_j - np.mean(d_j)
#    # Compute the square of the spectral profile
#    profile_square = d_j**2
#
#    ygrid, xgrid = np.mgrid[0:Ny, 0:Nx]
#    xgrid = xgrid.flatten()
#    ygrid = ygrid.flatten()

    logger.info('Step 3/3 Computing second cube of correlation values')
    profile = np.empty(shape)
    correl = -np.inf * np.ones(shape)
    correl_min = np.inf * np.ones(shape)
    for k in ProgressBar(list(range(len(Dico)))):
        # Second cube of correlation values
        d_j = Dico[k]
        d_j = d_j - np.mean(d_j)
        profile_square = d_j**2

        cube_profile2 = np.zeros((Nz,Ny,Nx))
        norm_profile2 = np.zeros((Nz,Ny,Nx))
        for y in range(Ny):
            for x in range(Nx):
                cube_profile = signal.fftconvolve(cube_fsf[:,y,x], d_j,
                                                         mode = 'same')
                norm_profile = signal.fftconvolve(norm_fsf[:,y,x],
                                                         profile_square,
                                                         mode = 'same')
                norm_profile2[:,y,x] = norm_profile
                cube_profile2[:,y,x] = cube_profile

                norm_profile[norm_profile <= 0] = np.inf
                tmp = cube_profile/np.sqrt(norm_profile)
                PROFILE_MAX = np.where( tmp > correl[:, y, x])[0]
                correl[:, y, x] = np.maximum( correl[:, y, x],tmp)
                correl_min[:, y, x] = np.minimum( correl_min[:, y, x],tmp)
                profile[PROFILE_MAX,y,x] = k

#        np.save('cube_profile'+str(k)+'.npy',cube_profile2)
#        np.save('norm_profile'+str(k)+'.npy',norm_profile2)

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return correl, profile, correl_min


def Compute_local_max_zone(correl, correl_min, mask, intx, inty, \
                                NbSubcube, neighboors):
    """Function to compute the local max of T_GLR values for each zone

    Parameters
    ----------
    correl    : array
                cube of maximum T_GLR values (correlations)
    correl_min: array
                cube of minimum T_GLR values (correlations)
    mask      : array
                boolean cube (true if pixel is masked)
    intx      : array
                limits in pixels of the columns for each zone
    inty      : array
                limits in pixels of the rows for each zone
    NbSubcube : int
                Number of subcube in the spatial segmentation
    threshold : float
                The threshold applied to the p-values cube
    neighboors: int
                Number of connected components

    Returns
    -------
    cube_Local_max : array
                     cube of local maxima from maximum correlation
    cube_Local_min : array
                     cube of local maxima from minus minimum correlation

    Date  : July, 6 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # initialization
    cube_Local_max = np.zeros(correl.shape)
    cube_Local_min = np.zeros(correl.shape)
    cube_Local_max = np.zeros(correl.shape)
    cube_Local_min = np.zeros(correl.shape)
    nl,Ny,Nx = correl.shape
    lag = 1

    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone
#            x1 = intx[numx]
#            x2 = intx[numx + 1]
#            y2 = inty[numy]
#            y1 = inty[numy + 1]

            x1 = np.maximum(0,intx[numx] - lag)
            x2 = np.minimum(intx[numx + 1]+lag,Nx)
            y1 = np.maximum(0,inty[numy+1 ] - lag)
            y2 = np.minimum(inty[numy ]+lag,Ny)

            x11 = intx[numx]-x1
            y11 = inty[numy+1]-y1
            x22 = intx[numx+1]-x1
            y22 = inty[numy]-y1

            correl_temp_edge = correl[:, y1:y2, x1:x2]
            correl_temp_edge_min = correl_min[:, y1:y2, x1:x2]
            mask_temp_edge = mask[:, y1:y2, x1:x2]
            # Cube of pvalues for each zone
            cube_Local_max_temp,cube_Local_min_temp= \
              Compute_localmax(correl_temp_edge,correl_temp_edge_min\
                                    ,mask_temp_edge,neighboors)


#            cube_Local_max[:, y1:y2, x1:x2] = cube_Local_max_temp
#            cube_Local_min[:, y1:y2, x1:x2] = cube_Local_min_temp
            cube_Local_max[:,inty[numy+1]:inty[numy],intx[numx]:intx[numx+1]]=\
            cube_Local_max_temp[:,y11:y22,x11:x22]
            cube_Local_min[:,inty[numy+1]:inty[numy],intx[numx]:intx[numx+1]]=\
            cube_Local_min_temp[:,y11:y22,x11:x22]


    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return cube_Local_max, cube_Local_min


def Compute_threshold_segmentation(purity, cube_local_max, cube_local_min, \
                           threshold_option, intx, inty, NbSubcube,
                           cube_cont, pfa):
    """Function to threshold the p-values from
    computatino of threshold from the local maxima of:
        - Maximum correlation
        - Minus minimum correlation

    Parameters
    ----------
    purity    : float
                the purity between 0 and 1
    cube_Local_max : array
                     cube of local maxima from maximum correlation
    cube_Local_min : array
                     cube of local maxima from minus minimum correlation
    threshold_option : float
                       float it is a manual threshold.
                       string
                       threshold based on background threshold

    cube_cont    : array
                   cube of standardized continuum estimated from DCT
    intx      : integer
                limits in pixels of the columns for each zone
    inty      : integer
                limits in pixels of the rows for each zone
    NbSubcube : integer
                Number of subcubes for the spatial segmentation
    pfa          : Pvalue for the test which performs segmentation

    Returns
    -------
    Confidence : float
                 the threshold associated to the purity
    PVal_M : array
             the P Value associated to Maximum Correlation local maxima
    PVal_m : array
             the P Value associated to Minus Minimum Correlation local maxima
    PVal_r : array
             The purity function
    index_pval: array
                index value to plot
    cube_pval_correl : array
                       cube of thresholded p-values associated
                       to the local max of T_GLR values

    Date  : July, 6 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """

    # Label
    map_in = Segmentation(cube_cont, pfa)

    # initialization
    cube_pval_correl = np.zeros(cube_local_max.shape)
    mapThresh = np.zeros((cube_local_max.shape[1],cube_local_max.shape[2]))
    threshold = []
    Pval_r = []
    index_pval = []
    det_m = []
    det_M = []


    # index of segmentation
    ind_y = []
    ind_x = []
    bck_y, bck_x = np.where(map_in==0)
    ind_y.append(bck_y)
    ind_x.append(bck_x)
    src_y, src_x = np.where(map_in>0)
    ind_y.append(src_y)
    ind_x.append(src_x)



    # threshold
    for ind_n in range(2):

        cube_local_max_edge = cube_local_max[:, ind_y[ind_n], ind_x[ind_n]]
        cube_local_min_edge = cube_local_min[:, ind_y[ind_n], ind_x[ind_n]]

        _threshold, _Pval_r, _index_pval, _det_m, _det_M \
        = Compute_threshold( purity, cube_local_max_edge, \
                                           cube_local_min_edge)
        threshold.append(_threshold)
        Pval_r.append((np.asarray(_Pval_r)))
        index_pval.append((np.asarray(_index_pval)))
        det_m.append((np.asarray(_det_m)))
        det_M.append((np.asarray(_det_M)))

        if threshold_option is not None:

            if threshold_option=='background':
                print('background')
                threshold[ind_n] = threshold[(0)]
            else:
                print('given')
                threshold[ind_n] = threshold_option

        cube_pval_correl_l = Threshold_pval(cube_local_max_edge.data, \
                                            threshold[ind_n])

        cube_pval_correl[:, ind_y[ind_n], ind_x[ind_n]]= cube_pval_correl_l
        mapThresh[ind_y[ind_n], ind_x[ind_n]] = threshold[ind_n]

    return np.asarray(threshold), Pval_r, index_pval,  \
            cube_pval_correl, mapThresh, map_in, det_m, det_M


def Compute_threshold(purity, cube_local_max, cube_local_min):
    """Function to compute the threshold from the local maxima of:
        - Maximum correlation
        - Minus minimum correlation

    Parameters
    ----------
    purity    : float
                the purity between 0 and 1
    cube_Local_max : array
                     cube of local maxima from maximum correlation
    cube_Local_min : array
                     cube of local maxima from minus minimum correlation

    Returns
    -------
    Confidence : float
                     the threshold associated to the fidelity
    PVal_M : array
             the P Value associated to Maximum Correlation local maxima
    PVal_m : array
             the P Value associated to Minus Minimum Correlation local maxima
    Pval_r:  list
             ratio of Pvalue thresholded by purity 
    index :  array
             index corresponding to the Pvalue 
    Det_M, Det_m : arrays
                   Number of voxels > threshold == detected sources
    Date  : July, 6 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    N = 100

    Lc_max = cube_local_max
    Lc_min = cube_local_min

    ind = (Lc_max)==0
    Lc_M = Lc_max[~ind]

    ind = (Lc_min)==0
    Lc_m = Lc_min[~ind]

    mini = np.minimum( Lc_m.min() , Lc_M.min() )
    maxi = np.maximum( Lc_m.max() , Lc_M.max() )

    dx = (maxi-mini)/N
    index = np.arange(mini,maxi,dx)

    Det_M = [np.sum( (Lc_M>seuil) ) for seuil in index ]
    Det_m = [np.sum( (Lc_m>seuil) ) for seuil in index ]

    PVal_M = [np.mean( (Lc_M>seuil) ) for seuil in index ]
    PVal_m = [np.mean( (Lc_m>seuil) ) for seuil in index ]

    Pval_r = 1 - np.array(PVal_m)/np.array(PVal_M)

    PVal_r = [(1-np.mean(Lc_m>=seuil)/np.mean(Lc_M>=seuil))>=purity for seuil in index]

    fid_ind = PVal_r.index(True)

    x2 = index[fid_ind]
    x1 = index[fid_ind-1]
    y2 = Pval_r[fid_ind]
    y1 = Pval_r[fid_ind-1]

    b = y2-y1
    a = x2-x1

    tan_theta = b/a
    threshold = (purity-y1)/tan_theta + x1

    return threshold, Pval_r, index, Det_M, Det_m


def Threshold_pval(cube_local_max, threshold):
    """Function to threshold the p-values

    Parameters
    ----------
    cube_Local_max : array
                     cube of local maxima from maximum correlation
    threshold : float
                The threshold applied to the p-values cube

    Returns
    -------

    cube_pval_correl : array
                       cube of thresholded p-values associated
                       to the local max of T_GLR values

    Date  : July, 6 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()
    # Threshold the pvalues
    cube_pval_lm_correl = cube_local_max.copy()
    cube_pval_lm_correl[cube_local_max <= threshold] = 0.
    
#    cube_pval_lm_correl = cube_local_max>threshold + 0.
    
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return cube_pval_lm_correl


def Compute_localmax(correl_temp_edge, correl_temp_edge_min, \
                          mask_temp_edge, neighboors):
    """Function to compute the local maxima of the maximum correlation and
    local maxima of minus the minimum correlation
    distribution

    Parameters
    ----------
    correl_temp_edge :  array
                        T_GLR values with edges excluded (from max correlation)
    correl_temp_edge_min :  array
                        T_GLR values with edges excluded (from min correlation)
    mask_temp_edge   :  array
                        mask array (true if pixel is masked)
    neighboors       :  int
                        Number of connected components
    Returns
    -------
    cube_pval_correl : array
                       p-values asssociated to the local maxima of T_GLR values

    Date  : June, 19 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    # connected components
    conn = (neighboors + 1)**(1 / 3.)
    # local maxima of maximum correlation
    Max_filter = filters.maximum_filter(correl_temp_edge,size=(conn,conn,conn))
    Local_max_mask= (correl_temp_edge == Max_filter)
    Local_max_mask[mask_temp_edge]=0
    Local_max=correl_temp_edge*Local_max_mask

    # local maxima of minus minimum correlation
    minus_correl_min = - correl_temp_edge_min
    Max_filter = filters.maximum_filter(minus_correl_min ,\
                                        size=(conn,conn,conn))
    Local_min_mask= (minus_correl_min == Max_filter)
    Local_min_mask[mask_temp_edge]=0
    Local_min=minus_correl_min*Local_min_mask

    return Local_max, Local_min


def Create_local_max_cat(correl, profile, cube_pval_lm_correl, wcs, wave):
    """Function to compute refrerent voxel of each group of connected voxels
    using the voxel with the higher T_GLR value.

    Parameters
    ----------
    correl            : array
                        cube of T_GLR values
    profile           : array
                        Number of the profile associated to the T_GLR
    cube_pval_lm_correl  : array
                        cube of thresholded p-values associated
                        to the local max of the T_GLR values
    wcs               : `mpdaf.obj.WCS`
                         RA-DEC coordinates.
    wave              : `mpdaf.obj.WaveCoord`
                         Spectral coordinates.

    Returns
    -------
    Cat_ref : astropy.Table
              Catalogue of the referent voxels coordinates for each group
              Columns of Cat_ref : x y z ra dec lba,
                                   T_GLR profile pvalC

    Date  : June, 19 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    logger = logging.getLogger('origin')
    t0 = time.time()


    zpixRef,ypixRef,xpixRef = np.where(cube_pval_lm_correl>0)
    correl_max = correl[zpixRef,ypixRef,xpixRef]
    profile_max = profile[zpixRef, ypixRef, xpixRef]
    pvalC = cube_pval_lm_correl[zpixRef, ypixRef, xpixRef]

    # add real coordinates
    pixcrd = [[p, q] for p, q in zip(ypixRef, xpixRef)]
    skycrd = wcs.pix2sky(pixcrd)
    ra = skycrd[:, 1]
    dec = skycrd[:, 0]
    lbda = wave.coord(zpixRef)
    # Catalogue of referent pixels
    Cat_ref = Table([xpixRef, ypixRef, zpixRef, ra, dec, lbda, correl_max,
                     profile_max, pvalC],
                    names=('x', 'y', 'z', 'ra', 'dec', 'lbda', 'T_GLR',
                           'profile', 'pvalC'))
    # Catalogue sorted along the Z axis
    Cat_ref.sort('z')
    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return Cat_ref


def extract_grid(raw_in, var_in, psf_in, weights_in, y, x, size_grid):

    """Function to extract data from an estimated source in catalog.

    Parameters
    ----------
    raw_in     : array
                 RAW data
    var_in     : array
                 MUSE covariance
    psf_in     : array
                 MUSE PSF
    weights_in : array
                 PSF weights
    y          : integer
                 y position in pixek estimated in previous catalog
    x          : integer
                 x position in pixek estimated in previous catalog
    size_grid  : integer
                 Maximum spatial shift for the grid

    Returns
    -------

    red_dat : cube of raw_in centered in y,x of size PSF+Max spatial shift
    red_var : cube of var_in centered in y,x of size PSF+Max spatial shift
    red_wgt : cube of weights_in centered in y,x of size PSF+Max spatial shift
    red_psf : cube of psf_in centered in y,x of size PSF+Max spatial shift

    Date  : June, 21 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """

    # size data
    nl,ny,nx = raw_in.shape

    # size psf
    if weights_in is None:
        sizpsf = psf_in.shape[1]
    else:
        sizpsf = psf_in[0].shape[1]

    # size minicube
    sizemc = 2*size_grid + sizpsf

    # half size psf
    longxy = int(sizemc // 2)

    # bound of image
    psx1 = np.maximum(0 ,x-longxy)
    psy1 = np.maximum(0 ,y-longxy)
    psx2 = np.minimum(nx,x+longxy+1)
    psy2 = np.minimum(ny,y+longxy+1)

    # take into account bordure of cube
    psx12 = np.maximum(0,longxy-x+psx1)
    psy12 = np.maximum(0,longxy-y+psy1)
    psx22 = np.minimum(sizemc,longxy-x+psx2)
    psy22 = np.minimum(sizemc,longxy-y+psy2)

    # create weight, data with bordure
    red_dat = np.zeros((nl,sizemc,sizemc))
    red_dat[:,psy12:psy22,psx12:psx22] = raw_in[:,psy1:psy2,psx1:psx2]

    red_var = np.ones((nl,sizemc,sizemc)) * np.inf
    red_var[:,psy12:psy22,psx12:psx22] = var_in[:,psy1:psy2,psx1:psx2]

    if weights_in is None:
        red_wgt = None
        red_psf = psf_in
    else:
        red_wgt = []
        red_psf = []
        for n,w in enumerate(weights_in):
            if np.sum(w[psy1:psy2,psx1:psx2])>0:
                w_tmp = np.zeros((sizemc,sizemc))
                w_tmp[psy12:psy22,psx12:psx22] = w[psy1:psy2,psx1:psx2]
                red_wgt.append(w_tmp)
                red_psf.append(psf_in[n])

    return red_dat, red_var, red_wgt, red_psf


def LS_deconv_wgt(data_in,var_in,psf_in):
    """Function to compute the Least Square estimation of a ponctual source.

    Parameters
    ----------
    data_in    : array
                 input data
    var_in     : array
                 input variance
    psf_in     : array
                 weighted MUSE PSF

    Returns
    -------
    deconv_out : LS Deconvolved spectrum

    varest_out : estimated theoretic variance

    Date  : June, 21 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """
    # deconvolution
    nl,sizpsf,tmp = psf_in.shape
    v = np.reshape(var_in,(nl,sizpsf*sizpsf))
    p = np.reshape(psf_in,(nl,sizpsf*sizpsf))
    s = np.reshape(data_in,(nl,sizpsf*sizpsf))
    varest_out = 1 / np.sum( p*p/v,axis=1)
    deconv_out = np.sum(p*s/np.sqrt(v),axis=1) * varest_out

    return deconv_out, varest_out


def conv_wgt(deconv_met, psf_in):
    """Function to compute the convolution of a spectrum. output is a cube of
    the good size for rest of algorithm

    Parameters
    ----------
    deconv_met : LS Deconvolved spectrum
                 input data

    psf_in     : array
                 weighted MUSE PSF

    Returns
    -------
    cube_conv  : Cube, convolution from deconv_met
    Date  : June, 21 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """
    cube_conv = psf_in * deconv_met[:, np.newaxis, np.newaxis]
    cube_conv = cube_conv*(np.abs(psf_in)>0)
    return cube_conv


def method_PCA_wgt(data_in, var_in, psf_in, order_dct):
    """Function to Perform PCA LS or Denoised PCA LS.
    algorithm:
        - principal eigen vector is computed, RAW data are orthogonalized
          this is the first estimation to modelize the continuum
        - on residual, the line is estimated by least square estimation
        - the estimated line is convolved by the psf and removed from RAW data
        - principal eigen vector is computed.

        - - PCA LS: RAW data are orthogonalized, this is the second estimation
                    to modelize the continuum

        - - Denoised PCA LS: The eigen vector is denoised by a DCT, with the
                             new eigen vector RAW data are orthogonalized,
                             this is the second estimation to modelize the
                             continuum
        - on residual, the line is estimated by least square estimation

    Parameters
    ----------
    data_in    : array
                 RAW data
    var_in     : array
                 MUSE covariance
    psf_in     : array
                 MUSE PSF
    order_dct  : integer
                 order of the DCT for the Denoised PCA LS
                 if None the PCA LS is performed

    Returns
    -------

    estimated_line : estimated line
    estimated_var  : estimated variance

    Date  : June, 21 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """

    nl,sizpsf,tmp = psf_in.shape

    # STD
    data_std = data_in / np.sqrt(var_in)
    data_st_pca = np.reshape(data_std,(nl,sizpsf*sizpsf))

    # PCA
    mean_in_pca = np.mean(data_st_pca, axis = 1)
    data_in_pca = data_st_pca - np.repeat(mean_in_pca[:,np.newaxis],
                                          sizpsf*sizpsf,axis=1)

    U,s,V = svds(data_in_pca , k=1)

    # orthogonal projection
    xest = np.dot( np.dot(U,np.transpose(U)), data_in_pca )
    residual = data_std - np.reshape(xest,(nl,sizpsf,sizpsf))

    # LS deconv
    deconv_out, varest_out = LS_deconv_wgt(residual, var_in, psf_in)

    # PSF convolution
    conv_out = conv_wgt(deconv_out, psf_in)

    # cleaning the data
    data_clean = (data_in - conv_out) / np.sqrt(var_in)

    # 2nd PCA
    data_in_pca = np.reshape(data_clean,(nl,sizpsf*sizpsf))
    mean_in_pca = np.mean( data_in_pca , axis = 1)
    data_in_pca-= np.repeat(mean_in_pca[:,np.newaxis],sizpsf*sizpsf,axis=1)

    U,s,V = svds( data_in_pca , k=1)

    if order_dct is not None:
        # denoise eigen vector with DCT
        D0 = DCTMAT(nl)
        D0 = D0[:, 0:order_dct+1]
        A = np.dot(D0, D0.T)
        U = np.dot(A,U)

    # orthogonal projection
    xest = np.dot( np.dot(U,np.transpose(U)), data_st_pca )
    cont = np.reshape(xest,(nl,sizpsf,sizpsf))
    residual = data_std - cont

    # LS deconvolution of the line
    estimated_line, estimated_var = LS_deconv_wgt(residual, var_in, psf_in)

    # PSF convolution of estimated line
    conv_out = conv_wgt(estimated_line, psf_in) 
    # cleaning line in data to estimate convolved continuum
    continuum = (data_in - conv_out) / np.sqrt(var_in)
    # LS deconvolution of the continuum
    estimated_cont, tmp = LS_deconv_wgt(continuum, var_in, psf_in)
    
    return estimated_line, estimated_var, estimated_cont


def GridAnalysis(data_in, var_in, psf, weight_in, horiz, \
               size_grid, y0, x0, z0, NY, NX, horiz_psf,
               criteria, order_dct):
    """Function to compute the estimated emission line and the optimal
    coordinates for each detected lines in a spatio-spectral grid.

    Parameters
    ----------
    data_in    : array
                 RAW data minicube
    var_in     : array
                 MUSE covariance minicube
    psf        : array
                 MUSE PSF minicube
    weight_in  : array
                 PSF weights minicube
    horiz      : integer
                 Maximum spectral shift to compute the criteria for gridding
    size_grid  : integer
                 Maximum spatial shift for the grid
    y0         : integer
                 y position in pixel from catalog
    x0         : integer
                 x position in pixel from catalog
    z0         : integer
                 z position in pixel from catalog
    NY         : integer
                 Number of y-pixels from Full data Cube
    NX         : integer
                 Number of x-pixels from Full data Cube
    y0         : integer
                 y position in pixel from catalog
    horiz_psf  : integer
                 Maximum spatial shift in size of PSF to compute the MSE
    criteria   : string
                 criteria used to choose the candidate in the grid: flux or mse
    order_dct  : integer
                 order of the DCT Used in the Denoised PCA LS, set to None the
                 method become PCA LS only

    Returns
    -------
    flux_est_5          :   float
                            Estimated flux +/- 5
    flux_est_10         :   float
                            Estimated flux +/- 10
    MSE_5               :   float
                            Mean square error +/- 5
    MSE_10              :   float
                            Mean square error +/- 10
    estimated_line      :   array
                            Estimated lines in data space
    estimated_variance  :   array
                            Estimated variance in data space
    y                   :   integer
                            re-estimated x position in pixel of the source
                            in the grid
    x                   :   integer
                            re-estimated x position in pixel of the source
                            in the grid
    z                   :   integer
                            re-estimated x position in pixel of the source
                            in the grid

    Date  : June, 21 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """

    zest = np.zeros((1+2*size_grid,1+2*size_grid))
    fest_00 = np.zeros((1+2*size_grid,1+2*size_grid))
    fest_05 = np.zeros((1+2*size_grid,1+2*size_grid))
    #fest_10 = np.zeros((1+2*size_grid,1+2*size_grid))
    mse = np.ones((1+2*size_grid,1+2*size_grid)) * np.inf
    mse_5 = np.ones((1+2*size_grid,1+2*size_grid)) * np.inf
    #mse_10 = np.ones((1+2*size_grid,1+2*size_grid)) * np.inf

    nl = data_in.shape[0]
    ind_max = slice(np.maximum(0,z0-5),np.minimum(nl,z0+5))
    if weight_in is None:
        nl,sizpsf,tmp = psf.shape
    else:
        nl,sizpsf,tmp = psf[0].shape

    lin_est = np.zeros((nl,1+2*size_grid,1+2*size_grid))
    var_est = np.zeros((nl,1+2*size_grid,1+2*size_grid))
    cnt_est = np.zeros((nl,1+2*size_grid,1+2*size_grid))
    # half size psf
    longxy = int(sizpsf // 2)
    inds = slice(longxy-horiz_psf,longxy+1+horiz_psf)
    for dx in range(0,1+2*size_grid):
        if (x0-size_grid+dx>=0) and (x0-size_grid+dx<NX):
            for dy in range(0,1+2*size_grid):
                if (y0-size_grid+dy>=0) and (y0-size_grid+dy<NY):

                    # extract data
                    r1 = data_in[:,dy:sizpsf+dy,dx:sizpsf+dx]
                    var = var_in[:,dy:sizpsf+dy,dx:sizpsf+dx]
                    if weight_in is not None:
                        wgt = np.array(weight_in)[:,dy:sizpsf+dy,dx:sizpsf+dx]
                        psf = np.sum(np.repeat(wgt[:,np.newaxis,:,:], nl, \
                                               axis=1)*psf, axis=0)

                    # estimate Full Line and theoretic variance
                    deconv_met,varest_met,cont = method_PCA_wgt(r1, var, psf, \
                                                           order_dct)

                    print(np.maximum(0,z0-5),np.minimum(nl,z0+5))
                    z_est = peakdet(deconv_met[ind_max],3)
#                    plt.clf()
#                    plt.plot(deconv_met[ind_max])
#                    xaze = deconv_met[ind_max]
#                    plt.plot(z_est,xaze[z_est],'or')
                    plt.pause(.1)
#                    print(z_est)
                    if z_est ==0:
                        break
                    
                    maxz = z0  - 5 + z_est
                    zest[dy,dx] = maxz
                    ind_z5 = np.arange(maxz-5,maxz+5)
                    #ind_z10 = np.arange(maxz-10,maxz+10)
                    ind_hrz = slice(maxz-horiz,maxz+horiz)

                    lin_est[:,dy,dx] = deconv_met
                    var_est[:,dy,dx] = varest_met
                    cnt_est[:,dy,dx] = cont

                    # compute MSE
                    LC = conv_wgt(deconv_met[ind_hrz], psf[ind_hrz,:,:])
                    LCred = LC[:,inds,inds]
                    r1red = r1[ind_hrz,inds,inds]
                    mse[dy,dx] = np.sum((r1red - LCred)**2)/ np.sum(r1red**2)

                    LC = conv_wgt(deconv_met[ind_z5], psf[ind_z5,:,:])
                    LCred = LC[:,inds,inds]
                    r1red = r1[ind_z5,inds,inds]
                    mse_5[dy,dx] = np.sum((r1red - LCred)**2)/ np.sum(r1red**2)

                    #LC = conv_wgt(deconv_met[ind_z10], psf[ind_z10,:,:])
                    #LCred = LC[:,inds,inds]
                    #r1red = r1[ind_z10,inds,inds]
                    #mse_10[dy,dx] = np.sum((r1red - LCred)**2)/np.sum(r1red**2)

                    # compute flux
                    fest_00[dy,dx] = np.sum(deconv_met[ind_hrz])
                    fest_05[dy,dx] = np.sum(deconv_met[ind_z5])
                    #fest_10[dy,dx] = np.sum(deconv_met[ind_z10])

    if criteria == 'flux':
        wy,wx = np.where(fest_00==fest_00.max())
    elif criteria == 'mse':
        wy,wx = np.where(mse==mse.min())
    else:
        raise IOError('Bad criteria: (flux) or (mse)')
    y = y0 - size_grid + wy
    x = x0 - size_grid + wx
    z = zest[wy,wx]

    flux_est_5 = float( fest_05[wy,wx] )
    #flux_est_10 = float( fest_10[wy,wx] )
    MSE_5 = float( mse_5[wy,wx] )
    #MSE_10 = float( mse_10[wy,wx] )
    estimated_line = lin_est[:,wy,wx]
    estimated_variance = var_est[:,wy,wx]
    estimated_continuum = cnt_est[:,wy,wx]

    return flux_est_5, MSE_5, estimated_line, \
            estimated_variance, int(y), int(x), int(z), estimated_continuum

def peakdet(v, delta):
    
    v = np.array(v)
    nv = len(v)
    mv = np.zeros(nv+2*delta)
    mv[:delta] = np.Inf
    mv[delta:-delta]=v
    mv[-delta:] = np.Inf    
    ind = []

    # find all local maxima
    ind = [n-delta for n in range(delta,nv+delta) if mv[n]>mv[n-1] and mv[n]>mv[n+1]]
    
    # take the maximum and closest from original estimation 
    indi = np.array(ind,dtype=int)

    sol = int(nv/2)
    if len(indi)>0:
    # methode : closest from initial estimate        
        out = indi[np.argmin( (indi-sol)**2)]
    # methode : maximum           
#        out = np.argmax( v[indi] )       
    else:
        out = sol

    plt.clf() 
    plt.plot(v)        
    plt.plot(indi,v[indi],'or')
    plt.pause(.1)    
    return out


def Estimation_Line(Cat1_T, RAW, VAR, PSF, WGT, wcs, wave, size_grid = 1, \
                    criteria = 'flux', order_dct = 30, horiz_psf = 1, \
                    horiz = 5):
    """Function to compute the estimated emission line and the optimal
    coordinates for each detected lines in a spatio-spectral grid.

    Parameters
    ----------
    Cat1_T     : astropy.Table
                 Catalogue of parameters of detected emission lines selected
                 with a narrow band test.
                 Columns of the Catalogue Cat1_T:
                 x y z T_GLR profile pvalC
    DATA       : array
                 RAW data
    VAR        : array
                 MUSE covariance
    PSF        : array
                 MUSE PSF
    WGT        : array
                 PSF weights
    size_grid  : integer
                 Maximum spatial shift for the grid
    criteria   : string
                 criteria used to choose the candidate in the grid: flux or mse
    order_dct  : integer
                 order of the DCT Used in the Denoised PCA LS, set to None the
                 method become PCA LS only
    horiz_psf  : integer
                 Maximum spatial shift in size of PSF to compute the MSE
    horiz      : integer
                 Maximum spectral shift to compute the criteria
    wcs        : `mpdaf.obj.WCS`
                  RA-DEC coordinates.
    wave       : `mpdaf.obj.WaveCoord`
                 Spectral coordinates.

    Returns
    -------
    Cat2             : astropy.Table
                       Catalogue of parameters of detected emission lines.
                       Columns of the Catalogue Cat2:
                       x y z ra dec lbda, T_GLR profile pvalC residual flux num_line
    Cat_est_line_raw : list of arrays
                       Estimated lines in data space
    Cat_est_line_std : list of arrays
                       Estimated lines in SNR space

    Date  : June, 21 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """

    logger = logging.getLogger('origin')
    t0 = time.time()
    # Initialization

    NL,NY,NX = RAW.shape
    Cat2_x_grid = []
    Cat2_y_grid = []
    Cat2_z_grid = []
    Cat2_res_min5 = []
    Cat2_flux5 = []
    Cat_est_line_raw = []
    Cat_est_line_var = []
    Cat_est_cont_raw = []
    for src in Cat1_T:
        y0 = src['y']
        x0 = src['x']
        z0 = src['z']

        red_dat, red_var, red_wgt, red_psf = extract_grid(RAW, VAR, PSF, WGT,\
                                                          y0, x0, size_grid)

        f5, m5, lin_est, var_est, y, x, z, cnt_est = GridAnalysis(red_dat,  \
                      red_var, red_psf, red_wgt, horiz,  \
                      size_grid, y0, x0, z0, NY, NX, horiz_psf, criteria,\
                      order_dct)

        Cat2_x_grid.append(x)
        Cat2_y_grid.append(y)
        Cat2_z_grid.append(z)
        Cat2_res_min5.append(m5)
        Cat2_flux5.append(f5)
        Cat_est_line_raw.append(lin_est.ravel())
        Cat_est_line_var.append(var_est.ravel())
        Cat_est_cont_raw.append(cnt_est.ravel())

    Cat2 = Cat1_T.copy()

    Cat2['x'] = Cat2_x_grid
    Cat2['y'] = Cat2_y_grid
    Cat2['z'] = Cat2_z_grid
    # add real coordinates
    pixcrd = [[p, q] for p, q in zip(Cat2_y_grid, Cat2_x_grid)]
    skycrd = wcs.pix2sky(pixcrd)
    ra = skycrd[:, 1]
    dec = skycrd[:, 0]
    lbda = wave.coord(Cat2_z_grid)
    Cat2['ra'] = ra
    Cat2['dec'] = dec
    Cat2['lbda'] = lbda
    #
    col_flux = Column(name='flux', data=Cat2_flux5)
    col_res = Column(name='residual', data=Cat2_res_min5)
    col_num = Column(name='num_line', data=np.arange(len(Cat2)))

    Cat2.add_columns([col_res, col_flux, col_num])

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))

    return Cat2, Cat_est_line_raw, Cat_est_line_var, Cat_est_cont_raw

def Purity_Estimation(Cat_in, correl, purity_curves, purity_index, 
                        bck_or_src): 
    
    """Function to compute the estimated purity for each line.

    Parameters
    ----------
    Cat_in     : astropy.Table
                 Catalogue of parameters of detected emission lines selected
                 with a narrow band test.
                 Columns of the Catalogue Cat1_T:
                 x y z T_GLR profile pvalC
    correl     : array
                 Origin Correlation data
    fidelity_curves     : array
                          purity curves related to area
    fidelity_index      : array
                          index of purity curves related to area             
    bck_or_src          : array
                          Map to know which area the source is in

    Returns
    -------
    Cat1_2            : astropy.Table
                       Catalogue of parameters of detected emission lines.
                       Columns of the Catalogue Cat2:
                       x y z ra dec lbda, 
                       T_GLR profile pvalC residual flux num_line
                       fidelity


    Date  : July, 25 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """
    
    Cat1_2 = Cat_in.copy()
    purity = []
    
    for n,src in enumerate(Cat1_2):
        y = src['y']
        x = src['x'] 
        z = src['z']         
        area = bck_or_src[y,x]
        seuil = purity_index[area]
        fidel = purity_curves[area]
        value = correl[z,y,x]
        if value>seuil[(fidel==1).tolist().index(True)]:
            fid_tmp = 1
        else:
            fid_ind = (seuil-value>0).tolist().index(True)
            # interpolation of correl value on fidelity curve
            fidel[fid_ind-1]
            x2 = seuil[fid_ind]
            x1 = seuil[fid_ind-1]
            y2 = fidel[fid_ind] 
            y1 = fidel[fid_ind-1] 
            fid_tmp = y1 + (value-x1)*(y2-y1)/(x2-x1)
        purity.append(fid_tmp)
        
    col_fid = Column(name='purity', data=purity)
    Cat1_2.add_columns([col_fid])
    return Cat1_2


def Spatial_Merging_Circle(Cat0, fwhm_fsf, wcs):
    """Construct a catalogue of sources by spatial merging of the detected
    emission lines in a circle with a diameter equal to the mean over the
    wavelengths of the FWHM of the FSF

    Parameters
    ----------
    Cat0     : astropy.Table
               catalogue
               Columns of Cat0:
               x y z ra dec lbda T_GLR profile pvalC
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
           pvalC residual flux num_line
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


def Segmentation(STD_in, pfa):
    """Generate from a 3D Cube a 2D map where sources and background are
    separated

    Parameters
    ---------
    STD_in       : Array
                   standard continu part of DCT from preprocessing step - Cube

    pfa          : Pvalue for the test which performs segmentation

    Returns
    -------
    map_in : label

    Date  : June, 26 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """

    nl,ny,nx = STD_in.shape


    # Standardized STD Cube
    mask = (STD_in==0)
    VAR = np.repeat( np.var(STD_in,axis=0)[np.newaxis,:,:],nl,axis=0)
    VAR[mask] = np.inf
    x = STD_in/np.sqrt(VAR)

    # test
    test = np.mean(x**2, axis=0) - 1
    gamma =  stats.chi2.ppf(1-pfa, 1)

    # threshold - erosion and dilation to clean ponctual "source"
    sources = test>gamma
    sources = binary_erosion(sources,border_value=1,iterations=1)
    sources = binary_dilation(sources,iterations=1)

    # Label
    map_in = measurements.label(sources)[0]

    return map_in


def SpatioSpectral_Merging(cat_in, pfa, cube_cont, cor_in, var_in , deltaz): 
    """Merge the detected emission lines distants to less than deltaz
    spectral channel in a source area

    Parameters
    ---------
    Cat          : astropy.Table
                   Catalogue of detected emission lines
                   Columns of Cat:
                   ID x_circle y_circle ra_circle dec_circle
                   x_centroid y_centroid ra_centroid, dec_centroid nb_lines
                   x y z ra dec lbda T_GLR profile pvalC
                   residual flux num_line
    pfa          : Pvalue for the test which performs segmentation
    cube_cont    : array
                   cube of standardized continuum estimated from DCT    
    cor_in       : Array
                   Correlation Cube
    var_in       : Array
                   Variance Cube given or computed in preprocessing step

    deltaz       : integer
                   Distance maximum between 2 different lines

    Returns
    -------
    CatF : astropy.Table
           Catalogue
           Columns of CatF:
           ID x_circle y_circle ra_circle dec_circle x_centroid y_centroid
           ra_centroid dec_centroid nb_lines x y z ra dec lbda T_GLR profile
           pvalC residual flux num_line
    map_in       : Array
                   segmentation map           

    Date  : June, 23 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """

    nl,ny,nx = cor_in.shape

    # label
    map_in = Segmentation(cube_cont, pfa) 

    # MAX Spectra for sources with same ID
    _id = []
    _area = []
    _specmax = []
    for id_src in np.unique(cat_in['ID']):
        cat_src = cat_in[cat_in['ID']==id_src]
        mean_x = int(np.mean(cat_src['x']))
        mean_y = int(np.mean(cat_src['y']))
        max_spectre = np.amax(cor_in[:,cat_src['y'], cat_src['x']], axis=1)
        area = map_in[mean_y, mean_x]

        _id.append(id_src)
        _area.append(area)
        _specmax.append(np.argmax(max_spectre))

    _area = np.asarray(_area)
    _id = np.asarray(_id)
    _specmax = np.asarray(_specmax)

    # analyze per area
    # we skip background and when a single source is detected in an area
    _area2 = []
    _id2 = []
    _specmax2 = []
    for a_n in np.unique(_area):
        a_n = int(a_n)
        if a_n > 0:
            ID = _id[_area==a_n]
            if len(ID)>1:
                _area2.append(a_n)
                _id2.append(ID)
                _specmax2.append(_specmax[_area==a_n])

    # for each area with several sources
    # make comparison and give unique ID
    id_cal = []
    for area, id_cp, listcopy in zip(_area2, _id2, _specmax2):
        while len(listcopy)>0:

            loc = listcopy[0]
            idloc = id_cp[0]
            oth = listcopy[1:]
            idoth = id_cp[1:]

            arg = np.where( np.abs( loc - oth ) < deltaz )[0]

            id_cal.append( (idloc,idloc) )

            if arg.sum()>0:  # MATCH LOOP on found
                for n in arg:
                    id_cal.append( (idoth[n],idloc) )
                listcopy = np.delete(listcopy,1+arg)
                id_cp = np.delete(id_cp,1+arg)

            listcopy = np.delete(listcopy,0)
            id_cp = np.delete(id_cp,0)

    #% Process the catalog
    cat_out = cat_in.copy()
    ID_tmp = cat_out['ID']

    ID_arr = np.array(id_cal)
    col_old_id = Column(name='ID_old', data=cat_in['ID'])
    for n in ID_arr:
        for m in range(len(ID_tmp)):
            if ID_tmp[m] == n[0]:
                ID_tmp[m] = n[1]

    cat_out['ID'] = ID_tmp

    xc = np.zeros(len(cat_in))
    yc = np.zeros(len(cat_in))
    seg = np.zeros(len(cat_in), dtype=np.int)

    # Spatial Centroid and label of the segmentation map
    for ind_id_src in np.unique(cat_out['ID']):
        ind_id_src = int(ind_id_src)
        x = np.mean( cat_out[cat_out['ID']==ind_id_src]['x'] )
        y = np.mean( cat_out[cat_out['ID']==ind_id_src]['y'] )
        seg_label = map_in[int(y), int(x)]
        xc[cat_out['ID']==ind_id_src] = x
        yc[cat_out['ID']==ind_id_src] = y
        seg[cat_out['ID']==ind_id_src] = seg_label

    cat_out['x_centroid'] = xc
    cat_out['y_centroid'] = yc

    #save the label of the segmentation map
    col_seg_label = Column(name='seg_label', data=seg)
    cat_out.add_columns([col_old_id, col_seg_label])
    return cat_out, map_in


def Construct_Object(k, ktot, cols, units, desc, fmt, step_wave,
                     origin, filename, maxmap, segmap, correl, fwhm_profiles, 
                     param, path, name, ThresholdPval, i, ra, dec, x_centroid,
                     y_centroid, seg_label, wave_pix, GLR, num_profil, pvalC,
                     nb_lines, Cat_est_line_data, Cat_est_line_var,
                     Cat_est_cont_data, Cat_est_cont_var,
                     y, x, flux, purity, src_vers, author):
    """Function to create the final source

    Parameters
    ----------
    """

    logger = logging.getLogger('origin')
    logger.info('{}/{} source ID {}'.format(k+1,ktot,i))
    cube = Cube(filename)
    cubevers = cube.primary_header.get('CUBE_V', '')
    origin.append(cubevers)

    if type(maxmap) is str:
        maxmap_ = Image(maxmap)
    else:
        maxmap_ = maxmap


    src = Source.from_data(i, ra, dec, origin)
    src.add_attr('x', x_centroid, desc='x position in pixel',
                 unit=u.pix, fmt='d')
    src.add_attr('y', y_centroid, desc='y position in pixel',
                 unit=u.pix, fmt='d')
    src.add_attr('seglabel', seg_label, desc='label in the segmentation map',
                 fmt='d')

    src.add_white_image(cube)
    src.add_cube(cube, 'MUSE_CUBE')
    src.add_image(maxmap_, 'MAXMAP')
    if seg_label > 0:
        if type(segmap) is str:
            segmap_ = Image(segmap)
        else:
            segmap_ = segmap
        src.add_image(segmap_, 'SEG_ORIGIN')
    src.add_attr('SRC_V', src_vers, desc='Source version')

    src.add_history('Source created with Origin', author)

    w = cube.wave.coord(wave_pix, unit=u.angstrom)
    names = np.array(['%04d'%w[j] for j in range(nb_lines)])
    if np.unique(names).shape != names.shape:
        names = names.astype(np.int)
        while ((names[1:]-names[:-1]) == 0).any():
            names[1:][(names[:-1]-names[1:]) == 0] += 1
        names = names.astype(np.str)

    if type(correl) is str:
        correl_ = Cube(correl)
    else:
        correl_ = correl
        correl_.mask = cube.mask

    for j in range(nb_lines):
        sp_est = Spectrum(data=Cat_est_line_data[j, :],
                          wave=cube.wave)
        ct_est = Spectrum(data=Cat_est_cont_data[j, :],
                          wave=cube.wave)
        ksel = np.where(sp_est._data != 0)
        z1 = ksel[0][0]
        z2 = ksel[0][-1] + 1
        # Estimated line
        src.spectra['LINE_{:s}'.format(names[j])] = sp_est[z1:z2]
        # Estimated Continu
        src.spectra['CONT_{:s}'.format(names[j])] = ct_est[z1:z2]
        # correl
        src.spectra['CORR_{:s}'.format(names[j])] = correl_[z1:z2, y[j], x[j]]
        # FWHM in arcsec of the profile
        profile_num = num_profil[j]
        profil_FWHM = step_wave * fwhm_profiles[profile_num]
        #profile_dico = Dico[profile_num]
        fl = flux[j]
        pu = purity[j]
        vals = [w[j], profil_FWHM, fl, GLR[j], pvalC[j], profile_num,pu]
        src.add_line(cols, vals, units, desc, fmt)

        src.add_narrow_band_image_lbdaobs(cube,
                                        'NB_LINE_{:s}'.format(names[j]),
                                        w[j], width=2 * profil_FWHM,
                                        is_sum=True, subtract_off=True)
        src.add_narrow_band_image_lbdaobs(correl_,
                                        'NB_CORR_{:s}'.format(names[j]),
                                        w[j], width=2 * profil_FWHM,
                                        is_sum=True, subtract_off=True)
    # header
    if 'nbsubcube' in param.keys():
        src.OP_NS = (param['nbsubcube'], 'Orig nb of subcubes')                                  
    for i in range(ThresholdPval.shape[0]):
        src.header['OP_TH%02d'%i] = (ThresholdPval[i], 'ThresholdPval')
    if 'dct_order' in param.keys():
        src.OP_DCT = (param['dct_order'], 'Orig dct order')
    if 'Noise_population' in param.keys():
        src.OP_FBG = (param['Noise_population'], 'Orig fraction of spectra estimated as background')
    if 'threshold_test' in param.keys():
        src.OP_THV = (param['threshold_test'], 'Orig threshold')
    if 'itermax' in param.keys():
        src.OP_ITMAX = (param['itermax'], 'Orig maximum number of iterations')
    if 'neighboors' in param.keys():
        src.OP_NG = (param['neighboors'], 'Orig Neighboors')    
    if 'purity' in param.keys():
        src.OP_PURI = (param['purity'], 'Orig purity')
    if 'threshold_option' in param.keys():
        src.OP_THP = (param['threshold_option'], 'Orig threshold option')
    if 'pfa' in param.keys():
        src.OP_PFA = (param['pfa'], 'Orig pfa')
    if 'grid_dxy' in param.keys():
        src.OP_DXY = (param['grid_dxy'], 'Orig Grid Nxy')
    if 'deltaz' in param.keys():
        src.OP_DZ = (param['deltaz'], 'Orig deltaz')
    if 'PSF' in param.keys():
        src.OP_FSF = (param['PSF'], 'Orig FSF cube')
    src.write('%s/%s-%05d.fits' % (path, name, src.ID))


def Construct_Object_Catalogue(Cat, Cat_est_line, correl, wave, fwhm_profiles,
                               path_src, name, param, src_vers, author,
                               path, maxmap, segmap, Cat_est_cont,
                               ThresholdPval, ncpu=1):
    """Function to create the final catalogue of sources with their parameters

    Parameters
    ----------
    Cat              : Catalogue of parameters of detected emission lines:
                       ID x_circle y_circle x_centroid y_centroid nb_lines
                       x y z T_GLR profile pvalCresidual
                       flux num_line RA DEC
    Cat_est_line     : list of spectra
                       Catalogue of estimated lines
    Cat_est_cont     : list of spectra
                       Catalogue of roughly estimated continuum
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

    cols = ['LBDA_ORI', 'FWHM_ORI', 'FLUX_ORI', 'GLR', 'PVALC', 'PROF','PURITY']
    units = [u.Angstrom, u.Angstrom, uflux, unone, unone, unone, unone]
    fmt = ['.2f', '.2f', '.1f', '.1f', '.1e', 'd', '.2f']
    desc = None

    step_wave = wave.get_step(unit=u.angstrom)
    filename = param['cubename']
    origin = ['ORIGIN', __version__, os.path.basename(filename)]

    path2 = os.path.abspath(path) + '/' + name
    if os.path.isfile('%s/maxmap.fits'%path2):
        f_maxmap = '%s/maxmap.fits'%path2
    else:
        maxmap.write('%s/tmp_maxmap.fits'%path2)
        f_maxmap = '%s/tmp_maxmap.fits'%path2
    if os.path.isfile('%s/segmentation_map.fits'%path2):
        f_segmap = '%s/segmentation_map.fits'%path2
    else:
        segmap.write('%s/tmp_segmap.fits'%path2)
        f_segmap = '%s/tmp_segmap.fits'%path2
    if os.path.isfile('%s/cube_correl.fits'%path2):
        f_correl = '%s/cube_correl.fits'%path2
    else:
        correl.write('%s/tmp_cube_correl.fits'%path2)
        f_correl = '%s/tmp_cube_correl.fits'%path2

    sources_arglist = []

    for i in np.unique(Cat['ID']):
        # Source = group
        E = Cat[Cat['ID'] == i]
        ra = E['ra_centroid'][0]
        dec = E['dec_centroid'][0]
        x_centroid = E['x_centroid'][0]
        y_centroid = E['y_centroid'][0]
        seg_label = E['seg_label'][0]
        # Lines of this group
        wave_pix = E['z']
        GLR = E['T_GLR']
        num_profil = E['profile']
        pvalC = E['pvalC']
        # Number of lines in this group
        nb_lines = E['nb_lines'][0]
        Cat_est_line_data = np.empty((nb_lines, wave.shape))
        Cat_est_line_var = np.empty((nb_lines, wave.shape))
        Cat_est_cont_data = np.empty((nb_lines, wave.shape))
        Cat_est_cont_var = np.empty((nb_lines, wave.shape))
        for j in range(nb_lines):
            Cat_est_line_data[j,:] = Cat_est_line[E['num_line'][j]]._data
            Cat_est_line_var[j,:] = Cat_est_line[E['num_line'][j]]._var
            Cat_est_cont_data[j,:] = Cat_est_cont[E['num_line'][j]]._data
            Cat_est_cont_var[j,:] = Cat_est_cont[E['num_line'][j]]._var
        y = E['y']
        x = E['x']
        flux = E['flux']
        purity = E['purity']
        source_arglist = (i, ra, dec, x_centroid, y_centroid, seg_label,
                          wave_pix, GLR, num_profil, pvalC, nb_lines,
                          Cat_est_line_data, Cat_est_line_var,
                          Cat_est_cont_data, Cat_est_cont_var,
                          y, x, flux, purity,
                          src_vers, author)
        sources_arglist.append(source_arglist)

    if ncpu > 1:
        # run in parallel
        errmsg = Parallel(n_jobs=ncpu, max_nbytes=1e6)(
            delayed(Construct_Object)(k, len(sources_arglist), cols, units, desc,
                                      fmt, step_wave, origin, filename,
                                      f_maxmap, f_segmap, f_correl, fwhm_profiles, 
                                      param, path_src, name, ThresholdPval,
                                      *source_arglist)

            for k,source_arglist in enumerate(sources_arglist))
        # print error messages if any
        for msg in errmsg:
            if msg is None: continue
            logger.error(msg)
    else:
        for k,source_arglist in enumerate(sources_arglist):
            msg = Construct_Object(k, len(sources_arglist), cols, units, desc,
                                      fmt, step_wave, origin, filename,
                                      maxmap, segmap, correl, fwhm_profiles, 
                                      param, path_src, name, ThresholdPval,
                                      *source_arglist)

            if msg is not None:
                logger.error(msg)

    if os.path.isfile('%s/tmp_maxmap.fits'%path2):
        os.remove('%s/tmp_maxmap.fits'%path2)
    if os.path.isfile('%s/tmp_segmap.fits'%path2):
        os.remove('%s/tmp_segmap.fits'%path2)
    if os.path.isfile('%s/tmp_cube_correl.fits'%path2):
        os.remove('%s/tmp_cube_correl.fits'%path2)

    logger.debug('%s executed in %0.1fs' % (whoami(), time.time() - t0))
    return len(np.unique(Cat['ID']))


def whoami():
    return sys._getframe(1).f_code.co_name
