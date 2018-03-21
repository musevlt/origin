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

import astropy.units as u
import logging
import numpy as np
import os.path

from astropy.table import Table, Column, MaskedColumn
from astropy.modeling.models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from functools import wraps
from joblib import Parallel, delayed
from numpy import fft
from scipy import stats, fftpack
from scipy.signal import fftconvolve
from scipy.ndimage import measurements, filters
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial import ConvexHull
from scipy.sparse.linalg import svds
from scipy.interpolate import interp1d
from time import time

from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.sdetect import Source

from .source_masks import gen_source_mask
from .version import __version__


def timeit(f):
    """Decorator which prints the execution time of a function."""
    @wraps(f)
    def timed(*args, **kw):
        logger = logging.getLogger('origin')
        t0 = time()
        result = f(*args, **kw)
        logger.debug('%s executed in %0.1fs', f.__name__, time() - t0)
        return result
    return timed


def isnotebook():  # pragma: no cover
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def ProgressBar(*args, **kwargs):
    from tqdm import tqdm, tqdm_notebook
    func = tqdm_notebook if isnotebook() else tqdm
    return func(*args, **kwargs)


def orthogonal_projection(a, b):
    """Compute the orthogonal projection: ``np.dot(np.dot(a, a.T), b)``."""
    # Using multi_dot which is faster than np.dot(np.dot(a, a.T), b)
    # Another option would be to use einsum, less readable but also very
    # fast with Numpy 1.14+ and optimize=True. This seems to be as fast as
    # multi_dot.
    # return np.einsum('i,j,jk->ik', a, a, b, optimize=True)
    if a.ndim == 1:
        a = a[:, None]
    return np.linalg.multi_dot([a, a.T, b])


@timeit
def Spatial_Segmentation(Nx, Ny, NbSubcube, start=None):
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
    start     : tuple
                if not None, the tupe is the (y,x) starting point
    Returns
    -------
    intx, inty : integer, integer
                  limits in pixels of the columns/rows for each zone

    Date  : Dec,10 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    # Segmentation of the rows vector in Nbsubcube parts from the right to the
    # left
    inty = np.linspace(Ny, 0, NbSubcube + 1, dtype=np.int)
    # Segmentation of the columns vector in Nbsubcube parts from the left to
    # the right
    intx = np.linspace(0, Nx, NbSubcube + 1, dtype=np.int)

    if start is not None:
        inty += start[0]
        intx += start[1]

    return inty, intx


def DCTMAT(nl, order):
    """Return the DCT Matrix[:,:order+1] of nl

    Parameters
    ----------
    order : int
        order of the dct (spectral length)

    Returns
    -------
    array: DCT Matrix

    """
    yy, xx = np.mgrid[:nl, :order + 1]
    D0 = np.sqrt(2 / nl) * np.cos((xx + 0.5) * (np.pi / nl) * yy)
    D0[0, :] /= np.sqrt(2)
    return D0


def continuum(D0, D0T, var, w_raw_var):
    A = np.linalg.inv(np.dot(D0T / var, D0))
    return np.dot(np.dot(np.dot(D0, A), D0T), w_raw_var)


@timeit
def dct_residual(w_raw, order, var, approx):
    """Function to compute the residual of the DCT on raw data.

    Parameters
    ----------
    RAW     :   array
                the RAW data

    order   :   integer
                The number of atom to keep for the dct decomposition

    var : array
          Variance

    Returns
    -------
    Faint     : array
                residual from the dct decomposition

    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """
    nl = w_raw.shape[0]
    D0 = DCTMAT(nl, order)
    if approx:
        A = np.dot(D0, D0.T)
        cont = np.tensordot(A, w_raw, axes=(0, 0))
    else:
        w_raw_var = w_raw / var
        D0T = D0.T
        cont = Parallel()(
            delayed(continuum)(D0, D0T, var[:, i, j], w_raw_var[:, i, j])
            for i in range(w_raw.shape[1]) for j in range(w_raw.shape[2]))
        cont = np.asarray(cont).T.reshape(w_raw.shape)
    #    cont = np.empty_like(w_raw)
    #    for i in range(w_raw.shape[1]):
    #        for j in range(w_raw.shape[2]):
    #            A = np.linalg.inv(np.dot(D0T/var[:,i,j], D0))
    #            cont[:,i,j] = np.dot(np.dot(np.dot(D0,A),D0T), w_raw_var[:,i,j])

    Faint = w_raw - cont
    return Faint, cont


@timeit
def Compute_Standardized_data(cube_dct, mask, var):
    """Function to compute the standardized data.

    Parameters
    ----------
    cube_dct:   array
                output of dct_residual
    mask  :   array
             Mask array (expmap==0)
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
    cube_dct[mask] = np.nan
    mean_lambda = np.nanmean(cube_dct, axis=(1, 2))
    var[mask] = np.inf

    STD = (cube_dct - mean_lambda[:, np.newaxis, np.newaxis]) / np.sqrt(var)
    STD[mask] = 0
    return STD


def createradvar(cu, ot):
    """Function to compute the compactness of areas using variance of
    position. The variance is computed on the position given by
    adding one of the 'ot' to 'cu'

    Parameters
    ----------
    cu :   2D array
           The current array
    ot :   3D array
           The other array

    Returns
    -------
    var :     array
              The radial variances

    Date  : Sept,27 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """
    N = ot.shape[0]
    out = np.zeros(N)
    for n in range(N):
        tmp = cu + ot[n, :, :]
        y, x = np.where(tmp > 0)
        r = np.sqrt((y - y.mean())**2 + (x - x.mean())**2)
        out[n] = np.var(r)
    return out


def fusion_areas(label, MinSize, MaxSize, option=None):
    """Function which merge areas which have a surface less than
    MinSize if the size after merging is less than MaxSize.
    The criteria of neighbor can be related to the minimum surface
    or to the compactness of the output area

    Parameters
    ----------
    label   :   area
                The labels of areas
    MinSize :   number
                The size of areas under which they need to merge
    MaxSize :   number
                The size of areas above which they cant merge
    option  :   string
                if 'var' the compactness criteria is used
                if None the minimum surface criteria is used

    Returns
    -------
    label :     array
                The labels of merged areas

    Date  : Sept,27 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """
    while True:
        indlabl = np.argsort(np.sum(label, axis=(1, 2)))
        tampon = label.copy()
        for n in indlabl:
            # if the label is not empty
            cu = label[n, :, :]
            cu_size = np.sum(cu)

            if cu_size > 0 and cu_size < MinSize:
                # search for neighbors
                labdil = label[n, :, :].copy()
                labdil = binary_dilation(labdil, iterations=1)

                # only neighbors
                test = np.sum(label * labdil[np.newaxis, :, :], axis=(1, 2)) > 0

                indice = np.where(test == 1)[0]
                ind = np.where(indice != n)[0]
                indice = indice[ind]

                # BOUCLER SUR LES CANDIDATS
                ot = label[indice, :, :]

                # test size of current with neighbor
                if option is None:
                    test = np.sum(ot, axis=(1, 2))
                elif option == 'var':
                    test = createradvar(cu, ot)
                else:
                    raise IOError('bad o^ption')

                if len(test) > 0:
                    # keep the min-size
                    ind = np.argmin(test)
                    cand = indice[ind]
                    if (np.sum(label[n, :, :]) + test[ind]) < MaxSize:
                        label[n, :, :] += label[cand, :, :]
                        label[cand, :, :] = 0

        # clean empty area
        ind = np.sum(label, axis=(1, 2)) > 0
        label = label[ind, :, :]
        tampon = tampon[ind, :, :]

        if np.sum(tampon - label) == 0:
            break
    return label


@timeit
def area_segmentation_square_fusion(nexpmap, MinS, MaxS, NbSubcube, Ny, Nx):
    """Function to create non square area based on continuum test. The full
    2D image is first segmented in subcube. The area are fused in case they
    are too small. Thanks to the continuum test, detected sources are
    fused with associated area. The convex enveloppe of the sources inside
    each area is then done. Finally all the convex enveloppe growth until
    using all the pixels

    Parameters
    ----------
    nexpmap :   2D array
                the active pixel of the image
    MinS    :   number
                The size of areas under which they need to merge
    MaxS    :   number
                The size of areas above which they cant merge
    NbSubcube : integer
                Number of subcubes for the spatial segmentation
    Nx        : integer
                Number of columns
    Ny        : integer
                Number of rows


    Returns
    -------
    label :     array
                label of the fused square

    Date  : Sept,13 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """
    # square area index with borders
    Vert = np.sum(nexpmap, axis=1)
    Hori = np.sum(nexpmap, axis=0)
    y1 = np.where(Vert > 0)[0][0]
    x1 = np.where(Hori > 0)[0][0]
    y2 = Ny - np.where(Vert[::-1] > 0)[0][0]
    x2 = Nx - np.where(Hori[::-1] > 0)[0][0]
    start = (y1, x1)
    inty, intx = Spatial_Segmentation(Nx, Ny, NbSubcube, start=start)

    # % FUSION square AREA
    label = []
    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            y1, y2, x1, x2 = inty[numy + 1], inty[numy], intx[numx], intx[numx + 1]
            tmp = nexpmap[y1:y2, x1:x2]
            if np.mean(tmp) != 0:
                labtest = measurements.label(tmp)[0]
                labtmax = labtest.max()

                for n in range(labtmax):
                    label_tmp = np.zeros((Ny, Nx))
                    label_tmp[y1:y2, x1:x2] = (labtest == (n + 1))
                    label.append(label_tmp)

    label = np.array(label)
    return fusion_areas(label, MinS, MaxS)


@timeit
def area_segmentation_sources_fusion(labsrc, label, pfa, Ny, Nx):
    """Function to create non square area based on continuum test. Thanks
    to the continuum test, detected sources are fused with associated area.
    The convex enveloppe of the sources inside
    each area is then done. Finally all the convex enveloppe growth until
    using all the pixels

    Parameters
    ----------
    labsrc : array
             segmentation map
    label :     array
                label of fused square generated in
                area_segmentation_square_fusion
    pfa   :     float
                Pvalue for the test which performs segmentation
    NbSubcube : integer
                Number of subcubes for the spatial segmentation
    Nx        : integer
                Number of columns
    Ny        : integer
                Number of rows


    Returns
    -------
    label_out : array
                label of the fused square and sources

    Date  : Sept,13 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """

    # compute the sources label
    nlab = labsrc.max()
    sources = np.zeros((nlab, Ny, Nx))
    for n in range(1, nlab + 1):
        sources[n - 1, :, :] = (labsrc == n) > 0
    sources_save = sources.copy()

    nlabel = label.shape[0]
    nsrc = sources.shape[0]
    for n in range(nsrc):
        cu_src = sources[n, :, :]
        # find the area in which the current source
        # has bigger probability to be

        test = np.sum(cu_src[np.newaxis, :, :] * label, axis=(1, 2))
        if len(test) > 0:
            ind = np.argmax(test)
            # associate the source to the label
            label[ind, :, :] = (label[ind, :, :] + cu_src) > 0
            # mask other labels from this sources
            mask = (1 - label[ind, :, :])[np.newaxis, :, :]
            ot_lab = np.delete(np.arange(nlabel), ind)
            label[ot_lab, :, :] *= mask
            # delete the source
            sources[n, :, :] = 0

    return label, np.sum(sources_save, axis=0)


@timeit
def area_segmentation_convex_fusion(label, src):
    """Function to compute the convex enveloppe of the sources inside
    each area is then done. Finally all the convex enveloppe growth until
    using all the pixels

    Parameters
    ----------
    label :     array
                label containing the fusion of fused squares and sources
                generated in area_segmentation_sources_fusion
    src :       array
                label of estimated sources from segmentation map

    Returns
    -------
    label_out : array
                label of the convex

    Date  : Sept,13 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """
    label_fin = []
    # for each label
    for lab_n in range(label.shape[0]):

        # keep only the sources inside the label
        lab = label[lab_n, :, :]
        data = src * lab
        if np.sum(data > 0):
            points = np.array(np.where(data > 0)).T

            y_0 = points[:, 0].min()
            x_0 = points[:, 1].min()

            points[:, 0] -= y_0
            points[:, 1] -= x_0

            sny, snx = points[:, 0].max() + 1, points[:, 1].max() + 1
            # compute the convex enveloppe of a sub part of the label
            lab_temp = Convexline(points, snx, sny)

            # in full size
            label_out = np.zeros((label.shape[1], label.shape[2]))
            label_out[y_0:y_0 + sny, x_0:x_0 + snx] = lab_temp
            label_out *= lab
            label_fin.append(label_out)

    return np.array(label_fin)


def Convexline(points, snx, sny):
    """Function to compute the convex enveloppe of the sources inside
    each area is then done and full the polygone

    Parameters
    ----------
    data :      array
                contain the position of source for one of the label
    snx,sny:    int,int
                the effective size of area in the label

    Returns
    -------
    lab_out :   array
                The filled convex enveloppe corresponding the sub label

    Date  : Sept,13 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """

    # convex enveloppe vertices
    hull = ConvexHull(points)

    xs = hull.points[hull.simplices[:, 1]]
    xt = hull.points[hull.simplices[:, 0]]

    sny, snx = points[:, 0].max() + 1, points[:, 1].max() + 1
    tmp = np.zeros((sny, snx))

    # create le line between vertices
    for n in range(hull.simplices.shape[0]):
        x0, x1, y0, y1 = xs[n, 1], xt[n, 1], xs[n, 0], xt[n, 0]

        nx = np.abs(x1 - x0)
        ny = np.abs(y1 - y0)
        if ny > nx:
            xa, xb, ya, yb = y0, y1, x0, x1
        else:
            xa, xb, ya, yb = x0, x1, y0, y1
        if xa > xb:
            xb, xa, yb, ya = xa, xb, ya, yb

        indx = np.arange(xa, xb, dtype=int)
        N = len(indx)
        indy = np.array(ya + (indx - xa) * (yb - ya) / N, dtype=int)

        if ny > nx:
            tmpx, tmpy = indx, indy
            indy, indx = tmpx, tmpy

        tmp[indy, indx] = 1

    radius = 1
    dxy = 2 * radius
    x = np.linspace(-dxy, dxy, 1 + (dxy) * 2)
    y = np.linspace(-dxy, dxy, 1 + (dxy) * 2)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv**2 + yv**2)
    mask = (np.abs(r) <= radius)

    # to close the lines
    conv_lab = fftconvolve(tmp, mask, mode='same') > 1e-9

    lab_out = conv_lab.copy()
    for n in range(conv_lab.shape[0]):
        ind = np.where(conv_lab[n, :] == 1)[0]
        lab_out[n, ind[0]:ind[-1]] = 1

    return lab_out


@timeit
def area_growing(label, mask):
    """Growing and merging of all areas

    Parameters
    ----------
    label :     array
                label containing convex enveloppe of each area
    mask :      array
                mask of positive pixels

    Returns
    -------
    label_out : array
                label of the convex envelop grown to the max number of pixels

    Date  : Sept,13 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """
    # start by smaller
    set_ind = np.argsort(np.sum(label, axis=(1, 2)))
    # closure horizon
    niter = 20

    label_out = label.copy()
    nlab = label_out.shape[0]
    while True:
        s = np.sum(label_out)
        for n in set_ind:
            cu_lab = label_out[n, :, :]
            ind = np.delete(np.arange(nlab), n)
            ot_lab = label_out[ind, :, :]
            border = (1 - (np.sum(ot_lab, axis=0) > 0)) * mask
            # closure in all case + 1 dilation
            cu_lab = binary_dilation(cu_lab, iterations=niter + 1)
            cu_lab = binary_erosion(cu_lab, border_value=1, iterations=niter)
            label_out[n, :, :] = cu_lab * border
        if np.sum(label_out) == np.sum(mask) or np.sum(label_out) == s:
            break

    return label_out


@timeit
def area_segmentation_final(label, MinS, MaxS):
    """Merging of small areas and give index

    Parameters
    ----------
    label :   array
              label containing convex enveloppe of each area
    MinS    :   number
                The size of areas under which they need to merge
    MaxS    :   number
                The size of areas above which they cant merge

    Returns
    -------
    sety,setx : array
                list of index of each label

    Date  : Sept,13 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """
    # if an area is too small
    label = fusion_areas(label, MinS, MaxS, option='var')

    # create label map
    areamap = np.zeros(label.shape[1:])
    for i in range(label.shape[0]):
        areamap[label[i, :, :] > 0] = i + 1
    return areamap


@timeit
def Compute_GreedyPCA_area(NbArea, cube_std, areamap, Noise_population,
                           threshold_test, itermax, testO2):
    """Function to compute the PCA on each zone of a data cube.

    Parameters
    ----------
    NbArea           : integer
                       Number of area
    cube_std         : array
                       Cube data weighted by the standard deviation
    areamap          : array
                       Map of areas
    Noise_population : float
                       Proportion of estimated noise part used to define the
                       background spectra
    threshold_test   : list
                       User given list of threshold (not pfa) to apply
                       on each area, the list is of lenght NbAreas
                       or of lenght 1.
    itermax          : integer
                       Maximum number of iterations
    testO2           : list of arrays
                       Result of the O2 test
    Returns
    -------
    cube_faint : array
                Faint greedy decomposition od STD Cube

    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """
    cube_faint = cube_std.copy()
    mapO2 = np.zeros(cube_std.shape[1:])
    nstop = 0
    area_iter = range(1, NbArea + 1)
    if NbArea > 1:
        area_iter = ProgressBar(area_iter)

    for area_ind in area_iter:
        # limits of each spatial zone
        ksel = (areamap == area_ind)

        # Data in this spatio-spectral zone
        cube_temp = cube_std[:, ksel]

        thr = threshold_test[area_ind - 1]
        test = testO2[area_ind - 1]
        cube_faint[:, ksel], mO2, kstop = Compute_GreedyPCA(
            cube_temp, test, thr, Noise_population, itermax)
        mapO2[ksel] = mO2
        nstop += kstop

    return cube_faint, mapO2, nstop


def Compute_PCA_threshold(faint, pfa_test):
    """

    Parameters
    ----------
    faint   :   array
                The 3D cube data clean
    pfa_test         : float
                       PFA of the test

    Returns
    -------
    histO2:
    frecO2:
    thresO2 :   Threshold for the O2 test
    """
    test = O2test(faint)

    # automatic threshold computation
    histO2, frecO2, thresO2, mea, std = Compute_thresh_PCA_hist(test, pfa_test)

    return test, histO2, frecO2, thresO2, mea, std


def Compute_GreedyPCA(cube_in, test, thresO2, Noise_population, itermax):
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

    Noise_population : float
                       Fraction of spectra estimated as background
    itermax          : integer
                       Maximum number of iterations

    Returns
    -------
    faint   :   array
                cleaned cube
    mapO2   :   array
                2D MAP filled with the number of iteration per spectra
    thresO2 :   float
                Threshold for the O2 test
    nstop   :   int
                Nb of times the iterations have been stopped when > itermax


    Date  : Mar, 28 2017
    Author: antony schutz (antonyschutz@gmail.com)
    """
    logger = logging.getLogger(__name__)

    # nuisance part
    pypx = np.where(test > thresO2)[0]
    npix = len(pypx)

    faint = cube_in.copy()
    mapO2 = np.zeros(faint.shape[1])
    nstop = 0

    with ProgressBar(total=npix, miniters=0) as bar:
        # greedy loop based on test
        nbiter = 0
        while len(pypx) > 0:
            nbiter += 1
            mapO2[pypx] += 1
            if nbiter > itermax:
                nstop += 1
                logger.info('Warning iterations stopped at %d', nbiter)
                break

            # vector data
            test_v = np.ravel(test)
            test_v = test_v[test_v > 0]
            nind = np.where(test_v <= thresO2)[0]
            sortind = np.argsort(test_v[nind])

            # at least one spectra is used to perform the test
            nb = 1 + int(len(nind) / Noise_population)

            # background estimation
            b = np.mean(faint[:, nind[sortind[:nb]]], axis=1)

            # cube segmentation
            x_red = faint[:, pypx]

            # orthogonal projection with background.
            x_red -= orthogonal_projection(b, x_red)
            x_red /= np.nansum(b**2)

            # remove spectral mean from residual data
            x_red -= x_red.mean(axis=1)[:, np.newaxis]

            # sparse svd if nb spectrum > 1 else normal svd
            if x_red.shape[1] == 1:
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
                U, s, V = np.linalg.svd(x_red, full_matrices=False)
            else:
                U, s, V = svds(x_red, k=1)

            # orthogonal projection
            faint -= orthogonal_projection(U[:, 0], faint)

            # test
            test = O2test(faint)

            # nuisance part
            pypx = np.where(test > thresO2)[0]
            bar.update(npix - len(pypx) - bar.n)

        bar.update(npix - len(pypx) - bar.n)

    return faint, mapO2, nstop


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
    # np.einsum('ij,ij->j', Cube_in, Cube_in) / Cube_in.shape[0]
    return np.mean(Cube_in**2, axis=0)


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
    logger = logging.getLogger(__name__)
    test_v = np.ravel(test)
    c = test_v[test_v > 0]
    histO2, frecO2 = np.histogram(c, bins='fd', normed=True)
    ind = np.argmax(histO2)
    mod = frecO2[ind]
    ind2 = np.argmin((histO2[ind] / 2 - histO2[:ind])**2)
    fwhm = mod - frecO2[ind2]
    sigma = fwhm / np.sqrt(2 * np.log(2))

    coef = stats.norm.ppf(threshold_test)
    thresO2 = mod - sigma * coef
    logger.debug('1st estimation mean/std/threshold: %f/%f/%f',
                 mod, sigma, thresO2)

    x = (frecO2[1:] + frecO2[:-1]) / 2
    g1 = Gaussian1D(amplitude=histO2.max(), mean=mod, stddev=sigma)
    fit_g = LevMarLSQFitter()
    xcut = g1.mean + gaussian_sigma_to_fwhm * g1.stddev / 2
    ksel = x < xcut
    g2 = fit_g(g1, x[ksel], histO2[ksel])
    mea, std = (g2.mean.value, g2.stddev.value)
    thresO2 = mea - std * coef

    return histO2, frecO2, thresO2, mea, std


def Correlation_GLR_test_zone(cube, sigma, PSF_Moffat, weights, Dico,
                              intx, inty, NbSubcube, threads):
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
    threads    : integer
                 number of threads

    Returns
    -------
    correl  : array
              cube of T_GLR values
    profile : array
              Number of the profile associated to the T_GLR

    Date  : Jul, 4 2017
    Author: Antony Schutz (antonyschutz@gmail.com)
    """
    logger = logging.getLogger(__name__)
    # initialization
    # size psf
    if weights is None:
        sizpsf = PSF_Moffat.shape[1]
    else:
        sizpsf = PSF_Moffat[0].shape[1]
    longxy = int(sizpsf // 2)

    Nl, Ny, Nx = cube.shape
    correl = np.zeros(cube.shape)
    correl_min = np.zeros(cube.shape)
    profile = np.zeros(cube.shape)

    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            logger.info('Area %d,%d / (%d,%d)',
                        numy + 1, numx + 1, NbSubcube, NbSubcube)
            # limits of each spatial zone, with PSF margins
            x1 = np.maximum(0, intx[numx] - longxy)
            x2 = np.minimum(intx[numx + 1] + longxy, Nx)
            y1 = np.maximum(0, inty[numy + 1] - longxy)
            y2 = np.minimum(inty[numy] + longxy, Ny)

            mini_cube = cube[:, y1:y2, x1:x2]
            mini_sigma = sigma[:, y1:y2, x1:x2]
            c, p, cm = Correlation_GLR_test(mini_cube, mini_sigma, PSF_Moffat,
                                            weights, Dico, threads)

            x11 = intx[numx] - x1
            y11 = inty[numy + 1] - y1
            x22 = intx[numx + 1] - x1
            y22 = inty[numy] - y1
            sy = slice(inty[numy + 1], inty[numy])
            sx = slice(intx[numx], intx[numx + 1])
            correl[:, sy, sx] = c[:, y11:y22, x11:x22]
            profile[:, sy, sx] = p[:, y11:y22, x11:x22]
            correl_min[:, sy, sx] = cm[:, y11:y22, x11:x22]

    return correl, profile, correl_min


def _convolve_fsf(psf, cube, sigma, weights=None):
    # Inverse of the MUSE covariance
    inv_var = 1. / sigma
    # data cube weighted by the MUSE covariance
    cube_var = cube * np.sqrt(inv_var)

    if weights is not None:
        cube_var *= weights
        inv_var *= weights

    psf = np.ascontiguousarray(psf[::-1, ::-1])
    psf -= psf.mean()

    # build a weighting map per PSF and convolve
    cube_fsf = fftconvolve(cube_var, psf, mode='same')

    # Spatial part of the norm of the 3D atom
    psf **= 2
    norm_fsf = fftconvolve(inv_var, psf, mode='same')

    return cube_fsf, norm_fsf


def _convolve_profile(Dico, cube_fft, norm_fft, fshape, n_jobs, parallel):
    # Second cube of correlation values
    dico_fft = fft.rfftn(Dico, fshape)[:, None] * cube_fft
    cube_profile = _convolve_spectral(parallel, n_jobs, dico_fft, fshape,
                                      func=fft.irfftn)
    dico_fft = fft.rfftn(Dico ** 2, fshape)[:, None] * norm_fft
    norm_profile = _convolve_spectral(parallel, n_jobs, dico_fft, fshape,
                                      func=fft.irfftn)

    # dico_fft = fft.rfftn(Dico, fshape)[:, None] * cube_fft
    # cube_profile = fft.irfftn(dico_fft, fshape, axes=(0,))
    # dico_fft = fft.rfftn(Dico_sq, fshape)[:, None] * norm_fft
    # norm_profile = fft.irfftn(dico_fft, fshape, axes=(0,))

    norm_profile[norm_profile <= 0] = np.inf
    np.sqrt(norm_profile, out=norm_profile)
    cube_profile /= norm_profile

    return cube_profile


def _convolve_spectral(parallel, nslices, arr, shape, func=fft.rfftn):
    arr = np.array_split(arr, nslices, axis=-1)
    out = parallel(delayed(func)(chunk, shape, axes=(0,)) for chunk in arr)
    return np.concatenate(out, axis=-1)


@timeit
def Correlation_GLR_test(cube, sigma, PSF_Moffat, weights, Dico, threads):
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
    threads    : integer
                 number of threads

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
    logger = logging.getLogger(__name__)

    # Dimensions of the data
    Nz, Ny, Nx = cube.shape

    # Spatial convolution of the weighted data with the zero-mean FSF
    logger.info('Step 1/3 and 2/3: '
                'Spatial convolution of weighted data with the zero-mean FSF, '
                'Computing Spatial part of the norm of the 3D atoms')
    if weights is None:  # one FSF
        PSF_Moffat = [PSF_Moffat]
        weights = [None]

    nfields = len(PSF_Moffat)

    with Parallel(n_jobs=threads) as parallel:
        for nf in range(nfields):
            res = parallel(ProgressBar([
                delayed(_convolve_fsf)(PSF_Moffat[nf][i], cube[i],
                                       sigma[i], weights=weights[nf])
                for i in range(Nz)
            ]))
            res = [np.stack(arr) for arr in zip(*res)]
            if nf == 0:
                cube_fsf, norm_fsf = res
            else:
                cube_fsf += res[0]
                norm_fsf += res[1]

    # First cube of correlation values
    # initialization with the first profile
    logger.info('Step 3/3 Computing second cube of correlation values')

    Dico = np.array(Dico)
    Dico -= np.mean(Dico, axis=1)[:, None]

    s1 = np.array(cube_fsf.shape)
    s2 = np.array((Dico.shape[1], 1, 1))
    fftshape = s1 + s2 - 1
    fshape = [fftpack.helper.next_fast_len(int(d)) for d in fftshape[:1]]
    startind = (fftshape - s1) // 2
    endind = startind + s1
    cslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    with Parallel(n_jobs=threads, backend='threading') as parallel:
        cube_fft = _convolve_spectral(parallel, threads, cube_fsf, fshape,
                                      func=fft.rfftn)
        norm_fft = _convolve_spectral(parallel, threads, norm_fsf, fshape,
                                      func=fft.rfftn)

    # cube_fft, norm_fft = Parallel(n_jobs=min(2, threads), backend='threading')(
    #     delayed(fft.rfftn)(arr, fshape, axes=(0,))
    #     for arr in (cube_fsf, norm_fsf))

    cube_fft = cube_fft.reshape(cube_fft.shape[0], -1)
    norm_fft = norm_fft.reshape(norm_fft.shape[0], -1)
    profile = np.empty((Nz, Ny * Nx), dtype=np.int)
    correl = np.full((Nz, Ny * Nx), -np.inf)
    correl_min = np.full((Nz, Ny * Nx), np.inf)

    with Parallel(n_jobs=threads, backend='threading') as parallel:
        for k in ProgressBar(range(len(Dico))):
            cube_profile = _convolve_profile(Dico[k], cube_fft, norm_fft,
                                             fshape, threads, parallel)
            cube_profile = cube_profile[cslice[0]]
            # norm_profile = norm_profile[cslice]
            profile[cube_profile > correl] = k
            np.maximum(correl, cube_profile, out=correl)
            np.minimum(correl_min, cube_profile, out=correl_min)

    profile = profile.reshape(Nz, Ny, Nx)
    correl = correl.reshape(Nz, Ny, Nx)
    correl_min = correl_min.reshape(Nz, Ny, Nx)
    return correl, profile, correl_min


@timeit
def Compute_local_max_zone(correl, correl_min, mask, intx, inty,
                           NbSubcube, neighbors):
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
    neighbors:  int
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
    # initialization
    cube_Local_max = np.zeros(correl.shape)
    cube_Local_min = np.zeros(correl.shape)
    cube_Local_max = np.zeros(correl.shape)
    cube_Local_min = np.zeros(correl.shape)
    nl, Ny, Nx = correl.shape
    lag = 1

    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            # limits of each spatial zone

            x1 = np.maximum(0, intx[numx] - lag)
            x2 = np.minimum(intx[numx + 1] + lag, Nx)
            y1 = np.maximum(0, inty[numy + 1] - lag)
            y2 = np.minimum(inty[numy] + lag, Ny)

            x11 = intx[numx] - x1
            y11 = inty[numy + 1] - y1
            x22 = intx[numx + 1] - x1
            y22 = inty[numy] - y1

            correl_temp_edge = correl[:, y1:y2, x1:x2]
            correl_temp_edge_min = correl_min[:, y1:y2, x1:x2]
            mask_temp_edge = mask[:, y1:y2, x1:x2]
            # Cube of pvalues for each zone
            cube_Local_max_temp, cube_Local_min_temp = Compute_localmax(
                correl_temp_edge, correl_temp_edge_min, mask_temp_edge,
                neighbors)

            cube_Local_max[:, inty[numy + 1]:inty[numy], intx[numx]:intx[numx + 1]] =\
                cube_Local_max_temp[:, y11:y22, x11:x22]
            cube_Local_min[:, inty[numy + 1]:inty[numy], intx[numx]:intx[numx + 1]] =\
                cube_Local_min_temp[:, y11:y22, x11:x22]

    return cube_Local_max, cube_Local_min


def _mask_circle_region(data, x0, y0, z0, spat_rad, spect_rad, thrdata=None, mthrdata=None):
    x, y = np.meshgrid(np.arange(data.shape[2]), np.arange(data.shape[1]))
    ksel = ((x - x0)**2 + (y - y0)**2) < spat_rad**2
    z1 = np.maximum(0, z0 - spect_rad)
    z2 = np.minimum(data.shape[0], z0 + spect_rad)
    if thrdata is None or mthrdata is None:
        data[z1:z2, ksel] = 0
    else:
        ksel2 = (thrdata[z1:z2, ksel] <= np.max(mthrdata[z1:z2, ksel]))
        data[z1:z2, ksel][ksel2] = 0


def CleanCube(Mdata, mdata, CatM, catm, Nz, Nx, Ny, spat_size, spect_size):
    (zM, yM, xM) = (CatM['z0'], CatM['y0'], CatM['x0'])
    (zm, ym, xm) = catm
    spat_rad = int(spat_size / 2)
    spect_rad = int(spect_size / 2)

    for n, z in enumerate(zm):
        _mask_circle_region(mdata, xm[n], ym[n], z, spat_rad, spect_rad)
    for n, z in enumerate(zM):
        _mask_circle_region(Mdata, xM[n], yM[n], z, spat_rad, spect_rad)

    return Mdata, mdata


def Compute_localmax(correl_temp_edge, correl_temp_edge_min,
                     mask_temp_edge, neighbors):
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
    neighbors        :  int
                        Number of connected components
    Returns
    -------
    cube_pval_correl : array
                       p-values asssociated to the local maxima of T_GLR values

    Date  : June, 19 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    # connected components
    conn = (neighbors + 1)**(1 / 3.)
    # local maxima of maximum correlation
    Max_filter = filters.maximum_filter(correl_temp_edge, size=(conn, conn, conn))
    Local_max_mask = (correl_temp_edge == Max_filter)
    Local_max_mask[mask_temp_edge] = 0
    Local_max = correl_temp_edge * Local_max_mask

    # local maxima of minus minimum correlation
    minus_correl_min = - correl_temp_edge_min
    Max_filter = filters.maximum_filter(minus_correl_min,
                                        size=(conn, conn, conn))
    Local_min_mask = (minus_correl_min == Max_filter)
    Local_min_mask[mask_temp_edge] = 0
    Local_min = minus_correl_min * Local_min_mask

    return Local_max, Local_min


def itersrc_mat(cat, coord, spatdist, area, tol_spat, tol_spec, n, iin, id_cu, IDorder):
    # MATRIX VERSION faster for smaller data
    xout, yout, zout, aout, iout = cat
    z, y, x = coord

    ind = np.where(spatdist[n, :] < tol_spat)[0]
    if len(ind) > 0:
        for indn in ind:
            if iin[indn] > 0:

                if spatdist[id_cu, indn] > tol_spat * np.sqrt(2):
                    # check spectral content
                    dz = np.sqrt((z[indn] - z[id_cu])**2)
                    if dz < tol_spec:
                        xout.append(x[indn])
                        yout.append(y[indn])
                        zout.append(z[indn])
                        aout.append(area[indn])

                        iout.append(id_cu)
                        iin[indn] = 0
                        spatdist[:, IDorder[indn]] = np.inf
                        cat = [xout, yout, zout, aout, iout]
                        coord = [z, y, x]
                        xout, yout, zout, aout, iout, spatdist, iin = \
                            itersrc_mat(cat, coord, spatdist, area, tol_spat,
                                        tol_spec, indn, iin, id_cu, IDorder)

                        spatdist[indn, :] = np.inf

                else:
                    xout.append(x[indn])
                    yout.append(y[indn])
                    zout.append(z[indn])
                    aout.append(area[indn])

                    iout.append(id_cu)
                    iin[indn] = 0
                    spatdist[:, IDorder[indn]] = np.inf
                    cat = [xout, yout, zout, aout, iout]
                    coord = [z, y, x]
                    xout, yout, zout, aout, iout, spatdist, iin = \
                        itersrc_mat(cat, coord, spatdist, area, tol_spat,
                                    tol_spec, indn, iin, id_cu, IDorder)

                    spatdist[indn, :] = np.inf

    return xout, yout, zout, aout, iout, spatdist, iin


def spatiospectral_merging_mat(z, y, x, map_in, tol_spat, tol_spec):
    # MATRIX VERSION faster for smaller data
    Nz = len(z)
    IDorder = np.arange(Nz)
    area = map_in[y, x]

    difx = x[np.newaxis, :].T - x[np.newaxis, :]
    dify = y[np.newaxis, :].T - y[np.newaxis, :]

    spatdist = np.sqrt(difx**2 + dify**2)
    spatdist[np.arange(Nz), np.arange(Nz)] = np.inf

    xout = []
    yout = []
    zout = []
    iout = []
    aout = []

    iin = np.ones(IDorder.shape)
    for n in IDorder:
        if iin[n] == 1:
            iin[n] = 0
            xout.append(x[n])
            yout.append(y[n])
            zout.append(z[n])
            iout.append(n)
            aout.append(area[n])
            spatdist[:, IDorder[n]] = np.inf
            cat = [xout, yout, zout, aout, iout]
            coord = [z, y, x]
            xout, yout, zout, aout, iout, spatdist, iin = \
                itersrc_mat(cat, coord, spatdist, area, tol_spat, tol_spec, n, iin, n, IDorder)

    xout = np.array(xout, dtype=int)
    yout = np.array(yout, dtype=int)
    zout = np.array(zout, dtype=int)
    iout = np.array(iout, dtype=int)
    aout = np.array(aout, dtype=int)

    # ID of Spatiale Merging
    xout2 = []
    yout2 = []
    zout2 = []
    aout2 = []
    iout2 = []

    for n, id_cu in enumerate(np.unique(iout)):
        area_in_ID = aout[iout == id_cu]
        area_cu = area_in_ID.max()
        for id_c in np.where(iout == id_cu)[0]:
            xout2.append(xout[id_c])
            yout2.append(yout[id_c])
            zout2.append(zout[id_c])
            iout2.append(n)
            aout2.append(area_cu)

    xout = np.array(xout2, dtype=int)
    yout = np.array(yout2, dtype=int)
    zout = np.array(zout2, dtype=int)
    iout = np.array(iout2, dtype=int)
    aout = np.array(aout2, dtype=int)

    # Group Spectrale Merging
    for n, area_cu in enumerate(np.unique(aout)):
        if area_cu > 0:
            ind = np.where(aout == area_cu)[0]
            # take all the group inside the area
            group_dep = np.unique(iout[ind])
            for cu in group_dep:
                group = np.unique(iout[ind])
                if len(group) == 1:  # if there is only one group remaining
                    break
                if cu in group:
                    for otg in group:
                        if otg != cu:
                            zin = zout[iout == cu]
                            zot = zout[iout == otg]
                            difz = zin[np.newaxis, :].T - zot[np.newaxis, :]
                            if np.sqrt(difz**2).min() < tol_spec:
                                iout[iout == otg] = cu

    return xout, yout, zout, aout, iout, iout2


def itersrc(cat, coord, area, tol_spat, tol_spec, n, iin, id_cu, IDorder):
    """recursive function to perform the spatial merging.
    if neighborhood are close spatially to a lines: they are merged,
    then the neighbor of the seed is analysed if they are enough close to
    the current line (a neighbor of the original seed) they are merged
    only if the frequency is enough close (surrogate) if the frequency is
    different it is rejected.
    If two line (or a group of lines and a new line) are:
        Enough close without a big spectral gap
        not in the same label (a group in background close to one source
        inside a source label)
    the resulting ID is the ID of the source label and not the background


    Parameters
    ----------
    cat     : kinda of catalog of the previously merged lines
              xout,yout,zout,aout,iout:
              the 3D position, area label and ID for all analysed lines
    coord   : the 3D position of the analysed line which become the current
              seed
    area    : array
              list of area

    tol_spat : int
               spatiale tolerance for the spatial merging

    tol_spec : int
               spectrale tolerance for the spectral merging
    n : int
        index of the original seed
    iin : 0-1
          index of (not) processed line
    id_cu : ID of the original seed
    IDorder :   list in which the ID are processed,
                *** maybe to improve ***
                can be by the max max loc correl
                can be by the closest distance
    Returns
    -------
    xout,yout,zout : array
                     the 3D position of the estimated lines
                     the same as z,y,x, they are not changed

    aout : array
           the index of the label in map_in
    iout : array
           the ID after spatial and spatio spectral merging
    spatdist : array
               the spatial distance of the current line with all others
    iin : 0-1
          index of (not) processed line


    Date  : October, 25 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    xout, yout, zout, aout, iout = cat
    z, y, x = coord
    spatdist = np.sqrt((x[n] - x)**2 + (y[n] - y)**2)
    spatdist[iin == 0] = np.inf

    cu_spat = np.sqrt((x[id_cu] - x)**2 + (y[id_cu] - y)**2)
    cu_spat[iin == 0] = np.inf

    ind = np.where(spatdist < tol_spat)[0]
    if len(ind) > 0:
        for indn in ind:
            if iin[indn] > 0:

                if cu_spat[indn] > tol_spat * np.sqrt(2):
                    # check spectral content
                    dz = np.sqrt((z[indn] - z[id_cu])**2)
                    if dz < tol_spec:
                        xout.append(x[indn])
                        yout.append(y[indn])
                        zout.append(z[indn])
                        aout.append(area[indn])

                        iout.append(id_cu)
                        iin[indn] = 0
                        cat = [xout, yout, zout, aout, iout]
                        coord = [z, y, x]
                        xout, yout, zout, aout, iout, spatdist, iin = \
                            itersrc(cat, coord, area, tol_spat,
                                    tol_spec, indn, iin, id_cu, IDorder)

                else:
                    xout.append(x[indn])
                    yout.append(y[indn])
                    zout.append(z[indn])
                    aout.append(area[indn])

                    iout.append(id_cu)
                    iin[indn] = 0
                    cat = [xout, yout, zout, aout, iout]
                    coord = [z, y, x]
                    xout, yout, zout, aout, iout, spatdist, iin = \
                        itersrc(cat, coord, area, tol_spat,
                                tol_spec, indn, iin, id_cu, IDorder)

    return xout, yout, zout, aout, iout, spatdist, iin


def spatiospectral_merging(z, y, x, map_in, tol_spat, tol_spec):
    """perform the spatial and spatio spectral merging.
    The spectral merging give the same ID if several group of lines (from
    spatiale merging) if they share at least one line frequency

    Parameters
    ----------
    z,y,x     : array
                the 3D position of the estimated lines
    map_in    : array
                Segmentation map

    tol_spat : int
               spatiale tolerance for the spatial merging

    tol_spec : int
               spectrale tolerance for the spectral merging

    Returns
    -------
    xout,yout,zout : array
                     the 3D position of the estimated lines
                     the same as z,y,x, they are not changed

    aout : array
           the index of the label in map_in
    iout : array
           the ID after spatial and spatio spectral merging
    iout2 : array
           the ID after spatial merging


    Date  : October, 25 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    Nz = len(z)
    IDorder = np.arange(Nz)
    area = map_in[y, x]

    # Spatiale Merging
    xout = []
    yout = []
    zout = []
    iout = []
    aout = []

    iin = np.ones(IDorder.shape)
    for n in IDorder:
        if iin[n] == 1:

            iin[n] = 0
            xout.append(x[n])
            yout.append(y[n])
            zout.append(z[n])
            iout.append(n)
            aout.append(area[n])
            cat = [xout, yout, zout, aout, iout]
            coord = [z, y, x]
            xout, yout, zout, aout, iout, spatdist, iin = \
                itersrc(cat, coord, area, tol_spat, tol_spec, n, iin, n, IDorder)

    xout = np.array(xout, dtype=int)
    yout = np.array(yout, dtype=int)
    zout = np.array(zout, dtype=int)
    iout = np.array(iout, dtype=int)
    aout = np.array(aout, dtype=int)

    # ID of Spatiale Merging
    xout2 = []
    yout2 = []
    zout2 = []
    aout2 = []
    iout2 = []

    for n, id_cu in enumerate(np.unique(iout)):
        area_in_ID = aout[iout == id_cu]
        area_cu = area_in_ID.max()
        for id_c in np.where(iout == id_cu)[0]:
            xout2.append(xout[id_c])
            yout2.append(yout[id_c])
            zout2.append(zout[id_c])
            iout2.append(n)
            aout2.append(area_cu)

    xout = np.array(xout2, dtype=int)
    yout = np.array(yout2, dtype=int)
    zout = np.array(zout2, dtype=int)
    iout = np.array(iout2, dtype=int)
    aout = np.array(aout2, dtype=int)

    # Group Spectrale Merging
    for n, area_cu in enumerate(np.unique(aout)):
        if area_cu > 0:
            ind = np.where(aout == area_cu)[0]
            # take all the group inside the area
            group_dep = np.unique(iout[ind])
            for cu in group_dep:
                group = np.unique(iout[ind])
                if len(group) == 1:  # if there is only one group remaining
                    break
                if cu in group:
                    for otg in group:
                        if otg != cu:
                            zin = zout[iout == cu]
                            zot = zout[iout == otg]
                            difz = zin[np.newaxis, :].T - zot[np.newaxis, :]
                            if np.sqrt(difz**2).min() < tol_spec:
                                iout[iout == otg] = cu

    return xout, yout, zout, aout, iout, iout2
    # LPI iout2 pour debbugger


def Thresh_Max_Min_Loc_filtering(MaxLoc, MinLoc, thresh, spat_size, spect_size,
                                 filter_act, both=True):
    """Filter the correl>thresh in + DATA by the correl>thresh in - DATA
    if both = True do the same in opposite

    if a line is detected at the z0,y0,x0 in the - data correlation for a
    threshold, the + data correl are cleaned from this line and vice versa

    Parameters
    ----------
    MaxLoc : array
           cube of local maxima from maximum correlation
    MinLoc : array
           cube of local maxima from minus minimum correlation
    thresh : float
             a threshold value
    spat_size : int
                spatiale size of the spatiale filter
    spect_size : int
                 spectral lenght of the spectral filter
    filter_act : Bool
                 activate or deactivate the spatio spectral filter
                 default: True
    both : Bool
           if true the process is applied in both sense, otherwise it s applied
           only in detection purpose and not to compute the purity

    Returns
    -------
    zM,yM,xM : list of tuple of int
               The spatio spectral position of the lines in the + data correl

    zM,yM,xM : (optional) list of tuple of int
               The spatio spectral position of the lines in the - data correl
    Date  : October, 25 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """

    locM = (MaxLoc > thresh)
    locm = (MinLoc > thresh)

    if filter_act:
        spat_rad = int(spat_size / 2)
        spect_rad = int(spect_size / 2)

        LM = locM.copy()
        if both:
            Lm = locm.copy()

        zm, ym, xm = np.where(locm)
        for x, y, z in zip(xm, ym, zm):
            _mask_circle_region(LM, x, y, z, spat_rad, spect_rad, MaxLoc, MinLoc)

        if both:
            zm, ym, xm = np.where(locM)
            for x, y, z in zip(xm, ym, zm):
                _mask_circle_region(Lm, x, y, z, spat_rad, spect_rad, MinLoc, MaxLoc)

        zM, yM, xM = np.where(LM > 0)
        if both:
            zm, ym, xm = np.where(Lm > 0)
    else:
        zM, yM, xM = np.where(locM > 0)
        if both:
            zm, ym, xm = np.where(locm > 0)
    if both:
        return zM, yM, xM, zm, ym, xm
    else:
        return zM, yM, xM


def purity_iter(locM, locm, thresh, spat_size, spect_size, map_in, tol_spat,
                tol_spec, filter_act, bkgrd):
    """Compute the purity values corresponding to a threshold

    Parameters
    ----------
    locM : array
           cube of local maxima from maximum correlation
    locm : array
           cube of local maxima from minus minimum correlation
    thresh : float
             a threshold value

    spat_size : int
                spatiale size of the spatiale filter
    spect_size : int
                 spectral lenght of the spectral filter
    map_in  : array
              labels of source segmentation basedd on continuum
    tol_spat : int
               spatiale tolerance for the spatial merging

    tol_spec : int
               spectrale tolerance for the spectral merging
    filter_act : Bool
                 activate or deactivate the spatio spectral filter
                 default: True
    Returns
    -------
    est_purity : float
                 The estimated purity for this threshold
    det_m     : float
                Number of unique ID (-DATA)
    det_M     : float
                Number of unique ID (+DATA)

    Date  : October, 25 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """

    zM, yM, xM, zm, ym, xm = Thresh_Max_Min_Loc_filtering(
        locM, locm, thresh, spat_size, spect_size, filter_act)
    if len(zM) > 1000:
        xoutM, youtM, zoutM, aoutM, iout1M, iout2M = spatiospectral_merging(
            zM, yM, xM, map_in, tol_spat, tol_spec)
    else:
        xoutM, youtM, zoutM, aoutM, iout1M, iout2M = spatiospectral_merging_mat(
            zM, yM, xM, map_in, tol_spat, tol_spec)
    if len(zm) > 1000:
        xoutm, youtm, zoutm, aoutm, iout1m, iout2m = spatiospectral_merging(
            zm, ym, xm, map_in, tol_spat, tol_spec)
    else:
        xoutm, youtm, zoutm, aoutm, iout1m, iout2m = spatiospectral_merging_mat(
            zm, ym, xm, map_in, tol_spat, tol_spec)
    if bkgrd:
        # purity computed on the background (aout==0)
        det_m, det_M = len(np.unique(iout1m[aoutm == 0])), len(np.unique(iout1M[aoutM == 0]))
        if len(np.unique(iout1M[aoutM == 0])) > 0:
            est_purity = 1 - det_m / det_M
        else:
            est_purity = 0
    else:
        det_m, det_M = len(np.unique(iout1m)), len(np.unique(iout1M))
        if len(np.unique(iout1M)) > 0:
            est_purity = 1 - det_m / det_M
        else:
            est_purity = 0

    return est_purity, det_m, det_M


@timeit
def Compute_threshold_purity(purity, cube_local_max, cube_local_min,
                             segmap, spat_size, spect_size,
                             tol_spat, tol_spec, filter_act, bkgrd,
                             auto=(5, 15, 0.1), threshlist=None):
    """Compute threshold values corresponding to a given purity

    Parameters
    ----------
    purity    : float
                the purity between 0 and 1
    cube_Local_max : array
                     cube of local maxima from maximum correlation
    cube_Local_min : array
                     cube of local maxima from minus minimum correlation
    segmap: array
            segmentation map
    spat_size : int
                spatiale size of the spatiale filter
    spect_size : int
                 spectral lenght of the spectral filter
    tol_spat : int
               spatiale tolerance for the spatial merging

    tol_spec : int
               spectrale tolerance for the spectral merging

    filter_act : Bool
                 activate or deactivate the spatio spectral filter
                 default: True
    auto       : tuple (npts1,npts2,pmargin)
                 nb of threshold sample for iteration 1 and 2, margin in purity
                 default (5,15,0.1)
    threshlist : list
                 list of thresholds to compute the purity
                 default None

    Returns
    -------
    threshold : float
                the threshold associated to the purity
    PVal_r : array
             The purity function
    index_pval: array
                index value to plot
    det_m     : array
                Number of detections (-DATA)
    det_M     : array
                Number of detections (+DATA)

    Date  : July, 6 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """

    logger = logging.getLogger(__name__)
    # initialization
    det_m = []
    det_M = []
    Pval_r = []
    Tval_r = []

    if threshlist is None:
        npts1, npts2, dp = auto
        thresh_max = np.minimum(cube_local_min.max(), cube_local_max.max())
        thresh_min = np.median(np.amax(cube_local_max, axis=0)) * 1.1

        # first exploration
        index_pval1 = np.exp(np.linspace(np.log(thresh_min),
                                         np.log(thresh_max), npts1))
        # make sure that last point is thresh_max (and not an
        # approximate value due to linspace)
        index_pval1[-1] = thresh_max
        n_pval1 = len(index_pval1)

        logger.debug('Iter 1 Threshold min %f max %f npts %d',
                     thresh_min, thresh_max, n_pval1)
        for k, thresh in enumerate(ProgressBar(index_pval1[::-1])):
            est_purity, det_mit, det_Mit = purity_iter(cube_local_max,
                                                       cube_local_min,
                                                       thresh, spat_size,
                                                       spect_size, segmap,
                                                       tol_spat, tol_spec,
                                                       filter_act, bkgrd)
            Tval_r.append(thresh)
            Pval_r.append(est_purity)
            det_m.append(det_mit)
            det_M.append(det_Mit)
            if est_purity == 1:
                thresh_max = thresh
            if est_purity < purity - dp:
                break
        thresh_min = thresh

        for k, (thresh, det_mit, det_Mit, est_purity) in enumerate(
                zip(Tval_r, det_m, det_M, Pval_r)):
            logger.debug('   %d/%d Threshold %f -data %d +data %d purity %f',
                         k + 1, n_pval1, thresh, det_mit, det_Mit, est_purity)

        # 2nd iter
        index_pval3 = np.exp(np.linspace(np.log(thresh_min),
                                         np.log(thresh_max), npts2))
        # make sure that last point is thresh_max (and not an
        # approximate value due to linspace)
        index_pval3[-1] = thresh_max

        logger.debug('Iter 2 Threshold min %f max %f npts %d',
                     index_pval3[0], index_pval3[-1], len(index_pval3))
        for k, thresh in enumerate(ProgressBar(index_pval3)):
            if np.any(np.isclose(thresh, Tval_r)):
                continue
            est_purity, det_mit, det_Mit = purity_iter(cube_local_max,
                                                       cube_local_min,
                                                       thresh, spat_size,
                                                       spect_size, segmap,
                                                       tol_spat, tol_spec,
                                                       filter_act, bkgrd)
            logger.debug('    %d/%d Threshold %f -data %d +data %d purity %f',
                         k + 1, len(index_pval3), thresh, det_mit, det_Mit,
                         est_purity)
            Tval_r.append(thresh)
            Pval_r.append(est_purity)
            det_m.append(det_mit)
            det_M.append(det_Mit)
            if est_purity > purity + dp:
                break
        Tval_r = np.asarray(Tval_r)
        ksort = Tval_r.argsort()
        Pval_r = np.asarray(Pval_r)[ksort]
        det_m = np.asarray(det_m)[ksort]
        det_M = np.asarray(det_M)[ksort]
        Tval_r = Tval_r[ksort]
    else:
        for k, thresh in enumerate(ProgressBar(threshlist)):
            est_purity, det_mit, det_Mit = purity_iter(cube_local_max,
                                                       cube_local_min,
                                                       thresh, spat_size,
                                                       spect_size, segmap,
                                                       tol_spat, tol_spec,
                                                       filter_act, bkgrd)
            logger.debug('%d/%d Threshold %f -data %d +data %d purity %f',
                         k + 1, len(threshlist), thresh, det_mit, det_Mit,
                         est_purity)
            Pval_r.append(est_purity)
            det_m.append(det_mit)
            det_M.append(det_Mit)
        Tval_r = np.asarray(threshlist)
        Pval_r = np.asarray(Pval_r)
        det_m = np.asanyarray(det_m)
        det_M = np.asanyarray(det_M)

    if Pval_r[-1] < purity:
        logger.warning('Maximum computed purity %.2f is below %.2f',
                       Pval_r[-1], purity)
        threshold = np.inf
    else:
        threshold = np.interp(purity, Pval_r, Tval_r)
        detect = np.interp(threshold, Tval_r, det_M)
        logger.debug('Interpolated Threshold %.3f Detection %d for Purity %.2f',
                     threshold, detect, purity)

    return threshold, Pval_r, Tval_r, det_m, det_M


@timeit
def Create_local_max_cat(thresh, cube_local_max, cube_local_min,
                         segmentation_map, spat_size, spect_size,
                         tol_spat, tol_spec, filter_act, profile, wcs, wave):
    """ Function which extract detection and performs  spatio spectral merging
    at same time for a given purity and segmentation map

    Parameters
    ----------
    thresh    : float
                the threshold for correl
    cube_local_max : array
                     cube of local maxima from maximum correlation
    cube_local_min : array
                     cube of local maxima from minus minimum correlation
    segmentation_map : array
                        map of estimated continuum for segmentation

    spat_size : int
                spatiale size of the spatiale filter
    spect_size : int
                 spectral lenght of the spectral filter
    tol_spat : int
               spatiale tolerance for the spatial merging

    tol_spec : int
               spectrale tolerance for the spectral merging
    filter_act : Bool
                 activate or deactivate the spatio spectral filter
                 default: True

    Returns
    -------
    Cat_ref : astropy.Table
              Catalogue of the referent voxels coordinates for each group
              Columns of Cat_ref : ID ra dec lba x0 y0 z0 profile seglabel T_GLR

    Date  : June, 19 2017
    Author: Antony Schutz(antonyschutz@gmail.com)
    """
    logger = logging.getLogger(__name__)

    logger.info('Thresholding...')
    zM, yM, xM, zm, ym, xm = Thresh_Max_Min_Loc_filtering(
        cube_local_max, cube_local_min, thresh, spat_size, spect_size, filter_act)
    logger.info('Spatio-spectral merging...')
    if len(zM) > 1000:
        xpixRef, ypixRef, zpixRef, seg_label, idout, iout2M = spatiospectral_merging(
            zM, yM, xM, segmentation_map, tol_spat, tol_spec)
    else:
        xpixRef, ypixRef, zpixRef, seg_label, idout, iout2M = spatiospectral_merging_mat(
            zM, yM, xM, segmentation_map, tol_spat, tol_spec)

    correl_max = cube_local_max[zpixRef, ypixRef, xpixRef]
    profile_max = profile[zpixRef, ypixRef, xpixRef]

    # add real coordinates
    dec, ra = wcs.pix2sky(np.stack((ypixRef, xpixRef)).T).T
    lbda = wave.coord(zpixRef)

    # Relabel IDs sequentially
    idout = np.asarray(idout)
    oldIDs = np.unique(idout)
    idmap = np.zeros(oldIDs.max() + 1, dtype=int)
    idmap[oldIDs] = np.arange(len(oldIDs))

    # Catalogue of referent pixels
    Cat_ref = Table([idmap[idout], ra, dec, lbda, xpixRef, ypixRef, zpixRef,
                     profile_max, seg_label, correl_max],
                    names=('ID', 'ra', 'dec', 'lbda', 'x0', 'y0', 'z0',
                           'profile', 'seg_label', 'T_GLR'))
    Cat_ref.sort('ID')
    return Cat_ref, (zm, ym, xm)


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
    nl, ny, nx = raw_in.shape

    # size psf
    if weights_in is None:
        sizpsf = psf_in.shape[1]
    else:
        sizpsf = psf_in[0].shape[1]

    # size minicube
    sizemc = 2 * size_grid + sizpsf

    # half size psf
    longxy = int(sizemc // 2)

    # bound of image
    psx1 = np.maximum(0, x - longxy)
    psy1 = np.maximum(0, y - longxy)
    psx2 = np.minimum(nx, x + longxy + 1)
    psy2 = np.minimum(ny, y + longxy + 1)

    # take into account bordure of cube
    psx12 = np.maximum(0, longxy - x + psx1)
    psy12 = np.maximum(0, longxy - y + psy1)
    psx22 = np.minimum(sizemc, longxy - x + psx2)
    psy22 = np.minimum(sizemc, longxy - y + psy2)

    # create weight, data with bordure
    red_dat = np.zeros((nl, sizemc, sizemc))
    red_dat[:, psy12:psy22, psx12:psx22] = raw_in[:, psy1:psy2, psx1:psx2]

    red_var = np.ones((nl, sizemc, sizemc)) * np.inf
    red_var[:, psy12:psy22, psx12:psx22] = var_in[:, psy1:psy2, psx1:psx2]

    if weights_in is None:
        red_wgt = None
        red_psf = psf_in
    else:
        red_wgt = []
        red_psf = []
        for n, w in enumerate(weights_in):
            if np.sum(w[psy1:psy2, psx1:psx2]) > 0:
                w_tmp = np.zeros((sizemc, sizemc))
                w_tmp[psy12:psy22, psx12:psx22] = w[psy1:psy2, psx1:psx2]
                red_wgt.append(w_tmp)
                red_psf.append(psf_in[n])

    return red_dat, red_var, red_wgt, red_psf


def LS_deconv_wgt(data_in, var_in, psf_in):
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
    nl, sizpsf, tmp = psf_in.shape
    v = np.reshape(var_in, (nl, sizpsf * sizpsf))
    p = np.reshape(psf_in, (nl, sizpsf * sizpsf))
    s = np.reshape(data_in, (nl, sizpsf * sizpsf))
    varest_out = 1 / np.sum(p * p / v, axis=1)
    deconv_out = np.sum(p * s / np.sqrt(v), axis=1) * varest_out

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
    cube_conv = cube_conv * (np.abs(psf_in) > 0)
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

    nl, sizpsf, tmp = psf_in.shape

    # STD
    data_std = data_in / np.sqrt(var_in)
    data_st_pca = np.reshape(data_std, (nl, sizpsf * sizpsf))

    # PCA
    mean_in_pca = np.mean(data_st_pca, axis=1)
    data_in_pca = data_st_pca - np.repeat(mean_in_pca[:, np.newaxis],
                                          sizpsf * sizpsf, axis=1)

    U, s, V = svds(data_in_pca, k=1)

    # orthogonal projection
    xest = orthogonal_projection(U, data_in_pca)
    residual = data_std - np.reshape(xest, (nl, sizpsf, sizpsf))

    # LS deconv
    deconv_out, varest_out = LS_deconv_wgt(residual, var_in, psf_in)

    # PSF convolution
    conv_out = conv_wgt(deconv_out, psf_in)

    # cleaning the data
    data_clean = (data_in - conv_out) / np.sqrt(var_in)

    # 2nd PCA
    data_in_pca = np.reshape(data_clean, (nl, sizpsf * sizpsf))
    mean_in_pca = np.mean(data_in_pca, axis=1)
    data_in_pca -= np.repeat(mean_in_pca[:, np.newaxis], sizpsf * sizpsf, axis=1)

    U, s, V = svds(data_in_pca, k=1)

    if order_dct is not None:
        # denoise eigen vector with DCT
        D0 = DCTMAT(nl, order_dct)
        U = orthogonal_projection(D0, U)

    # orthogonal projection
    xest = orthogonal_projection(U, data_st_pca)
    cont = np.reshape(xest, (nl, sizpsf, sizpsf))
    residual = data_std - cont

    # LS deconvolution of the line
    estimated_line, estimated_var = LS_deconv_wgt(residual, var_in, psf_in)

    # PSF convolution of estimated line
    conv_out = conv_wgt(estimated_line, psf_in)

    return estimated_line, estimated_var


def GridAnalysis(data_in, var_in, psf, weight_in, horiz,
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

    shape = (1 + 2 * size_grid, 1 + 2 * size_grid)
    zest = np.zeros(shape)
    fest_00 = np.zeros(shape)
    fest_05 = np.zeros(shape)
    mse = np.full(shape, np.inf)
    mse_5 = np.full(shape, np.inf)

    nl = data_in.shape[0]
    ind_max = slice(np.maximum(0, z0 - 5), np.minimum(nl, z0 + 5))
    if weight_in is None:
        nl, sizpsf, tmp = psf.shape
    else:
        nl, sizpsf, tmp = psf[0].shape

    lin_est = np.zeros((nl, ) + shape)
    var_est = np.zeros((nl, ) + shape)
    # half size psf
    longxy = int(sizpsf // 2)
    inds = slice(longxy - horiz_psf, longxy + 1 + horiz_psf)
    for dx in range(0, 1 + 2 * size_grid):
        if (x0 - size_grid + dx >= 0) and (x0 - size_grid + dx < NX):
            for dy in range(0, 1 + 2 * size_grid):
                if (y0 - size_grid + dy >= 0) and (y0 - size_grid + dy < NY):

                    # extract data
                    r1 = data_in[:, dy:sizpsf + dy, dx:sizpsf + dx]
                    var = var_in[:, dy:sizpsf + dy, dx:sizpsf + dx]
                    if weight_in is not None:
                        wgt = np.array(weight_in)[:, dy:sizpsf + dy, dx:sizpsf + dx]
                        psf = np.sum(np.repeat(wgt[:, np.newaxis, :, :], nl,
                                               axis=1) * psf, axis=0)

                    # estimate Full Line and theoretic variance
                    deconv_met, varest_met = method_PCA_wgt(r1, var, psf,
                                                            order_dct)

                    z_est = peakdet(deconv_met[ind_max], 3)
                    if z_est == 0:
                        break

                    maxz = z0 - 5 + z_est
                    zest[dy, dx] = maxz
                    ind_z5 = np.arange(max(0, maxz - 5), min(maxz + 5, nl))
                    # ind_z10 = np.arange(maxz-10,maxz+10)
                    ind_hrz = slice(maxz - horiz, maxz + horiz)

                    lin_est[:, dy, dx] = deconv_met
                    var_est[:, dy, dx] = varest_met

                    # compute MSE
                    LC = conv_wgt(deconv_met[ind_hrz], psf[ind_hrz, :, :])
                    LCred = LC[:, inds, inds]
                    r1red = r1[ind_hrz, inds, inds]
                    mse[dy, dx] = np.sum((r1red - LCred)**2) / np.sum(r1red**2)

                    LC = conv_wgt(deconv_met[ind_z5], psf[ind_z5, :, :])
                    LCred = LC[:, inds, inds]
                    r1red = r1[ind_z5, inds, inds]
                    mse_5[dy, dx] = np.sum((r1red - LCred)**2) / np.sum(r1red**2)

                    # compute flux
                    fest_00[dy, dx] = np.sum(deconv_met[ind_hrz])
                    fest_05[dy, dx] = np.sum(deconv_met[ind_z5])
                    # fest_10[dy,dx] = np.sum(deconv_met[ind_z10])

    if criteria == 'flux':
        wy, wx = np.where(fest_00 == fest_00.max())
    elif criteria == 'mse':
        wy, wx = np.where(mse == mse.min())
    else:
        raise IOError('Bad criteria: (flux) or (mse)')
    y = y0 - size_grid + wy
    x = x0 - size_grid + wx
    z = zest[wy, wx]

    flux_est_5 = float(fest_05[wy, wx])
    # flux_est_10 = float( fest_10[wy,wx] )
    MSE_5 = float(mse_5[wy, wx])
    # MSE_10 = float( mse_10[wy,wx] )
    estimated_line = lin_est[:, wy, wx]
    estimated_variance = var_est[:, wy, wx]

    return flux_est_5, MSE_5, estimated_line, \
        estimated_variance, int(y), int(x), int(z)


def peakdet(v, delta):

    v = np.array(v)
    nv = len(v)
    mv = np.zeros(nv + 2 * delta)
    mv[:delta] = np.Inf
    mv[delta:-delta] = v
    mv[-delta:] = np.Inf
    ind = []

    # find all local maxima
    ind = [n - delta for n in range(delta, nv + delta)
           if mv[n] > mv[n - 1] and mv[n] > mv[n + 1]]

    # take the maximum and closest from original estimation
    indi = np.array(ind, dtype=int)

    sol = int(nv / 2)
    if len(indi) > 0:
        # methode : closest from initial estimate
        out = indi[np.argmin((indi - sol)**2)]
    else:
        out = sol
    return out


@timeit
def Estimation_Line(Cat1_T, RAW, VAR, PSF, WGT, wcs, wave, size_grid=1,
                    criteria='flux', order_dct=30, horiz_psf=1,
                    horiz=5):
    """Function to compute the estimated emission line and the optimal
    coordinates for each detected lines in a spatio-spectral grid.

    Parameters
    ----------
    Cat1_T     : astropy.Table
                 Catalogue of parameters of detected emission lines selected
                 with a narrow band test.
                 Columns of the Catalogue Cat1_T:
                 ra dec lbda x0 y0 z0 T_GLR profile
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
                       ra dec lbda x0 x1 y0 y1 z0 z1 T_GLR profile residual
                       flux num_line
    Cat_est_line_raw : list of arrays
                       Estimated lines in data space
    Cat_est_line_std : list of arrays
                       Estimated lines in SNR space

    Date  : June, 21 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """

    # TODO: When computing the optimal position of the lines, we may end up
    # with duplicated lines at the very same (x, y, z) position because we
    # manage to correct for double, very near, detections.  We should then keep
    # only one line.  It's better to keep the “purest” line, but the purity
    # information is not available at this stage.

    # Initialization
    NL, NY, NX = RAW.shape
    Cat2_x_grid = []
    Cat2_y_grid = []
    Cat2_z_grid = []
    Cat2_res_min5 = []
    Cat2_flux5 = []
    Cat_est_line_raw = []
    Cat_est_line_var = []

    for src in ProgressBar(Cat1_T):
        y0 = src['y0']
        x0 = src['x0']
        z0 = src['z0']
        red_dat, red_var, red_wgt, red_psf = extract_grid(RAW, VAR, PSF, WGT,
                                                          y0, x0, size_grid)

        f5, m5, lin_est, var_est, y, x, z = GridAnalysis(
            red_dat, red_var, red_psf, red_wgt, horiz,
            size_grid, y0, x0, z0, NY, NX, horiz_psf, criteria, order_dct
        )

        Cat2_x_grid.append(x)
        Cat2_y_grid.append(y)
        Cat2_z_grid.append(z)
        Cat2_res_min5.append(m5)
        Cat2_flux5.append(f5)
        Cat_est_line_raw.append(lin_est.ravel())
        Cat_est_line_var.append(var_est.ravel())

    Cat2 = Cat1_T.copy()

    # add real coordinates
    dec, ra = wcs.pix2sky(np.stack((Cat2_y_grid, Cat2_x_grid)).T).T
    Cat2['ra'] = ra
    Cat2['dec'] = dec
    Cat2['lbda'] = wave.coord(Cat2_z_grid)

    col_flux = Column(name='flux', data=Cat2_flux5)
    col_res = Column(name='residual', data=Cat2_res_min5)
    col_num = Column(name='num_line', data=np.arange(len(Cat2)))
    col_x = Column(name='x', data=Cat2_x_grid)
    col_y = Column(name='y', data=Cat2_y_grid)
    col_z = Column(name='z', data=Cat2_z_grid)

    Cat2.add_columns([col_x, col_y, col_z, col_res, col_flux, col_num],
                     indexes=[4, 5, 6, 8, 8, 8])

    return Cat2, Cat_est_line_raw, Cat_est_line_var


def Purity_Estimation(Cat_in, purity_curves, purity_index):
    """Function to compute the estimated purity for each line.

    Parameters
    ----------
    Cat_in     : astropy.Table
                 Catalogue of parameters of detected emission lines selected
                 with a narrow band test.
    purity_curves     : array, array
                          purity curves related to area
    purity_index      : array, array
                          index of purity curves related to area

    Returns
    -------
    Cat1_2            : astropy.Table
                       Catalogue of parameters of detected emission lines.
                       Columns of the Catalogue Cat2:
                       ra dec lbda x0 x1 y0 y1 z0 z1 T_GLR profile residual
                       flux num_line purity


    Date  : July, 25 2017
    Author: Antony Schutz (antony.schutz@gmail.com)
    """

    Cat1_2 = Cat_in.copy()
    # set to 0 if only 1 purity meaurement
    purity = np.zeros(len(Cat1_2))

    # Comp=0
    ksel = Cat1_2['comp'] == 0
    if np.count_nonzero(ksel) > 1:
        tglr = Cat1_2['T_GLR'][ksel]
        f = interp1d(purity_index[0], purity_curves[0], bounds_error=False,
                     fill_value="extrapolate")
        purity[ksel] = f(tglr.data.data)

    # comp=1
    ksel = Cat1_2['comp'] == 1
    if np.count_nonzero(ksel) > 1:
        tglr = Cat1_2['STD'][ksel]
        f = interp1d(purity_index[1], purity_curves[1], bounds_error=False,
                     fill_value="extrapolate")
        purity[ksel] = f(tglr.data.data)

    # The purity by definition cannot be > 1 and < 0, if the interpolation
    # gives a value outside these limits, replace by 1 or 0
    Cat1_2['purity'] = np.clip(purity, 0, 1)

    return Cat1_2


def estimate_spectrum(nb_lines, wave_pix, num_profil, fwhm_profiles,
                      Cat_est_line_data, Cat_est_line_var, corr_line):
    """
    """
    if nb_lines == 1:
        return Cat_est_line_data[0, :], Cat_est_line_var[0, :], corr_line[0, :]
    else:
        nz = Cat_est_line_data[0].shape[0]
        FWHM = np.asarray([fwhm_profiles[i] for i in num_profil], dtype=np.int)
        min_pix = wave_pix - FWHM
        max_pix = wave_pix + FWHM + 1
        d = -np.minimum(0, min_pix[1:] - max_pix[:-1])
        min_pix[0] = 0
        min_pix[1:] += d // 2
        max_pix[:-1] -= (d - d // 2)
        max_pix[-1] = nz
        coeff = np.arange(min_pix[1] - max_pix[0]) / (min_pix[1] - max_pix[0])
        spe = np.zeros(nz)
        var = np.zeros(nz)
        corr = np.zeros(nz)
        for j in range(nb_lines):

            # flux coefficient
            cz = np.zeros(nz)
            cz[min_pix[j]:max_pix[j]] = 1
            if j > 0:
                cz[max_pix[j - 1]:min_pix[j]] = coeff
            if j < (nb_lines - 1):
                coeff = np.arange(min_pix[j + 1] - max_pix[j]) / (min_pix[j + 1] - max_pix[j])
                cz[max_pix[j]:min_pix[j + 1]] = coeff[::-1]

            spe += cz * Cat_est_line_data[j, :]
            var += cz**2 * Cat_est_line_var[j, :]
            corr += cz * corr_line[j, :]

        return spe, var, corr


def Construct_Object(k, ktot, cols, units, desc, fmt, step_wave,
                     origin, filename, maxmap, segmap, correl, fwhm_profiles,
                     param, path, name, i, ra, dec, x_centroid,
                     y_centroid, seg_label, wave_pix, GLR, num_profil,
                     nb_lines, Cat_est_line_data, Cat_est_line_var,
                     y, x, flux, purity, comp, src_vers, author):
    """Function to create the final source

    Parameters
    ----------
    """
    logger = logging.getLogger(__name__)
    logger.debug('{}/{} source ID {}'.format(k + 1, ktot, i))
    cube = Cube(filename)
    cubevers = cube.primary_header.get('CUBE_V', '')
    origin.append(cubevers)

    if type(maxmap) is str:
        maxmap_ = Image(maxmap)
    else:
        maxmap_ = maxmap

    src = Source.from_data(i, ra, dec, origin)
    src.add_attr('SRC_V', src_vers, desc='Source version')
    src.add_history('Source created with Origin', author)
    src.add_attr('OR_X', x_centroid, desc='x position in pixel',
                 unit=u.pix, fmt='d')
    src.add_attr('OR_Y', y_centroid, desc='y position in pixel',
                 unit=u.pix, fmt='d')
    src.add_attr('OR_SEG', seg_label, desc='label in the segmentation map',
                 fmt='d')
    src.add_attr('OR_V', origin[1], desc='Orig version')
    # param
    if 'profiles' in param.keys():
        src.OR_PROF = (param['profiles'], 'OR input Spectral profiles')
    if 'PSF' in param.keys():
        src.OR_FSF = (param['PSF'], 'OR input FSF cube')
    if 'pfa_areas' in param.keys():
        src.OR_PFAA = (param['pfa_areas'], 'OR input PFA uses to create the area map')
    if 'size_areas' in param.keys():
        src.OR_SIZA = (param['size_areas'], 'OR input Side size in pixels')
    if 'minsize_areas' in param.keys():
        src.OR_MSIZA = (param['minsize_areas'], 'OR input Minimum area size in pixels')
    if 'nbareas' in param.keys():
        src.OR_NA = (param['nbareas'], 'OR Nb of areas')
    if 'expmap' in param.keys():
        src.OR_EXP = (param['expmap'], 'OR input Exposure map')
    if 'dct_order' in param.keys():
        src.OR_DCT = (param['dct_order'], 'OR input DCT order')
    if 'Noise_population' in param.keys():
        src.OR_FBG = (param['Noise_population'], 'OR input Fraction of spectra estimated as background')
    if 'pfa_test' in param.keys():
        src.OR_PFAT = (param['pfa_test'], 'OR input PFA test')
    if 'itermax' in param.keys():
        src.OR_ITMAX = (param['itermax'], 'OR input Maximum number of iterations')
    if 'threshold_list' in param.keys():
        th = param['threshold_list']
        for i, th in enumerate(param['threshold_list']):
            src.header['OR_THL%02d' % i] = ('%0.2f' % th, 'OR input Threshold per area')
    if 'neighbors' in param.keys():
        src.OR_NG = (param['neighbors'], 'OR input Neighbors')
    if 'NbSubcube' in param.keys():
        src.OR_NS = (param['NbSubcube'], 'OR input Nb of subcubes for the spatial segmentation')
    if 'tol_spat' in param.keys():
        src.OR_DXY = (param['tol_spat'], 'spatial tolerance for the spatial merging (distance in pixels)')
    if 'tol_spec' in param.keys():
        src.OR_DZ = (param['tol_spec'], 'spectral tolerance for the spatial merging (distance in pixels)')
    if 'spat_size' in param.keys():
        src.OR_SXY = (param['spat_size'], 'spatiale size of the spatiale filter')
    if 'spect_size' in param.keys():
        src.OR_SZ = (param['spect_size'], 'spectral lenght of the spectral filter')
    if 'grid_dxy' in param.keys():
        src.OR_DXY = (param['grid_dxy'], 'OR input Grid Nxy')
    src.COMP_CAT = (comp[0], 'OR complemantary catalog')
    if comp[0]:
        if 'threshold2' in param.keys():
            src.OR_TH = ('%0.2f' % param['threshold2'], 'OR threshold')
        if 'purity2' in param.keys():
            src.OR_PURI = ('%0.2f' % param['purity2'], 'OR input Purity')
        cols[3] = 'STD'
    else:
        if 'threshold' in param.keys():
            src.OR_TH = ('%0.2f' % param['threshold'], 'OR threshold')
        if 'purity' in param.keys():
            src.OR_PURI = ('%0.2f' % param['purity'], 'OR input Purity')
        cols[3] = 'GLR'

    # WHITE IMAGE
    src.add_white_image(cube)
    # MUSE CUBE
    src.add_cube(cube, 'MUSE_CUBE')
    # MAXMAP
    src.add_image(maxmap_, 'OR_MAXMAP')
    # Segmentation map
    if seg_label > 0:
        if type(segmap) is str:
            segmap_ = Image(segmap)
        else:
            segmap_ = segmap
        src.add_image(segmap_, 'OR_SEG')

    w = cube.wave.coord(wave_pix, unit=u.angstrom)
    names = np.array(['%04d' % w[j] for j in range(nb_lines)])
    if np.unique(names).shape != names.shape:
        names = names.astype(np.int)
        while ((names[1:] - names[:-1]) == 0).any():
            names[1:][(names[:-1] - names[1:]) == 0] += 1
        names = names.astype(np.str)

    if type(correl) is str:
        correl_ = Cube(correl)
    else:
        correl_ = correl
        correl_.mask = cube.mask
    corr_line = []

    # Loop on lines
    for j in range(nb_lines):
        corr_line.append(correl_[:, y[j], x[j]]._data)
        # FWHM in arcsec of the profile
        profile_num = num_profil[j]
        profil_FWHM = step_wave * fwhm_profiles[profile_num]
        # profile_dico = Dico[profile_num]
        fl = flux[j]
        pu = purity[j]
        vals = [w[j], profil_FWHM, fl, GLR[j], profile_num, pu]
        src.add_line(cols, vals, units, desc, fmt)

        src.add_narrow_band_image_lbdaobs(cube,
                                          'NB_LINE_{:s}'.format(names[j]),
                                          w[j], width=2 * profil_FWHM,
                                          is_sum=True, subtract_off=True)
        src.add_narrow_band_image_lbdaobs(correl_,
                                          'OR_CORR_{:s}'.format(names[j]),
                                          w[j], width=2 * profil_FWHM,
                                          is_sum=True, subtract_off=False)

    sp, var, corr = estimate_spectrum(nb_lines, wave_pix, num_profil,
                                      fwhm_profiles, Cat_est_line_data,
                                      Cat_est_line_var, np.asarray(corr_line))
    src.spectra['ORIGIN'] = Spectrum(data=sp, var=var, wave=cube.wave)
    src.spectra['OR_CORR'] = Spectrum(data=corr, wave=cube.wave)
    # TODO Estimated continuum

    # write source
    src.write('%s/%s-%05d.fits' % (path, name, src.ID))


@timeit
def Construct_Object_Catalogue(Cat, Cat_est_line, correl, wave, fwhm_profiles,
                               path_src, name, param, src_vers, author, path,
                               maxmap, segmap, ncpu=1):
    """Function to create the final catalogue of sources with their parameters

    Parameters
    ----------
    Cat              : Catalogue of parameters of detected emission lines:
                       ID x_circle y_circle x_centroid y_centroid nb_lines
                       x y z T_GLR profile residual
                       flux num_line RA DEC
    Cat_est_line     : list of spectra
                       Catalogue of estimated lines
    correl            : array
                        Cube of T_GLR values
    wave              : `mpdaf.obj.WaveCoord`
                        Spectral coordinates
    fwhm_profiles     : array
                        List of fwhm values (in pixels) of the input spectra
                        profiles (DICO).


    Date  : Dec, 16 2015
    Author: Carole Clastre (carole.clastres@univ-lyon1.fr)
    """
    logger = logging.getLogger(__name__)
    uflux = u.erg / (u.s * u.cm**2)
    unone = u.dimensionless_unscaled

    cols = ['LBDA_ORI', 'FWHM_ORI', 'FLUX_ORI', 'GLR', 'PROF', 'PURITY']
    units = [u.Angstrom, u.Angstrom, uflux, unone, unone, unone]
    fmt = ['.2f', '.2f', '.1f', '.1f', 'd', '.2f']
    desc = None

    step_wave = wave.get_step(unit=u.angstrom)
    filename = param['cubename']
    origin = ['ORIGIN', __version__, os.path.basename(filename)]

    path2 = os.path.abspath(path) + '/' + name
    if os.path.isfile('%s/maxmap.fits' % path2):
        f_maxmap = '%s/maxmap.fits' % path2
    else:
        maxmap.write('%s/tmp_maxmap.fits' % path2)
        f_maxmap = '%s/tmp_maxmap.fits' % path2
    if os.path.isfile('%s/segmentation_map.fits' % path2):
        f_segmap = '%s/segmentation_map.fits' % path2
    else:
        segmap.write('%s/tmp_segmap.fits' % path2)
        f_segmap = '%s/tmp_segmap.fits' % path2
    if os.path.isfile('%s/cube_correl.fits' % path2):
        f_correl = '%s/cube_correl.fits' % path2
    else:
        correl.write('%s/tmp_cube_correl.fits' % path2)
        f_correl = '%s/tmp_cube_correl.fits' % path2

    sources_arglist = []
    logger.debug('Creating source list from Cat2 catalog (%d lines)',len(Cat))
    for i in np.unique(Cat['ID']):
        # Source = group
        E = Cat[Cat['ID'] == i]
        # TODO change to compute barycenter using flux
        ra = E['ra'][0]
        dec = E['dec'][0]
        x_centroid = E['x'][0]
        y_centroid = E['y'][0]
        seg_label = E['seg_label'][0]
        # Lines of this group
        E.sort('z')
        wave_pix = E['z'].data
        num_profil = E['profile'].data
        # Number of lines in this group
        nb_lines = len(E)
        Cat_est_line_data = np.empty((nb_lines, wave.shape))
        Cat_est_line_var = np.empty((nb_lines, wave.shape))
        for j in range(nb_lines):
            Cat_est_line_data[j, :] = Cat_est_line[E['num_line'][j]]._data
            Cat_est_line_var[j, :] = Cat_est_line[E['num_line'][j]]._var
        y = E['y']
        x = E['x']
        flux = E['flux']
        purity = E['purity']
        comp = E['comp']
        if comp[0]:
            GLR = E['STD']
        else:
            GLR = E['T_GLR']
        source_arglist = (i, ra, dec, x_centroid, y_centroid, seg_label,
                          wave_pix, GLR, num_profil, nb_lines,
                          Cat_est_line_data, Cat_est_line_var,
                          y, x, flux, purity, comp,
                          src_vers, author)
        sources_arglist.append(source_arglist)

    logger.debug('Creating sources (%d sources)',len(source_arglist))
    if ncpu > 1:
        # run in parallel
        errmsg = Parallel(n_jobs=ncpu, max_nbytes=1e6)(
            delayed(Construct_Object)(k, len(sources_arglist), cols, units,
                                      desc, fmt, step_wave, origin, filename,
                                      f_maxmap, f_segmap, f_correl,
                                      fwhm_profiles, param, path_src, name,
                                      *source_arglist)

            for k, source_arglist in enumerate(sources_arglist)
        )
        # print error messages if any
        for msg in errmsg:
            if msg is None:
                continue
            logger.error(msg)
    else:
        for k, source_arglist in enumerate(sources_arglist):
            msg = Construct_Object(k, len(sources_arglist), cols, units, desc,
                                   fmt, step_wave, origin, filename,
                                   maxmap, segmap, correl, fwhm_profiles,
                                   param, path_src, name,
                                   *source_arglist)
            if msg is not None:
                logger.error(msg)

    if os.path.isfile('%s/tmp_maxmap.fits' % path2):
        os.remove('%s/tmp_maxmap.fits' % path2)
    if os.path.isfile('%s/tmp_segmap.fits' % path2):
        os.remove('%s/tmp_segmap.fits' % path2)
    if os.path.isfile('%s/tmp_cube_correl.fits' % path2):
        os.remove('%s/tmp_cube_correl.fits' % path2)

    return len(np.unique(Cat['ID']))


def unique_sources(table):
    """Return unique source positions in table.

    ORIGIN produces a list of lines associated to various sources identified by
    the ID column.  Some objects contain several lines found at slightly
    different positions.

    This function computes the list of unique sources averaging the RA and Dec
    of each line using the flux as weight.  The resulting table contains:

    - ID: the identifier of the source (unique);
    - ra, dec: the position
    - n_lines: the number of lines associated to the source;
    - seg_label: the label of the segment associated to the source in the
      segmentation map;
    - comp: boolean flag true for complementary sources detected only in the
      cube before the PCA.
    - line_merged_flag: boolean flag indicating if any of the lines associated
      to the source was merged with another nearby line.

    Note: The n_lines contains the number of unique lines associated to the
    source, but for computing the position of the source, we are using all the
    duplicated lines as shredded sources may have identical lines found at
    different positions.

    Parameters
    ----------
    table: astropy.table.Table
        A table of lines from ORIGIN. The table must contain the columns: ID,
        ra, dec, flux, seg_label, comp, merged_in, and line_merged_flag.

    Returns
    -------
    astropy.table.Table
        Table with unique sources.

    """
    table_by_id = table.group_by('ID')

    result_rows = []
    for key, group in zip(table_by_id.groups.keys, table_by_id.groups):
        group_id = key['ID']

        ra_waverage = np.average(group['ra'], weights=group['flux'])
        dec_waverage = np.average(group['dec'], weights=group['flux'])

        n_lines = len(group[group['merged_in'].mask])

        seg_label = group['seg_label'][0]
        comp = group['comp'][0]
        # TODO: seg_label and comp should be the same for all the lines
        # associated to the source, shall we nevertheless check this is the
        # case?

        line_merged_flag = np.any(group["line_merged_flag"])

        result_rows.append([group_id, ra_waverage, dec_waverage, n_lines,
                            seg_label, comp, line_merged_flag])

    return Table(rows=result_rows, names=["ID", "ra", "dec", "n_lines",
                                          "seg_label", "comp",
                                          "line_merged_flag"])


def remove_identical_duplicates(table):
    """Remove strictly identical duplicated lines.

    ORIGIN may find lines at slightly different (x0, y0, z0) positions that are
    set to the very same (x, y, z) position when computing the optimal
    position.

    For such duplicates, this function only keep the line with the highest
    purity.

    Parameters
    ----------
    table: astropy.table.Table
        A table of lines from ORIGIN. The table must contain the columns ID, x,
        y, z, and purity.

    Returns
    -------
    astropy.table.Table
        Table with only unique (x, y, z) rows.

    """
    table = table.copy()

    # Sort by decreasing purity
    table.sort('purity')
    table.reverse()

    # Find position of first unique (x, y, z)
    _, idx = np.unique(table['x', 'y', 'z'], axis=0, return_index=True)

    result = table[idx].copy()
    result.sort(['ID', 'z'])

    return result


def merge_similar_lines(table, *, z_pix_threshold=5):
    """Merge and flag possibily duplicated lines.

    Some ORIGIN tables associate several identical lines at different positions
    to the same object (same ID).  Lines are considered as duplicated if they
    are withing the given threshold in the spectral (z) axis.

    We mark the duplicated lines as merged into the line of highest flux and
    we flag the object as having duplicated lines in the table, as the
    information may not be reliable.

    Parameters
    ----------
    table: astropy.table.Table
        A table of lines from ORIGIN. The table must contain the columns: ID,
        z, num_line, and purity.
    z_pix_threshold: int
        Pixel threshold on the spectral axis.  When two lines are nearer than
        this threshold, they are considered as the same line. Note that the
        association percolates and may associated lines further than the
        threshold.

    Returns
    -------
    astropy.table.Table
        Table with the same rows and with the supplementary merged_in and
        line_merged_flag columns.

    """
    table = table.copy()

    # We use the table grouping functionality of astropy to browse by object
    # and group of identical lines.  Table grouping does not allow to modify
    # the underlying table, so we first browse the groups and get the indexes
    # of rows to modify and then we perform the modifications on the full
    # table.
    # List of row indexes to flag has having been merged.
    idx_to_flag = []
    # Dictionary associating line identifiers (from num_line column) to the
    # index of the row that have been merged with this line.
    merge_dict = {}
    # Column containing the row indexes to access them while in groups.
    table.add_column(Column(data=np.arange(len(table), dtype=int),
                            name="_idx"))

    for group in table.group_by('ID').groups:
        # TODO: If astropy guaranties that grouping retains the row order, it
        # is faster to sort by z before grouping (and before adding _idx).
        group.sort('z')

        # Boolean array of the same length of the group indicating for each
        # line if it's different from the previous (with True for the first
        # line).
        different_from_previous = np.concatenate(
            ([True], (group['z'][1:] - group['z'][:-1]) >= z_pix_threshold)
        )
        # By computing the cumulative sum on this array, we get an array of
        # increasing integers where a succession of same number identify
        # identical lines.
        line_groups = np.cumsum(different_from_previous)

        for subgroup in group.group_by(line_groups).groups:
            if len(subgroup) > 1:
                subgroup.sort('flux')
                idx_to_flag += list(subgroup['_idx'])
                merge_dict[subgroup['num_line'][-1]] = subgroup['_idx'][:-1]

    table['line_merged_flag'] = False
    table['line_merged_flag'][idx_to_flag] = True

    table.add_column(MaskedColumn(data=np.full(len(table), -9999, dtype=int),
                                  name="merged_in",
                                  mask=np.full(len(table), True),
                                  fill_value=-9999))
    for line_id, row_indexes in merge_dict.items():
        table['merged_in'][row_indexes] = line_id

    table.remove_columns('_idx')
    table.sort(['ID', 'z'])

    return table


def trim_spectrum_list(line_table, spectra, profile_fwhm, *, size_fwhm=3):
    """Keep only relevant spectra and limit their extent around the line.

    The “compute spectra” step creates a list of spectra (data and variance)
    associated to each line.  These spectra are based on the full MUSE
    wavelength grid.  This function:

    - limits the spectra list to the list of lines present in the line_table
      (e.g. if the table was processed by remove_identical_duplicates the
      duplicated spectra will be removed);
    - limit the wavelength grid of the spectra around the associated line.

    TODO: Include the limiting of the spectra in the spectrum computation code.

    Parameters
    ----------
    line_table: astropy.table.Table
        An ORIGIN table of lines, this table must contain the columns:
        num_line, profile, and z.
    spectra: List[mpdaf.obj.spectrum.Spectrum]
        List of spectrum object associated to each line. The index in the list
        must be the num_line of the associated line in the line table.
    profile_fwhm: list
        List of the profile FWHMs in pixels. The index in the list is the
        profile number.
    size_fwhm: float
        The length of the spectrum to extract around the line in FWHM factor.

    Returns
    -------
    List[mpdaf.obj.spectrum.Spectrum]
        List of spectrum objects.  The index in the list correspond to the
        index of the line in the line_table.

    """
    radius = np.ceil(np.array(profile_fwhm) * size_fwhm / 2)

    result = []

    for row in line_table:
        num_line, line_profile, line_z = row[['num_line', 'profile', 'z']]
        sp = spectra[num_line]
        result.append(sp.subspec(line_z - radius[line_profile],
                                 line_z + radius[line_profile], unit=None))

    return result


def create_masks(line_table, source_table, profile_fwhm, correl_cube,
                 correl_threshold, std_cube, std_threshold, segmap, out_dir, *,
                 mask_size=50, seg_thres_factor=.5, plot_problems=True):
    """Create the mask of each source.

    This function creates the masks and sky masks of the sources in the line
    table using the ``origin.source_masks.gen_source_mask`` function on each
    source. The primary source masks are created using the correl_cube while
    the complementary source masks are created using the std_cube.

    The correl_cube and std_cube are expected to have the same WCS.

    TODO: Implement parallel processing.

    Parameters
    ----------
    line_table: astropy.table.Table
        ORIGIN table of lines, this table must contain the columns: ID, x0, y0,
        z0, comp, and profile.
    source_table: astropy.table.Table
        ORIGIN table containing the source list.  This table is used to get the
        position of the source.
    profile_fwhm: list
        List of the profile FWHMs. The index in the list is the profile number.
    correl_cube: mpdaf.obj.Cube
        Correlation cube where primary sources where detected.
    correl_threshold: float
        Threshold used for detection of sources in the correl_cube.
    std_cube: mpdaf.obj.Cube
        STD cube where complementary sources where detected.
    std_threshold: float
        Threshold used for detection of sources in the STD cube.
    segmap: mpdaf.obj.Image
        Segmentation map. Must have the same spatial WCS as the cube. The sky
        must be in segment 0.
    out_dir: str
        Directory into which the masks will be created.
    mask_size: int
        Width in pixel for the square masks.
    seg_thres_factor: float
        Factor applied to the detection thresholds to get the threshold used
        for segmentation. The default is to take half of it.
    plot_problems: bool
        If true, the problematic sources will be reprocessed by gen_source_mask
        in verbose mode to produce various plots of the mask creation process.

    """
    source_table = source_table.copy()
    source_table.add_index('ID')

    # Add pixel positions of the sources in the main WCS.
    source_table['ra'].unit, source_table['dec'].unit = u.deg, u.deg
    source_table['y'], source_table['x'] = correl_cube.wcs.sky2pix(
        np.array([source_table['dec'], source_table['ra']]).T).T

    # The segmentation must be done at the exact position of the lines found by
    # ORIGIN (x0, y0, z0) and not the computed “optimal” position (x, y, z, ra,
    # and dec).  Using this last postion may cause problem when it falls just
    # outside the segment.  We replace ra, dec, and z by the initial values.
    line_table = line_table.copy()
    line_table['dec'], line_table['ra'] = correl_cube.wcs.pix2sky(
        np.array([line_table['y0'], line_table['x0']]).T).T
    line_table['z'] = line_table['z0']
    # We also add a fwhm column containing the FWHM of the line profile as
    # it is used for mask creation.
    line_table['fwhm'] = [profile_fwhm[profile]
                          for profile in line_table['profile']]

    # Convert segmap to sky map (1 where sky)
    skymap = segmap.copy()
    skymap._data = (skymap._data == 0).astype(int)

    by_id = line_table.group_by('ID')

    for key, group in zip(by_id.groups.keys, by_id.groups):
        source_id = key['ID']
        source_x, source_y = source_table.loc[source_id]['x', 'y']

        if source_table.loc[source_id]['comp'] == 0:
            detection_cube = correl_cube
            threshold = correl_threshold * seg_thres_factor
        else:
            detection_cube = std_cube
            threshold = std_threshold * seg_thres_factor

        gen_mask_return = gen_source_mask(
            source_id, source_x, source_y,
            lines=group, detection_cube=detection_cube, threshold=threshold,
            cont_sky=skymap, out_dir=out_dir
        )

        if gen_mask_return is not None:
            with open(f"{out_dir}/problematic_masks.txt", 'a') as out:
                out.write(f"{gen_mask_return}\n")
            if plot_problems:
                gen_mask_return = gen_source_mask(
                    source_id, source_x, source_y,
                    lines=group, detection_cube=detection_cube,
                    threshold=threshold, cont_sky=skymap, out_dir=out_dir,
                    verbose=True
                )
