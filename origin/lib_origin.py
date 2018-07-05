"""Contains most of the methods that compose the ORIGIN software."""

import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np

from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Gaussian1D
from astropy.nddata import overlap_slices
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table, Column
from functools import wraps
from joblib import Parallel, delayed
from numpy import fft
from numpy.linalg import multi_dot
from scipy import stats, fftpack
from scipy.signal import fftconvolve
from scipy.ndimage import label as ndi_label, maximum_filter
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial import ConvexHull, cKDTree
from scipy.sparse.linalg import svds
from scipy.interpolate import interp1d
from time import time

from .source_masks import gen_source_mask


def timeit(f):
    """Decorator which prints the execution time of a function."""
    @wraps(f)
    def timed(*args, **kw):
        logger = logging.getLogger(__name__)
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
    logger = logging.getLogger('origin')
    if logging.getLevelName(logger.getEffectiveLevel()) == 'ERROR':
        kwargs['disable'] = True

    from tqdm import tqdm, tqdm_notebook
    func = tqdm_notebook if isnotebook() else tqdm
    return func(*args, **kwargs)


def orthogonal_projection(a, b):
    """Compute the orthogonal projection: a.(a^T.a)-1.a^T.b
    NOTE: does not include the (a^T.a)-1 term as it is often not needed (when
    a is already normalized).
    """
    # Using multi_dot which is faster than np.dot(np.dot(a, a.T), b)
    # Another option would be to use einsum, less readable but also very
    # fast with Numpy 1.14+ and optimize=True. This seems to be as fast as
    # multi_dot.
    # return np.einsum('i,j,jk->ik', a, a, b, optimize=True)
    if a.ndim == 1:
        a = a[:, None]
    return multi_dot([a, a.T, b])


@timeit
def spatial_segmentation(Nx, Ny, NbSubcube, start=None):
    """Compute indices to split spatially in NbSubcube x NbSubcube regions.

    Each zone is computed from the left to the right and the top to the bottom
    First pixel of the first zone has coordinates : (row,col) = (Nx,1).

    Parameters
    ----------
    Nx : int
        Number of columns
    Ny : int
        Number of rows
    NbSubcube : int
        Number of subcubes for the spatial segmentation
    start : tuple
        if not None, the tupe is the (y,x) starting point

    Returns
    -------
    intx, inty : int, int
        limits in pixels of the columns/rows for each zone

    """
    # Segmentation of the rows vector in Nbsubcube parts from right to left
    inty = np.linspace(Ny, 0, NbSubcube + 1, dtype=np.int)
    # Segmentation of the columns vector in Nbsubcube parts from left to right
    intx = np.linspace(0, Nx, NbSubcube + 1, dtype=np.int)

    if start is not None:
        inty += start[0]
        intx += start[1]

    return inty, intx


def DCTMAT(nl, order):
    """Return the DCT transformation matrix of size nl-by-(order+1).

    Equivalent function to Matlab/Octave's dtcmtx.
    https://octave.sourceforge.io/signal/function/dctmtx.html

    Parameters
    ----------
    order : int
        Order of the DCT (spectral length).

    Returns
    -------
    array: DCT Matrix

    """
    yy, xx = np.mgrid[:nl, :order + 1]
    D0 = np.sqrt(2 / nl) * np.cos((yy + 0.5) * (np.pi / nl) * xx)
    D0[:, 0] *= 1 / np.sqrt(2)
    return D0


@timeit
def dct_residual(w_raw, order, var, approx, mask):
    """Function to compute the residual of the DCT on raw data.

    Parameters
    ----------
    w_raw : array
        Data array.
    order : int
        The number of atom to keep for the DCT decomposition.
    var : array
        Variance array.
    approx : bool
        If True, an approximate computation is used, not taking the variance
        into account.

    Returns
    -------
    Faint, cont : array
        Residual and continuum estimated from the DCT decomposition.

    """
    nl = w_raw.shape[0]
    D0 = DCTMAT(nl, order)
    shape = w_raw.shape[1:]
    nspec = np.prod(shape)

    if approx:
        # Compute the DCT transformation, without using the variance.
        #
        # Given the transformation matrix D0, we compute for each spectrum S:
        #
        #   C = D0.D0^t.S
        #

        # Old version using tensordot:
        # A = np.dot(D0, D0.T)
        # cont = np.tensordot(A, w_raw, axes=(0, 0))

        # Looping on spectra and using multidot is ~6x faster:
        # D0 is typically 3681x11 elements, so it is much more efficient
        # to compute D0^t.S first (note the array is reshaped below)
        cont = [multi_dot([D0, D0.T, w_raw[:, y, x]])
                for y, x in ProgressBar(np.ndindex(shape), total=nspec)]

        # For reference, this is identical to the following scipy version,
        # though scipy is 2x slower than tensordot (probably because it
        # computes all the coefficients)
        # from scipy.fftpack import dct
        # w = (np.arange(nl) < (order + 1)).astype(int)
        # cont = dct(dct(w_raw, type=2, norm='ortho', axis=0) * w[:,None,None],
        #            type=3, norm='ortho', axis=0, overwrite_x=False)
    else:
        # Compute the DCT transformation, using the variance.
        #
        # As the noise differs on each spectral component, we need to take into
        # account the (diagonal) covariance matrix Σ for each spectrum S:
        #
        #   C = D0.(D^t.Σ^-1.D)^-1.D0^t.Σ^-1.S
        #

        w_raw_var = w_raw / var
        D0T = D0.T

        # Old version (slow):
        # def continuum(D0, D0T, var, w_raw_var):
        #     A = np.linalg.inv(np.dot(D0T / var, D0))
        #     return np.dot(np.dot(np.dot(D0, A), D0T), w_raw_var)
        #
        # cont = Parallel()(
        #     delayed(continuum)(D0, D0T, var[:, i, j], w_raw_var[:, i, j])
        #     for i in range(w_raw.shape[1]) for j in range(w_raw.shape[2]))
        # cont = np.asarray(cont).T.reshape(w_raw.shape)

        # map of valid spaxels, i.e. spaxels with at least one valid value
        valid = ~np.any(mask, axis=0)

        from numpy.linalg import inv
        cont = []
        for y, x in ProgressBar(np.ndindex(shape), total=nspec):
            if valid[y, x]:
                res = multi_dot([D0,
                                 inv(np.dot(D0T / var[:, y, x], D0)),
                                 D0T,
                                 w_raw_var[:, y, x]])
            else:
                res = multi_dot([D0, D0.T, w_raw[:, y, x]])
            cont.append(res)

    return np.stack(cont).T.reshape(w_raw.shape)


def compute_segmap_gauss(data, pfa, fwhm_fsf=0):
    """Compute segmentation map from an image, using gaussian statistics.

    Parameters
    ----------
    data : array
        Input values, typically from a O2 test.
    pfa : float
        Desired false alarm.
    fwhm : int
        Width (in integer pixels) of the filter, to convolve with a PSF disc.

    Returns
    -------
    float, array
        threshold, and labeled image.

    """
    # test threshold : uses a Gaussian approximation of the test statistic
    # under H0
    histO2, frecO2, gamma, mea, std = compute_thresh_gaussfit(data, pfa)

    # threshold - erosion and dilation to clean ponctual "source"
    mask = data > gamma
    mask = binary_erosion(mask, border_value=1, iterations=1)
    mask = binary_dilation(mask, iterations=1)

    # convolve with PSF
    if fwhm_fsf > 0:
        fwhm_pix = int(fwhm_fsf) // 2
        size = fwhm_pix*2 + 1
        disc = np.hypot(*list(np.mgrid[:size, :size] - fwhm_pix)) < fwhm_pix
        mask = fftconvolve(mask, disc, mode='same')
        mask = mask > 1e-9

    return gamma, ndi_label(mask)[0]


def createradvar(cu, ot):
    """Compute the compactness of areas using variance of position.

    The variance is computed on the position given by adding one of the 'ot'
    to 'cu'.

    Parameters
    ----------
    cu : 2D array
        The current array
    ot : 3D array
        The other array

    Returns
    -------
    var : array
        The radial variances

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
    NbSubcube : int
        Number of subcubes for the spatial segmentation
    Nx        : int
        Number of columns
    Ny        : int
                Number of rows


    Returns
    -------
    label :     array
                label of the fused square

    """
    # square area index with borders
    Vert = np.sum(nexpmap, axis=1)
    Hori = np.sum(nexpmap, axis=0)
    y1 = np.where(Vert > 0)[0][0]
    x1 = np.where(Hori > 0)[0][0]
    y2 = Ny - np.where(Vert[::-1] > 0)[0][0]
    x2 = Nx - np.where(Hori[::-1] > 0)[0][0]
    start = (y1, x1)
    inty, intx = spatial_segmentation(Nx, Ny, NbSubcube, start=start)

    # % FUSION square AREA
    label = []
    for numy in range(NbSubcube):
        for numx in range(NbSubcube):
            y1, y2, x1, x2 = inty[numy + 1], inty[numy], intx[numx], intx[numx + 1]
            tmp = nexpmap[y1:y2, x1:x2]
            if np.mean(tmp) != 0:
                labtest = ndi_label(tmp)[0]
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
        label of fused square generated in area_segmentation_square_fusion
    pfa   :     float
        Pvalue for the test which performs segmentation
    NbSubcube : int
        Number of subcubes for the spatial segmentation
    Nx        : int
        Number of columns
    Ny        : int
        Number of rows


    Returns
    -------
    label_out : array
                label of the fused square and sources

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
    NbArea           : int
        Number of area
    cube_std         : array
        Cube data weighted by the standard deviation
    areamap          : array
        Map of areas
    Noise_population : float
        Proportion of estimated noise part used to define the
        background spectra
    threshold_test   : list
        User given list of threshold (not pfa) to apply on each area, the
        list is of length NbAreas or of length 1.
    itermax          : int
        Maximum number of iterations
    testO2           : list of arrays
        Result of the O2 test

    Returns
    -------
    cube_faint : array
                Faint greedy decomposition od STD Cube

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


def Compute_PCA_threshold(faint, pfa):
    """Compute threshold for the PCA.

    Parameters
    ----------
    faint : array
        Standardized data.
    pfa : float
        PFA of the test.

    Returns
    -------
    test, histO2, frecO2, thresO2, mea, std
        Threshold for the O2 test

    """
    test = O2test(faint)

    # automatic threshold computation
    histO2, frecO2, thresO2, mea, std = compute_thresh_gaussfit(test, pfa)

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
    itermax          : int
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
                logger.warning('Warning iterations stopped at %d', nbiter)
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


def O2test(arr):
    """Compute the second order test on spaxels.

    The test estimate the background part and nuisance part of the data by mean
    of second order test: Testing mean and variance at same time of spectra.

    Parameters
    ----------
    arr : array-like
        The 3D cube data to test.

    Returns
    -------
    ndarray
        result of the test.
    """
    # np.einsum('ij,ij->j', arr, arr) / arr.shape[0]
    return np.mean(arr**2, axis=0)


def compute_thresh_gaussfit(data, pfa):
    """Compute a threshold with a gaussian fit of a distribution.

    Parameters
    ----------
    data : array
        2D data from the O2 test.
    pfa : float
        Desired false alarm.

    Returns
    -------
    histO2  : histogram value of the test
    frecO2  : frequencies of the histogram
    thresO2 : automatic threshold for the O2 test
    mea     : mean value of the fit
    std     : sigma value of the fit

    """
    logger = logging.getLogger(__name__)
    data = data[data > 0]
    histO2, frecO2 = np.histogram(data, bins='fd', density=True)
    ind = np.argmax(histO2)
    mod = frecO2[ind]
    ind2 = np.argmin((histO2[ind] / 2 - histO2[:ind])**2)
    fwhm = mod - frecO2[ind2]
    sigma = fwhm / np.sqrt(2 * np.log(2))

    coef = stats.norm.ppf(pfa)
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

    norm_profile[norm_profile <= 0] = np.inf
    np.sqrt(norm_profile, out=norm_profile)
    cube_profile /= norm_profile
    return cube_profile


def _convolve_spectral(parallel, nslices, arr, shape, func=fft.rfftn):
    arr = np.array_split(arr, nslices, axis=-1)
    out = parallel(delayed(func)(chunk, shape, axes=(0,)) for chunk in arr)
    return np.concatenate(out, axis=-1)


@timeit
def Correlation_GLR_test(cube, sigma, fsf, weights, profiles, nthreads=1,
                         pcut=None, pmeansub=True):
    """Function to compute the cube of GLR test values obtained with the given
    PSF and dictionary of spectral profile.

    Parameters
    ----------
    cube : array
        data cube
    sigma : array
        variance cube
    fsf : list of arrays
        FSF for each field of this data cube
    weights : list of array
        Weight maps of each field
    profiles : list of ndarray
        Dictionary of spectral profiles to test
    nthreads : int
        number of threads
    pcut : float
        Cut applied to the profiles to limit their width
    pmeansub : bool
        Subtract the mean of the profiles

    Returns
    -------
    correl : array
        cube of T_GLR values of maximum correlation
    profile : array
        Number of the profile associated to the T_GLR
    correl_min : array
        cube of T_GLR values of minimum correlation

    """
    logger = logging.getLogger(__name__)
    Nz, Ny, Nx = cube.shape

    # Spatial convolution of the weighted data with the zero-mean FSF
    logger.info('Step 1/3 and 2/3: '
                'Spatial convolution of weighted data with the zero-mean FSF, '
                'Computing Spatial part of the norm of the 3D atoms')
    if weights is None:  # one FSF
        fsf = [fsf]
        weights = [None]

    nfields = len(fsf)
    fields = range(nfields)
    if nfields > 1:
        fields = ProgressBar(fields)

    if nthreads != 1:
        # copy the arrays because otherwise joblib's memmap handling fails
        # (maybe because of astropy.io.fits doing weird things with the memap?)
        cube = np.array(cube)
        sigma = np.array(sigma)

    with Parallel(n_jobs=nthreads) as parallel:
        for nf in fields:
            # convolve spatially each spectral channel by the FSF, and do the
            # same for the norm (inverse variance)
            res = parallel(ProgressBar([
                delayed(_convolve_fsf)(fsf[nf][i], cube[i], sigma[i],
                                       weights=weights[nf])
                for i in range(Nz)]))
            res = [np.stack(arr) for arr in zip(*res)]
            if nf == 0:
                cube_fsf, norm_fsf = res
            else:
                cube_fsf += res[0]
                norm_fsf += res[1]

    # First cube of correlation values
    # initialization with the first profile
    logger.info('Step 3/3 Computing second cube of correlation values')

    # Prepare profiles:
    # Cut the profiles and subtract the mean, if asked to do so
    prof_cut = []
    for prof in profiles:
        prof = prof.copy()
        if pcut is not None:
            lmin, lmax = np.where(prof >= pcut)[0][[0, -1]]
            prof = prof[lmin:lmax + 1]
        prof /= np.linalg.norm(prof)
        if pmeansub:
            prof -= prof.mean()
        prof_cut.append(prof)

    # compute the optimal shape for FFTs (on the wavelength axis).
    # For profiles with different shapes, we need to know the indices to
    # extract the signal from the inverse fft.
    s1 = np.array(cube_fsf.shape)                          # cube shape
    s2 = np.array([(d.shape[0], 1, 1) for d in prof_cut])  # profiles shape
    fftshape = s1 + s2 - 1                                 # fft shape
    fshape = [fftpack.helper.next_fast_len(int(d))         # optimal fft shape
              for d in fftshape.max(axis=0)[:1]]

    # and now computes the indices to extract the cube from the inverse fft.
    startind = (fftshape - s1) // 2
    endind = startind + s1
    cslice = [slice(startind[k, 0], endind[k, 0]) for k in range(len(endind))]

    # Compute the FFTs of the cube and norm cube, splitting them on multiple
    # threads if needed
    with Parallel(n_jobs=nthreads, backend='threading') as parallel:
        cube_fft = _convolve_spectral(parallel, nthreads, cube_fsf, fshape,
                                      func=fft.rfftn)
        norm_fft = _convolve_spectral(parallel, nthreads, norm_fsf, fshape,
                                      func=fft.rfftn)

    cube_fsf = norm_fsf = res = None

    cube_fft = cube_fft.reshape(cube_fft.shape[0], -1)
    norm_fft = norm_fft.reshape(norm_fft.shape[0], -1)
    profile = np.empty((Nz, Ny * Nx), dtype=np.uint8)
    correl = np.full((Nz, Ny * Nx), -np.inf)
    correl_min = np.full((Nz, Ny * Nx), np.inf)

    # for each profile, compute convolve the convolved cube and norm cube.
    # Then for each pixel we keep the maximum correlation (and min correlation)
    # and the profile number with the max correl.
    with Parallel(n_jobs=nthreads, backend='threading') as parallel:
        for k in ProgressBar(range(len(prof_cut))):
            cube_profile = _convolve_profile(prof_cut[k], cube_fft, norm_fft,
                                             fshape, nthreads, parallel)
            cube_profile = cube_profile[cslice[k]]
            profile[cube_profile > correl] = k
            np.maximum(correl, cube_profile, out=correl)
            np.minimum(correl_min, cube_profile, out=correl_min)

    profile = profile.reshape(Nz, Ny, Nx)
    correl = correl.reshape(Nz, Ny, Nx)
    correl_min = correl_min.reshape(Nz, Ny, Nx)
    return correl, profile, correl_min


def compute_local_max(correl, correl_min, mask, size=3):
    """Compute the local maxima of the maximum correlation and local maxima
    of minus the minimum correlation distribution.

    Parameters
    ----------
    correl : array
        T_GLR values with edges excluded (from max correlation)
    correl_min : array
        T_GLR values with edges excluded (from min correlation)
    mask : array
        mask array (true if pixel is masked)
    size : int
        Number of connected components

    Returns
    -------
    array, array
        local maxima of correlations and local maxima of -correlations

    """
    # local maxima of maximum correlation
    if np.isscalar(size):
        size = (size, size, size)
    local_max = maximum_filter(correl, size=size)
    local_mask = (correl == local_max)
    local_mask[mask] = False
    local_max *= local_mask

    # local maxima of minus minimum correlation
    minus_correl_min = - correl_min
    local_min = maximum_filter(minus_correl_min, size=size)
    local_mask = (minus_correl_min == local_min)
    local_mask[mask] = False
    local_min *= local_mask

    return local_max, local_min


def itersrc(cat, tol_spat, tol_spec, n, id_cu):
    """Recursive function to perform the spatial merging.

    If neighborhood are close spatially to a lines: they are merged,
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
    cat : kinda of catalog of the previously merged lines
        xout,yout,zout,aout,iout:
        the 3D position, area label and ID for all analysed lines
    tol_spat : int
        spatial tolerance for the spatial merging
    tol_spec : int
        spectral tolerance for the spectral merging
    n : int
        index of the original seed
    id_cu :
        ID of the original seed

    """
    # compute spatial distance to other points.
    # - id_cu is the detection processed at the start (from
    #   spatiospectral_merging), while n is the detection currently processed
    #   in the recursive call
    matched = cat['matched']
    spatdist = np.hypot(cat['x0'][n] - cat['x0'],
                        cat['y0'][n] - cat['y0'])
    spatdist[matched] = np.inf

    cu_spat = np.hypot(cat['x0'][id_cu] - cat['x0'],
                       cat['y0'][id_cu] - cat['y0'])
    cu_spat[matched] = np.inf

    ind = np.where(spatdist < tol_spat)[0]
    if len(ind) == 0:
        return

    for indn in ind:
        if not matched[indn]:
            if cu_spat[indn] > tol_spat * np.sqrt(2):
                # check spectral content
                dz = np.sqrt((cat['z0'][indn] - cat['z0'][id_cu])**2)
                if dz < tol_spec:
                    cat[indn]['matched'] = True
                    cat[indn]['imatch'] = id_cu
                    itersrc(cat, tol_spat, tol_spec, indn, id_cu)
            else:
                cat[indn]['matched'] = True
                cat[indn]['imatch'] = id_cu
                itersrc(cat, tol_spat, tol_spec, indn, id_cu)


def spatiospectral_merging(tbl, tol_spat, tol_spec):
    """Perform the spatial and spatio spectral merging.

    The spectral merging give the same ID if several group of lines (from
    spatial merging) if they share at least one line frequency

    Parameters
    ----------
    tbl : `astropy.table.Table`
        ID,x,y,z,...
    tol_spat : int
        spatial tolerance for the spatial merging
    tol_spec : int
        spectral tolerance for the spectral merging

    Returns
    -------
    `astropy.table.Table`
        Table: id, x, y, z, area, imatch, imatch2
        imatch is the ID after spatial and spatio spectral merging.
        imatch2 is the ID after spatial merging only.

    """
    Nz = len(tbl)
    tbl['_id'] = np.arange(Nz)                 # id of the detection
    tbl['matched'] = np.zeros(Nz, dtype=bool)  # is the detection matched ?
    tbl['imatch'] = np.arange(Nz)              # id of the matched detection

    for row in tbl:
        if not row['matched']:
            row['matched'] = True
            itersrc(tbl, tol_spat, tol_spec, row['_id'], row['_id'])

    # renumber output IDs
    for n, imatch in enumerate(np.unique(tbl['imatch'])):
        # for detections in multiple segmap regions, set the max region
        # number... this is needed to select all detections in the loop below
        ind = tbl['imatch'] == imatch
        tbl['area'][ind] = tbl['area'][ind].max()
        tbl['imatch'][ind] = n
    tbl.sort('imatch')

    # Special treatment for segmap regions, merge sources with close
    # spectral lines
    tbl['imatch2'] = tbl['imatch']  # store result before spectral merging
    iout = tbl['imatch']
    zout = tbl['z0']
    for n, area_cu in enumerate(np.unique(tbl['area'])):
        if area_cu > 0:
            # take all detections inside a segmap region
            ind = np.where(tbl['area'] == area_cu)[0]
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
                                # if the minimum z distance is less than
                                # tol_spec, then merge the sources
                                iout[iout == otg] = cu

    tbl.remove_columns(('_id', 'matched'))
    return tbl


@timeit
def Compute_threshold_purity(purity, cube_local_max, cube_local_min,
                             segmap=None, threshlist=None):
    """Compute threshold values corresponding to a given purity.

    Parameters
    ----------
    purity : float
        The target purity between 0 and 1.
    cube_local_max : array
        Cube of local maxima from maximum correlation.
    cube_local_min : array
        Cube of local maxima from minus minimum correlation.
    segmap : array
        Segmentation map to get the background regions.
    threshlist : list
        List of thresholds to compute the purity (default None).

    Returns
    -------
    threshold : float
        The estimated threshold associated to the purity.
    res : astropy.table.Table
        Table with the purity results for each threshold:
        - PVal_r : The purity function
        - index_pval : index value to plot
        - Det_m : Number of detections (-DATA)
        - Det_M : Number of detections (+DATA)

    """
    logger = logging.getLogger(__name__)

    # total number of spaxels
    L1 = np.prod(cube_local_min.shape[1:])

    # background only
    if segmap is not None:
        segmask = segmap == 0
        cube_local_min = cube_local_min * segmask
        # number of spaxels considered for calibration
        L0 = np.count_nonzero(segmask)
        logger.info('using only background pixels (%.1f%%)', L0 / L1 * 100)
    else:
        L0 = L1

    if threshlist is None:
        threshmax = min(cube_local_min.max(), cube_local_max.max())
        threshmin = np.median(np.amax(cube_local_max, axis=0)) * 1.1
        threshlist = np.linspace(threshmin, threshmax, 50)
    else:
        threshmin = np.min(threshlist)

    locM = cube_local_max[cube_local_max > threshmin]
    locm = cube_local_min[cube_local_min > threshmin]

    n0, n1 = [], []
    for thresh in ProgressBar(threshlist):
        n1.append(np.count_nonzero(locM > thresh))
        n0.append(np.count_nonzero(locm > thresh))

    n0 = np.array(n0) * (L1 / L0)
    n1 = np.array(n1)
    est_purity = 1 - n0/n1
    res = Table([threshlist, est_purity, n0.astype(int), n1],
                names=('Tval_r', 'Pval_r', 'Det_m', 'Det_M'))
    res['Tval_r'].format = '.2f'
    res['Pval_r'].format = '.2f'
    res.sort('Tval_r')
    logger.debug("purity values:\n%s", res)

    if est_purity[-1] < purity:
        logger.warning('Maximum computed purity %.2f is below %.2f',
                       est_purity[-1], purity)
        threshold = np.inf
    else:
        threshold = np.interp(purity, res['Pval_r'], res['Tval_r'])
        detect = np.interp(threshold, res['Tval_r'], res['Det_M'])
        logger.info('Interpolated Threshold %.2f Detection %d for Purity %.2f',
                    threshold, detect, purity)

    return threshold, res


def extract_grid(raw_in, var_in, psf_in, weights_in, y, x, size_grid):
    """Function to extract data from an estimated source in catalog.

    Parameters
    ----------
    raw_in : array
        RAW data
    var_in : array
        MUSE covariance
    psf_in : array
        MUSE PSF
    weights_in : array
        PSF weights
    y : int
        y position in pixek estimated in previous catalog
    x : int
        x position in pixek estimated in previous catalog
    size_grid : int
        Maximum spatial shift for the grid

    Returns
    -------
    red_dat : cube of raw_in centered in y,x of size PSF+Max spatial shift
    red_var : cube of var_in centered in y,x of size PSF+Max spatial shift
    red_wgt : cube of weights_in centered in y,x of size PSF+Max spatial shift
    red_psf : cube of psf_in centered in y,x of size PSF+Max spatial shift

    """
    # size psf
    if weights_in is None:
        psf_shape = psf_in.shape[1:]
    else:
        psf_shape = psf_in[0].shape[1:]

    # desired shape
    margin = 2 * size_grid
    shape = (psf_shape[0] + margin, psf_shape[1] + margin)
    cshape = (raw_in.shape[0], ) + shape

    (psy, psx), (psy2, psx2) = overlap_slices(raw_in.shape[1:], shape, (y, x))

    # create weight, data with bordure
    red_dat = np.zeros(cshape)
    red_dat[:, psy2, psx2] = raw_in[:, psy, psx]

    red_var = np.full(cshape, np.inf)
    red_var[:, psy2, psx2] = var_in[:, psy, psx]

    if weights_in is None:
        red_wgt = None
        red_psf = psf_in
    else:
        red_wgt = []
        red_psf = []
        for n, w in enumerate(weights_in):
            if np.sum(w[psy, psx]) > 0:
                w_tmp = np.zeros(shape)
                w_tmp[psy2, psx2] = w[psy, psx]
                red_wgt.append(w_tmp)
                red_psf.append(psf_in[n])

    return red_dat, red_var, red_wgt, red_psf


def LS_deconv_wgt(data_in, var_in, psf_in):
    """Function to compute the Least Square estimation of a ponctual source.

    Parameters
    ----------
    data_in : array
        input data
    var_in : array
        input variance
    psf_in : array
        weighted MUSE PSF

    Returns
    -------
    deconv_out : array
        LS Deconvolved spectrum
    varest_out : array
        estimated theoretic variance

    """
    # deconvolution
    nl = psf_in.shape[0]
    var = var_in.reshape(nl, -1)
    psf = psf_in.reshape(nl, -1)
    data = data_in.reshape(nl, -1)
    varest_out = 1 / np.sum(psf * psf / var, axis=1)
    deconv_out = np.sum(psf * data / np.sqrt(var), axis=1) * varest_out

    return deconv_out, varest_out


def conv_wgt(deconv_met, psf_in):
    """Compute the convolution of a spectrum. output is a cube.

    Parameters
    ----------
    deconv_met : array
        LS Deconvolved spectrum
    psf_in : array
        weighted MUSE PSF

    Returns
    -------
    cube_conv : array
        Cube, convolution from deconv_met

    """
    cube_conv = psf_in * deconv_met[:, np.newaxis, np.newaxis]
    # FIXME: how the following can be useful ?
    cube_conv = cube_conv * (np.abs(psf_in) > 0)
    return cube_conv


def method_PCA_wgt(data_in, var_in, psf_in, order_dct):
    """Function to Perform PCA LS or Denoised PCA LS.

    Algorithm:

    - principal eigen vector is computed, RAW data are orthogonalized
      this is the first estimation to modelize the continuum
    - on residual, the line is estimated by least square estimation
    - the estimated line is convolved by the psf and removed from RAW data
    - principal eigen vector is computed.

      - PCA LS: RAW data are orthogonalized, this is the second estimation
        to modelize the continuum

      - Denoised PCA LS: The eigen vector is denoised by a DCT, with the
        new eigen vector RAW data are orthogonalized, this is the second
        estimation to modelize the continuum

    - on residual, the line is estimated by least square estimation

    Parameters
    ----------
    data_in : array
        RAW data
    var_in : array
        MUSE covariance
    psf_in : array
        MUSE PSF
    order_dct : int
        order of the DCT for the Denoised PCA LS. If None use PCA LS.

    Returns
    -------
    estimated_line : estimated line
    estimated_var  : estimated variance

    """

    # STD
    nl = psf_in.shape[0]
    data_std = data_in / np.sqrt(var_in)
    data_st_pca = data_std.reshape(nl, -1)

    # PCA
    data_in_pca = data_st_pca - data_st_pca.mean(axis=1)[:, np.newaxis]
    U, s, V = svds(data_in_pca, k=1)

    # orthogonal projection
    xest = orthogonal_projection(U, data_in_pca)
    residual = data_std - np.reshape(xest, psf_in.shape)

    # LS deconv
    deconv_out, _ = LS_deconv_wgt(residual, var_in, psf_in)

    # PSF convolution
    conv_out = conv_wgt(deconv_out, psf_in)

    # cleaning the data
    data_clean = (data_in - conv_out) / np.sqrt(var_in)

    # 2nd PCA
    data_in_pca = data_clean.reshape(nl, -1)
    data_in_pca -= data_in_pca.mean(axis=1)[:, np.newaxis]
    U, s, V = svds(data_in_pca, k=1)

    if order_dct is not None:
        # denoise eigen vector with DCT
        D0 = DCTMAT(nl, order_dct)
        U = orthogonal_projection(D0, U)

    # orthogonal projection
    xest = orthogonal_projection(U, data_st_pca)
    cont = np.reshape(xest, psf_in.shape)
    residual = data_std - cont

    # LS deconvolution of the line
    estimated_line, estimated_var = LS_deconv_wgt(residual, var_in, psf_in)

    # PSF convolution of estimated line
    # FIXME: any reason to compute this ??
    # conv_out = conv_wgt(estimated_line, psf_in)

    return estimated_line, estimated_var


def GridAnalysis(data_in, var_in, psf, weight_in, horiz,
                 size_grid, y0, x0, z0, NY, NX, horiz_psf,
                 criteria, order_dct):
    """Compute the estimated emission line and the optimal
    coordinates for each detected lines in a spatio-spectral grid.

    Parameters
    ----------
    data_in : array
        RAW data minicube
    var_in : array
        MUSE covariance minicube
    psf : array
        MUSE PSF minicube
    weight_in : array
        PSF weights minicube
    horiz : int
        Maximum spectral shift to compute the criteria for gridding
    size_grid : int
        Maximum spatial shift for the grid
    y0 : int
        y position in pixel from catalog
    x0 : int
        x position in pixel from catalog
    z0 : int
        z position in pixel from catalog
    NY : int
        Number of y-pixels from Full data Cube
    NX : int
        Number of x-pixels from Full data Cube
    y0 : int
        y position in pixel from catalog
    horiz_psf : int
        Maximum spatial shift in size of PSF to compute the MSE
    criteria : string
        criteria used to choose the candidate in the grid: flux or mse
    order_dct : int
        order of the DCT Used in the Denoised PCA LS, set to None the
        method become PCA LS only

    Returns
    -------
    flux_est_5 : float
        Estimated flux +/- 5
    flux_est_10 : float
        Estimated flux +/- 10
    MSE_5 : float
        Mean square error +/- 5
    MSE_10 : float
        Mean square error +/- 10
    estimated_line : array
        Estimated lines in data space
    estimated_variance : array
        Estimated variance in data space
    y : int
        re-estimated x position in pixel of the source in the grid
    x : int
        re-estimated x position in pixel of the source in the grid
    z : int
        re-estimated x position in pixel of the source in the grid

    """
    if criteria not in ('flux', 'mse'):
        raise ValueError('Bad criteria: (flux) or (mse)')

    shape = (1 + 2 * size_grid, 1 + 2 * size_grid)
    zest = np.zeros(shape)
    if criteria == 'flux':
        fest_00 = np.zeros(shape)
    if criteria == 'mse':
        mse = np.full(shape, np.inf)

    fest_05 = np.zeros(shape)
    mse_5 = np.full(shape, np.inf)

    nl = data_in.shape[0]
    ind_max = slice(max(0, z0 - 5), min(nl, z0 + 6))
    if weight_in is None:
        nl, sizpsf, _ = psf.shape
    else:
        nl, sizpsf, _ = psf[0].shape

    lin_est = np.zeros((nl, ) + shape)
    var_est = np.zeros((nl, ) + shape)
    # half size psf
    longxy = sizpsf // 2
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

                    z_est = peakdet(deconv_met[ind_max])
                    if z_est == 0:
                        break

                    maxz = z0 - 5 + z_est
                    zest[dy, dx] = maxz
                    # ind_z10 = np.arange(maxz-10,maxz+10)

                    lin_est[:, dy, dx] = deconv_met
                    var_est[:, dy, dx] = varest_met

                    # compute MSE
                    ind_hrz = slice(maxz - horiz, maxz + horiz + 1)
                    if criteria == 'mse':
                        LC = conv_wgt(deconv_met[ind_hrz], psf[ind_hrz])
                        LCred = LC[:, inds, inds]
                        r1red = r1[ind_hrz, inds, inds]
                        mse[dy, dx] = np.sum((r1red - LCred)**2) / np.sum(r1red**2)

                    # FIXME: if horiz=5, this is the same as above...
                    ind_z5 = np.arange(max(0, maxz - 5), min(maxz + 6, nl))
                    LC = conv_wgt(deconv_met[ind_z5], psf[ind_z5, :, :])
                    LCred = LC[:, inds, inds]
                    r1red = r1[ind_z5, inds, inds]
                    mse_5[dy, dx] = np.sum((r1red - LCred)**2) / np.sum(r1red**2)

                    # compute flux
                    if criteria == 'flux':
                        fest_00[dy, dx] = np.sum(deconv_met[ind_hrz])
                    fest_05[dy, dx] = np.sum(deconv_met[ind_z5])
                    # fest_10[dy,dx] = np.sum(deconv_met[ind_z10])

    if criteria == 'flux':
        wy, wx = np.where(fest_00 == fest_00.max())
    elif criteria == 'mse':
        wy, wx = np.where(mse == mse.min())

    y = y0 - size_grid + wy
    x = x0 - size_grid + wx
    z = zest[wy, wx]

    flux_est_5 = float(fest_05[wy, wx])
    # flux_est_10 = float( fest_10[wy,wx] )
    MSE_5 = float(mse_5[wy, wx])
    # MSE_10 = float( mse_10[wy,wx] )
    estimated_line = lin_est[:, wy, wx]
    estimated_variance = var_est[:, wy, wx]

    return (flux_est_5, MSE_5, estimated_line.ravel(),
            estimated_variance.ravel(), int(y), int(x), int(z))


def peakdet(v):
    # find all local maxima: x>x-1 & x>x+1
    ind = np.where((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:]))[0] + 1

    # take the maximum and closest from the center
    imax = v.size // 2
    if len(ind) > 0:
        imax = ind[np.argmin((ind - imax)**2)]
    return imax


@timeit
def estimation_line(Cat1, RAW, VAR, PSF, WGT, wcs, wave, size_grid=1,
                    criteria='flux', order_dct=30, horiz_psf=1,
                    horiz=5):
    """Function to compute the estimated emission line and the optimal
    coordinates for each detected lines in a spatio-spectral grid.

    Parameters
    ----------
    Cat1_T : astropy.Table
        Catalog of parameters of detected emission lines selected
        with a narrow band test. Columns: ra dec lbda x0 y0 z0 T_GLR profile
    DATA : array
        RAW data
    VAR : array
        MUSE covariance
    PSF : array
        MUSE PSF
    WGT : array
        PSF weights
    size_grid : int
        Maximum spatial shift for the grid
    criteria : string
        criteria used to choose the candidate in the grid: flux or mse
    order_dct : int
        order of the DCT Used in the Denoised PCA LS, set to None the
        method become PCA LS only
    horiz_psf : int
        Maximum spatial shift in size of PSF to compute the MSE
    horiz : int
        Maximum spectral shift to compute the criteria
    wcs : `mpdaf.obj.WCS`
        RA-DEC coordinates.
    wave : `mpdaf.obj.WaveCoord`
        Spectral coordinates.

    Returns
    -------
    Cat2 : astropy.Table
        Catalog of parameters of detected emission lines.  Columns:
        ra dec lbda x0 x1 y0 y1 z0 z1 T_GLR profile residual flux num_line
    lin_est : list of arrays
        Estimated lines in data space
    var_est : list of arrays
        Estimated lines in SNR space

    """
    ny, nx = RAW.shape[1:]
    res = []
    for src in ProgressBar(Cat1):
        z0, y0, x0 = tuple(src[['z0', 'y0', 'x0']])
        red_dat, red_var, red_wgt, red_psf = extract_grid(RAW, VAR, PSF, WGT,
                                                          y0, x0, size_grid)
        rg = GridAnalysis(red_dat, red_var, red_psf, red_wgt, horiz, size_grid,
                          y0, x0, z0, ny, nx, horiz_psf, criteria, order_dct)
        res.append(rg)

    flux5, res_min5, lin_est, var_est, y_grid, x_grid, z_grid = zip(*res)

    # add real coordinates
    Cat2 = Cat1.copy()
    dec, ra = wcs.pix2sky(np.stack((y_grid, x_grid)).T).T
    Cat2['ra'] = ra
    Cat2['dec'] = dec
    Cat2['lbda'] = wave.coord(z_grid)

    col_flux = Column(name='flux', data=flux5)
    col_res = Column(name='residual', data=res_min5)
    col_num = Column(name='num_line', data=np.arange(1, len(Cat2) + 1))
    col_x = Column(name='x', data=x_grid)
    col_y = Column(name='y', data=y_grid)
    col_z = Column(name='z', data=z_grid)

    Cat2.add_columns([col_x, col_y, col_z, col_res, col_flux, col_num],
                     indexes=[4, 5, 6, 8, 8, 8])

    return Cat2, lin_est, var_est


def purity_estimation(cat, Pval, Pval_comp):
    """Function to compute the estimated purity for each line.

    Parameters
    ----------
    cat : astropy.Table
        Catalog of parameters of detected emission lines selected
        with a narrow band test.
    Pval : astropy.table.Table
        Table with the purity results for each threshold
    Pval_comp : astropy.table.Table
        Table with the purity results for each threshold, in complementary

    Returns
    -------
    astropy.Table
        Catalog of parameters of detected emission lines.
        Columns: ra dec lbda x0 x1 y0 y1 z0 z1 T_GLR
        profile residual flux num_line purity

    """
    # set to 0 if only 1 purity meaurement
    purity = np.zeros(len(cat))

    # Comp=0
    ksel = cat['comp'] == 0
    if np.count_nonzero(ksel) > 0:
        tglr = cat['T_GLR'][ksel]
        f = interp1d(Pval['Tval_r'], Pval['Pval_r'],
                     bounds_error=False, fill_value="extrapolate")
        purity[ksel] = f(tglr.data)

    # comp=1
    ksel = cat['comp'] == 1
    if np.count_nonzero(ksel) > 0:
        tglr = cat['STD'][ksel]
        f = interp1d(Pval_comp['Tval_r'], Pval_comp['Pval_r'],
                     bounds_error=False, fill_value="extrapolate")
        purity[ksel] = f(tglr.data)

    # The purity by definition cannot be > 1 and < 0, if the interpolation
    # gives a value outside these limits, replace by 1 or 0
    cat['purity'] = np.clip(purity, 0, 1)
    cat['purity'].format = '.3f'

    return cat


def unique_sources(table):
    """Return unique source positions in table.

    ORIGIN produces a list of lines associated to various sources identified by
    the ID column.  Some objects contain several lines found at slightly
    different positions.

    This function computes the list of unique sources averaging the RA and Dec
    of each line using the flux as weight.  The resulting table contains:

    - ID: the identifier of the source (unique);
    - ra, dec: the RA, Dec position in degrees
    - x, y: the spatial position in pixels,
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
        ra, dec, x, y, flux, seg_label, comp, merged_in, and line_merged_flag.

    Returns
    -------
    astropy.table.Table
        Table with unique sources.

    """
    table_by_id = table.group_by('ID')

    result_rows = []
    for key, group in ProgressBar(
            zip(table_by_id.groups.keys, table_by_id.groups),
            total=len(table_by_id.groups)):
        group_id = key['ID']

        ra_waverage = np.average(group['ra'], weights=group['flux'])
        dec_waverage = np.average(group['dec'], weights=group['flux'])

        x_waverage = np.average(group['x'], weights=group['flux'])
        y_waverage = np.average(group['y'], weights=group['flux'])

        n_lines = len(group[group['merged_in'] != -9999])

        seg_label = group['seg_label'][0]
        comp = group['comp'][0]  # FIXME: not necessarily true
        line_merged_flag = np.any(group["line_merged_flag"])

        result_rows.append([group_id, ra_waverage, dec_waverage, x_waverage,
                            y_waverage, n_lines, seg_label, comp,
                            line_merged_flag])

    return Table(rows=result_rows, names=["ID", "ra", "dec", "x", "y",
                                          "n_lines", "seg_label", "comp",
                                          "line_merged_flag"])


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
    table.add_column(Column(data=np.arange(len(table)), name="_idx"))

    for group in ProgressBar(table.group_by('ID').groups):
        if len(group) == 1:
            continue

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

    table['merged_in'] = np.full(len(table), -9999, dtype=int)
    for line_id, row_indexes in merge_dict.items():
        table['merged_in'][row_indexes] = line_id

    table.remove_columns('_idx')
    table.sort(['ID', 'z'])

    return table


def create_masks(line_table, source_table, profile_fwhm, cube_correl,
                 threshold_correl, cube_std, threshold_std, segmap, out_dir, *,
                 mask_size=50, seg_thres_factor=.5, plot_problems=True):
    """Create the mask of each source.

    This function creates the masks and sky masks of the sources in the line
    table using the ``origin.source_masks.gen_source_mask`` function on each
    source. The primary source masks are created using the cube_correl while
    the complementary source masks are created using the cube_std.

    The cube_correl and cube_std are expected to have the same WCS.

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
    cube_correl: mpdaf.obj.Cube
        Correlation cube where primary sources where detected.
    threshold_correl: float
        Threshold used for detection of sources in the cube_correl.
    cube_std: mpdaf.obj.Cube
        STD cube where complementary sources where detected.
    threshold_std: float
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
    logger = logging.getLogger(__name__)

    source_table = source_table.copy()
    source_table.add_index('ID')

    # The segmentation must be done at the exact position of the lines found by
    # ORIGIN (x0, y0, z0) and not the computed “optimal” position (x, y, z, ra,
    # and dec).  Using this last postion may cause problem when it falls just
    # outside the segment.  We replace ra, dec, and z by the initial values.
    line_table = line_table.copy()
    line_table['dec'], line_table['ra'] = cube_correl.wcs.pix2sky(
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

    for key, group in ProgressBar(zip(by_id.groups.keys, by_id.groups),
                                  total=len(by_id.groups)):
        source_id = key['ID']
        source_x, source_y = source_table.loc[source_id]['x', 'y']
        logger.debug("Making mask of source %s.", source_id)

        if source_table.loc[source_id]['comp'] == 0:
            detection_cube = cube_correl
            threshold = threshold_correl * seg_thres_factor
        else:
            detection_cube = cube_std
            threshold = threshold_std * seg_thres_factor

        gen_mask_return = gen_source_mask(
            source_id, source_x, source_y,
            lines=group, detection_cube=detection_cube, threshold=threshold,
            cont_sky=skymap, out_dir=out_dir, mask_size=mask_size
        )

        if gen_mask_return is not None:
            logger.warning("The source %s mask is problematic. You may want "
                           "to check source-mask-%0.5d.fits", gen_mask_return,
                           gen_mask_return)
            with open(f"{out_dir}/problematic_masks.txt", 'a') as out:
                out.write(f"{gen_mask_return}\n")
            if plot_problems:
                gen_mask_return = gen_source_mask(
                    source_id, source_x, source_y,
                    lines=group, detection_cube=detection_cube,
                    threshold=threshold, cont_sky=skymap, out_dir=out_dir,
                    mask_size=mask_size, verbose=True
                )


def compute_true_purity(cube_local_max, refcat, maxdist=4.5, threshmin=4,
                        threshmax=7, plot=False, Pval=None):
    """Compute the true purity using a reference catalog."""

    ref = Table.read(refcat)
    reflines = ref[ref['TYPE'] == 6]
    zref = cube_local_max.wave.pixel(reflines['LOBS'])
    kdref = cKDTree(np.array([reflines['Q'], reflines['P'], zref]).T)
    nref = len(ref)

    zM, yM, xM = np.where(cube_local_max._data > threshmin)
    cat0 = Table([xM, yM, zM], names=('x0', 'y0', 'z0'))
    cat0['T_GLR'] = cube_local_max._data[zM, yM, xM]

    thresh = np.arange(threshmin, threshmax, 0.1)
    res = []

    for thr in thresh:
        cat = cat0[cat0['T_GLR'] > thr]
        ndetect = len(cat)
        kdt = cKDTree(np.array([cat['x0'], cat['y0'], cat['z0']]).T)
        true = [x for x in kdt.query_ball_tree(kdref, maxdist) if x]
        ntrue = len(true)
        nmiss = nref - len(set(itertools.chain.from_iterable(true)))
        res.append((thr, ndetect, ntrue, ndetect - ntrue, nmiss))

    tbl = Table(rows=res, names=['thresh', 'ndetect', 'ntrue', 'nfalse',
                                 'nmiss'])
    tbl['purity'] = 1 - tbl['nfalse']/tbl['ndetect']

    if plot:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(tbl['thresh'], tbl['purity'], drawstyle='steps-mid',
                label='true purity')

        if Pval is None:
            print('a Pval table is required to plot the estimated purity')
        else:
            ind = (Pval['Tval_r'] >= threshmin) & (Pval['Tval_r'] <= threshmax)
            ax.plot(Pval['Tval_r'][ind], Pval['Pval_r'][ind],
                    drawstyle='steps-mid', label='estimated purity')
            # err_est_purity = (np.sqrt(Pval['Det_m']) / Pval['Det_m'] +
            #                   np.sqrt(Pval['Det_M']) / Pval['Det_m']**2)
            # ax.errorbar(Pval['Tval_r'][ind], Pval['Pval_r'][ind],
            #             err_est_purity[ind], fmt='o', label='Estimated Purity')

        ax.plot(tbl['thresh'], 1 - tbl['nmiss']/nref, drawstyle='steps-mid',
                label='completeness')
        ax.set_ylim((0, 1))
        ax.set_ylabel('purity / completeness')

        ax3 = ax.twinx()
        ax3.plot(tbl['thresh'], tbl['ntrue'], '-.', color='gray',
                 drawstyle='steps-mid')
        ax3.plot(tbl['thresh'], tbl['nfalse'], '--', color='gray',
                 drawstyle='steps-mid')
        ax3.set_yscale('log')
        fig.legend(ncol=2, loc='upper center')

    return tbl
