import inspect
import logging
import numpy as np
import time
# from astropy.utils import lazyproperty
from mpdaf.obj import Cube, Image


class LogMixin:

    def _logdebug(self, *args):
        self.logger.debug(*args)

    def _loginfo(self, *args):
        self.logger.info(*args)

    def _logwarning(self, *args):
        self.logger.warning(*args)


class Step(LogMixin):
    """Define a processing step."""

    name = None
    desc = None
    require = None

    def __init__(self, orig, idx, param):
        self.orig = orig
        self.idx = idx
        self.method_name = 'step%02d_%s' % (idx, self.name)
        self.param = param[self.method_name] = {}
        self.logger = logging.getLogger(__name__)

    def __call__(self, *args, **kwargs):
        t0 = time.time()
        info = self._loginfo
        info('Step %02d - %s', self.idx, self.desc)

        sig = inspect.signature(self.run)
        for name, p in sig.parameters.items():
            if name == 'orig':
                continue
            annotation = ((' - ' + p.annotation)
                          if p.annotation is not p.empty else '')
            default = p.default if p.default is not p.empty else ''
            info('   - %s = %r (default: %r)%s', name,
                 kwargs.get(name, ''), default, annotation)
            self.param[name] = kwargs.get(name, p.default)

        self.run(self.orig, *args, **kwargs)
        tot = time.time() - t0
        info('%02d Done - %.2f sec.', self.idx, tot)

    def new_cube(self, data, **kwargs):
        return Cube(data=data, wave=self.orig.wave, wcs=self.orig.wcs,
                    mask=np.ma.nomask, copy=False, **kwargs)

    def new_image(self, data, **kwargs):
        return Image(data=data, wcs=self.orig.wcs, copy=False, **kwargs)


class Preprocessing(Step):
    """ Preprocessing of data, dct, standardization and noise compensation

    Parameters
    ----------
    dct_order : int
        The number of atom to keep for the dct decomposition
    dct_approx : bool
        if True, the DCT computation is approximated

    Returns
    -------
    self.cube_std : `~mpdaf.obj.Cube`
        standardized data for PCA
    self.cont_dct : `~mpdaf.obj.Cube`
        DCT continuum
    self.ima_std : `~mpdaf.obj.Image`
        Mean of standardized data for PCA along the wavelength axis
    self.ima_dct : `~mpdaf.obj.Image`
        Mean of DCT continuum cube along the wavelength axis

    """

    name = 'preprocessing'
    desc = 'Preprocessing'

    def run(self, orig, dct_order=10, dct_approx=True):
        from .lib_origin import dct_residual, Compute_Standardized_data

        self._loginfo('DCT computation')
        faint_dct, cont_dct = dct_residual(orig.cube_raw, dct_order, orig.var,
                                           dct_approx)

        # compute standardized data
        self._loginfo('Data standardizing')
        cube_std = Compute_Standardized_data(faint_dct, orig.mask, orig.var)
        cont_dct /= np.sqrt(orig.var)

        self._loginfo('Std signal saved in self.cube_std and self.ima_std')
        orig.cube_std = self.new_cube(cube_std)
        self._loginfo('DCT continuum saved in self.cont_dct and self.ima_dct')
        orig.cont_dct = self.new_cube(cont_dct)


class Areas(Step):
    """ Creation of automatic area

    Parameters
    ----------
    pfa : float
        PFA of the segmentation test to estimates sources with
        strong continuum
    minsize : int
        Lenght in pixel of the side of typical surface wanted
        enough big area to satisfy the PCA
    maxsize : int
        Lenght in pixel of the side of maximum surface wanted

    Returns
    -------
    self.nbAreas : int
        number of areas
    self.areamap : `~mpdaf.obj.Image`
        The map of areas

    """

    name = 'areas'
    desc = 'Areas creation'

    def run(self, orig, pfa: "pfa of the test"=.2,
            minsize: "minimum size"=100, maxsize=None):
        from .lib_origin import (area_segmentation_square_fusion,
                                 area_segmentation_sources_fusion,
                                 area_segmentation_convex_fusion,
                                 area_segmentation_final, area_growing)

        self.param['pfa_areas'] = pfa
        self.param['minsize_areas'] = minsize
        self.param['maxsize_areas'] = maxsize

        nexpmap = (np.sum(~orig.mask, axis=0) > 0).astype(np.int)

        NbSubcube = np.maximum(1, int(np.sqrt(np.sum(nexpmap) / (minsize**2))))
        if NbSubcube > 1:
            if maxsize is None:
                maxsize = minsize * 2

            MinSize = minsize**2
            MaxSize = maxsize**2

            self._loginfo('First segmentation of %d^2 square', NbSubcube)
            self._logdebug('Squares segmentation and fusion')
            square_cut_fus = area_segmentation_square_fusion(
                nexpmap, MinSize, MaxSize, NbSubcube, orig.Ny, orig.Nx)

            self._logdebug('Sources fusion')
            square_src_fus, src = area_segmentation_sources_fusion(
                orig.segmap.data, square_cut_fus, pfa, orig.Ny, orig.Nx)

            self._logdebug('Convex envelope')
            convex_lab = area_segmentation_convex_fusion(square_src_fus, src)

            self._logdebug('Areas dilation')
            Grown_label = area_growing(convex_lab, nexpmap)

            self._logdebug('Fusion of small area')
            self._logdebug('Minimum Size: %d px' % MinSize)
            self._logdebug('Maximum Size: %d px' % MaxSize)
            areamap = area_segmentation_final(Grown_label, MinSize, MaxSize)

        elif NbSubcube == 1:
            areamap = nexpmap

        self._loginfo('Save the map of areas in self.areamap')

        orig.areamap = self.new_image(areamap.astype(int))
        self._loginfo('%d areas generated' % orig.nbAreas)
        self.param['nbareas'] = orig.nbAreas


class PCAThreshold(Step):
    """ Loop on each zone of the data cube and estimate the threshold

    Parameters
    ----------
    pfa_test : float
        Threshold of the test (default=0.01)

    Returns
    -------
    self.testO2 : list of arrays (one per PCA area)
        Result of the O2 test.
    self.histO2 : lists of arrays (one per PCA area)
        PCA histogram
    self.binO2 : lists of arrays (one per PCA area)
        bin for the PCA histogram
    self.thresO2 : list of float
        For each area, threshold value
    self.meaO2 : list of float
        Location parameter of the Gaussian fit used to estimate the threshold
    self.stdO2 : list of float
        Scale parameter of the Gaussian fit used to estimate the threshold

    """

    name = 'compute_PCA_threshold'
    desc = 'PCA threshold computation'

    def run(self, orig, pfa_test: 'pfa of the test'=.01):
        from .lib_origin import Compute_PCA_threshold

        if orig.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
        if orig.areamap is None:
            raise IOError('Run the step 02 to initialize self.areamap ')

        results = []

        for area_ind in range(1, orig.nbAreas + 1):
            # limits of each spatial zone
            ksel = (orig.areamap._data == area_ind)

            # Data in this spatio-spectral zone
            cube_temp = orig.cube_std._data[:, ksel]

            res = Compute_PCA_threshold(cube_temp, pfa_test)
            results.append(res)
            self._loginfo('Area %d, estimation mean/std/threshold: %f/%f/%f',
                          area_ind, res[4], res[5], res[3])

        (orig.testO2, orig.histO2, orig.binO2, orig.thresO2, orig.meaO2,
         orig.stdO2) = zip(*results)


class GreedyPCA(Step):
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
    Noise_population : float
        Fraction of spectra used to estimate the background signature
    itermax : int
        Maximum number of iterations
    threshold_list : list
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
        eigenvalues of the data cube (representing the faint signal)
    self.mapO2 : `~mpdaf.obj.Image`
        The numbers of iterations used by testO2 for each spaxel

    """

    name = 'compute_greedy_PCA'
    desc = 'Greedy PCA computation'

    def run(self, orig, Noise_population=50,
            itermax: 'Max number of iterations'=100, threshold_list=None):
        if orig.cube_std is None:
            raise IOError('Run the step 01 to initialize self.cube_std')
        if orig.areamap is None:
            raise IOError('Run the step 02 to initialize self.areamap')
        if threshold_list is None:
            if orig.thresO2 is None:
                raise IOError('Run the step 03 to initialize self.thresO2')
            thr = orig.thresO2
        else:
            thr = threshold_list

        # self._loginfo('   - Noise_population = %0.2f' % Noise_population)
        self._loginfo('   - List of threshold = ' +
                      " ".join("%.2f" % x for x in thr))

        # self.param['threshold_list'] = thr
        # self.param['Noise_population'] = Noise_population
        # self.param['itermax'] = itermax

        self._loginfo('Compute greedy PCA on each zone')
        from .lib_origin import Compute_GreedyPCA_area

        faint, mapO2, nstop = Compute_GreedyPCA_area(
            orig.nbAreas, orig.cube_std._data, orig.areamap._data,
            Noise_population, thr, itermax, orig.testO2)
        if nstop > 0:
            self._logwarning('The iterations have been reached the limit '
                             'of %d in %d cases', itermax, nstop)

        self._loginfo('Save the faint signal in self.cube_faint')
        orig.cube_faint = self.new_cube(faint)
        self._loginfo('Save the numbers of iterations used by the'
                      ' testO2 for each spaxel in self.mapO2')
        orig.mapO2 = self.new_image(mapO2)


pipeline = [
    Preprocessing,
    Areas,
    PCAThreshold,
    GreedyPCA
]
