import inspect
import logging
import numpy as np
import time
# from astropy.utils import lazyproperty
from enum import Enum
from mpdaf.obj import Cube, Image

# TODO:
# - Add execution datetime
# - manage requirements between steps
# - save and update params


class LogMixin:

    def _logdebug(self, *args):
        self.logger.debug(*args)

    def _loginfo(self, *args):
        self.logger.info(*args)

    def _logwarning(self, *args):
        self.logger.warning(*args)


class Status(Enum):
    NOTRUN = 1
    RUN = 2
    DUMPED = 3
    LOADED = 4
    FAILED = 5


class Step(LogMixin):
    """Define a processing step."""

    """Name of the function to run the step."""
    name = None

    """Description of the step."""
    desc = None

    """Step requirement (not implemented yet!)."""
    require = None

    def __init__(self, orig, idx, param):
        self.logger = logging.getLogger(__name__)
        self.orig = orig
        self.idx = idx
        self.method_name = 'step%02d_%s' % (idx, self.name)
        self.param = param[self.method_name] = {}
        self.outputs = {}
        self.status = Status.NOTRUN

    def __repr__(self):
        return '<{}(status: {})>'.format(self.__class__.__name__,
                                         self.status.name)

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

        try:
            self.run(self.orig, *args, **kwargs)
        except Exception:
            self.status = Status.FAILED
            raise
        else:
            self.status = Status.RUN

        tot = time.time() - t0
        info('%02d Done - %.2f sec.', self.idx, tot)

    def store_cube(self, name, data, **kwargs):
        cube = Cube(data=data, wave=self.orig.wave, wcs=self.orig.wcs,
                    mask=np.ma.nomask, copy=False, **kwargs)
        setattr(self.orig, name, cube)
        self.outputs[name] = {'type': 'cube', 'obj': cube}

    def store_image(self, name, data, **kwargs):
        im = Image(data=data, wcs=self.orig.wcs, copy=False, **kwargs)
        setattr(self.orig, name, im)
        self.outputs[name] = {'type': 'image', 'obj': im}

    def dump(self, outpath):
        if self.status is not Status.RUN:
            self.logger.debug('%s - nothing to dump', self.method_name)
            return
        for name, out in self.outputs.items():
            if out['type'] in ('cube', 'image'):
                obj = getattr(self.orig, name)
                if obj is not None:
                    obj.write('{}/{}.fits'.format(outpath, name))
        self.status = Status.DUMPED


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
        self.store_cube('cube_std', cube_std)
        self._loginfo('DCT continuum saved in self.cont_dct and self.ima_dct')
        self.store_cube('cont_dct', cont_dct)


class CreateAreas(Step):
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

        # TODO: remove this and change in source creation
        orig.param['pfa_areas'] = pfa
        orig.param['minsize_areas'] = minsize
        orig.param['maxsize_areas'] = maxsize

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
            self._logdebug('Minimum Size: %d px', MinSize)
            self._logdebug('Maximum Size: %d px', MaxSize)
            areamap = area_segmentation_final(Grown_label, MinSize, MaxSize)

        elif NbSubcube == 1:
            areamap = nexpmap

        self._loginfo('Save the map of areas in self.areamap')
        self.store_image('areamap', areamap.astype(int))
        self._loginfo('%d areas generated', orig.nbAreas)
        orig.param['nbareas'] = orig.nbAreas


class ComputePCAThreshold(Step):
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


class ComputeGreedyPCA(Step):
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

        # orig.param['threshold_list'] = thr
        # orig.param['Noise_population'] = Noise_population
        # orig.param['itermax'] = itermax

        self._loginfo('Compute greedy PCA on each zone')
        from .lib_origin import Compute_GreedyPCA_area

        faint, mapO2, nstop = Compute_GreedyPCA_area(
            orig.nbAreas, orig.cube_std._data, orig.areamap._data,
            Noise_population, thr, itermax, orig.testO2)
        if nstop > 0:
            self._logwarning('The iterations have been reached the limit '
                             'of %d in %d cases', itermax, nstop)

        self._loginfo('Save the faint signal in self.cube_faint')
        self.store_cube('cube_faint', faint)
        self._loginfo('Save the numbers of iterations used by the'
                      ' testO2 for each spaxel in self.mapO2')
        self.store_image('mapO2', mapO2)


class ComputeTGLR(Step):
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
      number of thresholded p-values of the correlations per spectral channel.

    Parameters
    ----------
    NbSubcube : int
        Number of sub-cubes for the spatial segmentation
        If NbSubcube>1 the correlation and local maximas and
        minimas are performed on smaller subcube and combined
        after. Useful to avoid swapp
    neighbors : int
        Connectivity of contiguous voxels
    ncpu : int
        Number of CPUs used

    Returns
    -------
    self.cube_correl : `~mpdaf.obj.Cube`
        Cube of T_GLR values
    self.cube_profile : `~mpdaf.obj.Cube` (type int)
        Number of the profile associated to the T_GLR
    self.maxmap : `~mpdaf.obj.Image`
        Map of maxima along the wavelength axis
    self.cube_local_max : `~mpdaf.obj.Cube`
        Local maxima from max correlation
    self.cube_local_min : `~mpdaf.obj.Cube`
        Local maxima from minus min correlation

    """

    name = 'compute_TGLR'
    desc = 'GLR test'

    def run(self, orig, NbSubcube=1, neighbors=26, ncpu=4):
        if orig.cube_faint is None:
            raise IOError('Run the step 04 to initialize self.cube_faint')

        # orig.param['neighbors'] = neighbors
        # orig.param['NbSubcube'] = NbSubcube
        from .lib_origin import (Spatial_Segmentation, Correlation_GLR_test,
                                 Correlation_GLR_test_zone,
                                 Compute_local_max_zone)

        # TGLR computing (normalized correlations)
        self._loginfo('Correlation')
        inty, intx = Spatial_Segmentation(orig.Nx, orig.Ny, NbSubcube)
        if NbSubcube == 1:
            correl, profile, cm = Correlation_GLR_test(
                orig.cube_faint._data, orig.var, orig.PSF, orig.wfields,
                orig.profiles, ncpu)
        else:
            correl, profile, cm = Correlation_GLR_test_zone(
                orig.cube_faint._data, orig.var, orig.PSF, orig.wfields,
                orig.profiles, intx, inty, NbSubcube, ncpu)

        self._loginfo('Save the TGLR value in self.cube_correl')
        correl[orig.mask] = 0
        self.store_cube('cube_correl', correl)

        self._loginfo('Save the number of profile associated to the TGLR'
                      ' in self.cube_profile')
        profile[orig.mask] = 0
        self.store_cube('cube_profile', profile.astype(int))

        self._loginfo('Save the map of maxima in self.maxmap')
        carte_2D_correl = np.amax(orig.cube_correl._data, axis=0)
        self.store_image('maxmap', carte_2D_correl)

        self._loginfo('Compute p-values of local maximum of correlation '
                      'values')
        cube_local_max, cube_local_min = Compute_local_max_zone(
            correl, cm, orig.mask, intx, inty, NbSubcube, neighbors)
        self._loginfo('Save self.cube_local_max from max correlations')
        self.store_cube('cube_local_max', cube_local_max)
        self._loginfo('Save self.cube_local_min from min correlations')
        self.store_cube('cube_local_min', cube_local_min)


pipeline = [
    Preprocessing,
    CreateAreas,
    ComputePCAThreshold,
    ComputeGreedyPCA,
    ComputeTGLR
]
