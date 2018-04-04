import inspect
import logging
import numpy as np
import os
import shutil
import time
import warnings

from astropy.table import vstack, Table
from collections import defaultdict
from datetime import datetime
from enum import Enum
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.sdetect import Catalog

from .lib_origin import (
    area_growing,
    area_segmentation_convex_fusion,
    area_segmentation_final,
    area_segmentation_sources_fusion,
    area_segmentation_square_fusion,
    CleanCube,
    Compute_GreedyPCA_area,
    Compute_local_max_zone,
    Compute_PCA_threshold,
    Compute_threshold_purity,
    Correlation_GLR_test,
    Create_local_max_cat,
    create_masks,
    dct_residual,
    Estimation_Line,
    merge_similar_lines,
    Purity_Estimation,
    remove_identical_duplicates,
    Spatial_Segmentation,
    trim_spectrum_list,
    unique_sources,
)


def _format_cat(cat):
    columns = {'.1f': ('flux', ),
               '.2f': ('lbda', 'T_GLR', 'STD'),
               '.3f': ('ra', 'dec', 'residual', 'purity')}
    for fmt, colnames in columns.items():
        for name in colnames:
            if name in cat.colnames:
                cat[name].format = fmt
    return cat


class LogMixin:

    def _logdebug(self, *args):
        self.logger.debug(*args)

    def _loginfo(self, *args):
        self.logger.info(*args)

    def _logwarning(self, *args):
        self.logger.warning(*args)


class Status(Enum):
    """Step processing status."""
    NOTRUN = 'not run yet'
    RUN = 'run'
    DUMPED = 'dumped outputs'
    # LOADED = 'reloaded outputs'
    FAILED = 'failed'


class Step(LogMixin):
    """Define a processing step."""

    """Name of the function to run the step."""
    name = None

    """Description of the step."""
    desc = None

    """Step requirement (not implemented yet!)."""
    require = None

    """Objects created by the processing step."""
    attrs = tuple()

    def __init__(self, orig, idx, param):
        self.logger = logging.getLogger(__name__)
        self.orig = orig
        self.idx = idx
        self.method_name = 'step%02d_%s' % (idx, self.name)
        # when a session is reloaded, use its param dict (don't overwrite it)
        self.meta = param.setdefault(self.name, {})
        self.meta.setdefault('stepidx', idx)
        self.param = self.meta.setdefault('params', {})
        self.outputs = self.meta.setdefault('outputs', defaultdict(list))
        for attr in self.attrs:
            setattr(orig, attr, None)

    def __repr__(self):
        return 'Step {:02d}: <{}(status: {})>'.format(
            self.idx, self.__class__.__name__, self.status.name)

    @property
    def status(self):
        return self.meta.get('status', Status.NOTRUN)

    @status.setter
    def status(self, val):
        self.meta['status'] = val

    def __call__(self, *args, **kwargs):
        t0 = time.time()
        self._loginfo('Step %02d - %s', self.idx, self.desc)

        sig = inspect.signature(self.run)
        for name, p in sig.parameters.items():
            if name == 'orig':  # hide the orig param
                continue
            annotation = ((' - ' + p.annotation)
                          if p.annotation is not p.empty else '')
            default = p.default if p.default is not p.empty else ''
            self._logdebug('   - %s = %r (default: %r)%s', name,
                           kwargs.get(name, ''), default, annotation)
            self.param[name] = kwargs.get(name, p.default)

        if self.require is not None:
            for req in self.require:
                step = self.orig.steps[req]
                if step.status not in (Status.RUN, Status.DUMPED):
                    raise RuntimeError('step {:02d} must be run before'
                                       .format(step.idx))

        try:
            self.run(self.orig, *args, **kwargs)
        except Exception:
            self.status = Status.FAILED
            raise
        else:
            self.status = Status.RUN

        self.meta['runtime'] = tot = time.time() - t0
        self.meta['execution_date'] = datetime.now().isoformat()
        self._loginfo('%02d Done - %.2f sec.', self.idx, tot)

    def store_cube(self, name, data, **kwargs):
        cube = Cube(data=data, wave=self.orig.wave, wcs=self.orig.wcs,
                    mask=np.ma.nomask, copy=False, **kwargs)
        setattr(self.orig, name, cube)
        self.outputs['cube'].append(name)

    def store_image(self, name, data, **kwargs):
        im = Image(data=data, wcs=self.orig.wcs, copy=False, **kwargs)
        setattr(self.orig, name, im)
        self.outputs['image'].append(name)

    def dump(self, outpath):
        if self.status is not Status.RUN:
            return

        self.logger.debug('%s - DUMP', self.method_name)
        for kind, names in self.outputs.items():
            for name in names:
                obj = getattr(self.orig, name)
                if obj is not None:
                    outf = '{}/{}'.format(outpath, name)
                    if kind in ('cube', 'image'):
                        try:
                            obj.write(outf + '.fits', convert_float32=False)
                        except TypeError:
                            warnings.warn('MPDAF version too old to support '
                                          'the new type conversion parameter, '
                                          'data will be saved as float32.')
                            obj.write(outf + '.fits')
                    elif kind in ('table', ):
                        obj.write(outf + '.fits', overwrite=True)
                    elif kind in ('array', ):
                        np.savetxt(outf + '.txt', obj)
        self.status = Status.DUMPED

    def load(self, outpath):
        if self.status is not Status.DUMPED:
            return

        self.logger.debug('%s - LOAD', self.method_name)
        for kind, names in self.outputs.items():
            for name in names:
                outf = '{}/{}.{}'.format(outpath, name,
                                         'txt' if kind == 'array' else 'fits')
                if os.path.isfile(outf):
                    if kind == 'cube':
                        obj = Cube(outf)
                    if kind == 'image':
                        obj = Image(outf)
                    elif kind == 'table':
                        obj = _format_cat(Table.read(outf))
                    elif kind == 'array':
                        obj = np.loadtxt(outf, ndmin=1)
                else:
                    obj = None
                setattr(self.orig, name, obj)
        # self.status = Status.LOADED


class Preprocessing(Step):
    """ Preprocessing of data, dct, standardization and noise compensation

    Parameters
    ----------
    dct_order : int
        The number of atom to keep for the dct decomposition.
    dct_approx : bool
        if True, the DCT computation does not take the variance into account
        for the computation of the DCT coefficients.

    Returns
    -------
    self.cube_std : `~mpdaf.obj.Cube`
        standardized data for PCA.
    self.cont_dct : `~mpdaf.obj.Cube`
        continuum estimated with a DCT.
    self.ima_std : `~mpdaf.obj.Image`
        Mean of standardized data cube.
    self.ima_dct : `~mpdaf.obj.Image`
        Mean of DCT continuum cube.

    """

    name = 'preprocessing'
    desc = 'Preprocessing'
    attrs = ('cube_std', 'cube_dct')

    def run(self, orig, dct_order=10, dct_approx=False):
        self._loginfo('DCT computation')
        cont_dct = dct_residual(orig.cube_raw, dct_order, orig.var, dct_approx)
        data = orig.cube_raw - cont_dct
        data[orig.mask] = np.nan

        # compute standardized data
        self._loginfo('Data standardizing')
        std = np.sqrt(orig.var)
        cont_dct /= std

        mean = np.nanmean(data, axis=(1, 2))
        # orig.var[orig.mask] = np.inf
        data -= mean[:, np.newaxis, np.newaxis]
        data /= std
        data[orig.mask] = 0

        self._loginfo('Std signal saved in self.cube_std and self.ima_std')
        self.store_cube('cube_std', data)
        self.store_image('ima_std', data.mean(axis=0))
        self._loginfo('DCT continuum saved in self.cont_dct and self.ima_dct')
        self.store_cube('cont_dct', cont_dct)
        self.store_image('ima_dct', cont_dct.mean(axis=0))


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
    attrs = ('areamap', )

    def run(self, orig, pfa: "pfa of the test"=.2,
            minsize: "min area size"=100, maxsize: "max area size"=None):
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

        areamap = areamap.astype(int)
        labels = np.unique(areamap)
        if 0 in labels:  # expmap=0
            nbAreas = len(labels) - 1
        else:
            nbAreas = len(labels)
        orig.param['nbareas'] = nbAreas

        self.store_image('areamap', areamap)
        self._loginfo('Save the map of areas in self.areamap')
        self._loginfo('%d areas generated', nbAreas)


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
    attrs = ('threshO2', 'testO2', 'histO2', 'binO2', 'meaO2', 'stdO2')
    require = ('preprocessing', 'areas')

    def run(self, orig, pfa_test: 'pfa of the test'=.01):
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
        # FIXME: test02, histO2 and binO2 are lists of arrays with variable
        # sizes so they cannot be managed by the Step class currently
        self.outputs['array'].extend(['thresO2', 'meaO2', 'stdO2'])
        # self.outputs['array'].extend(['testO2', 'histO2', 'binO2'])


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
    attrs = ('cube_faint', 'mapO2')
    require = ('preprocessing', 'areas', 'compute_PCA_threshold')

    def run(self, orig, Noise_population=50,
            itermax: 'Max number of iterations'=100, threshold_list=None):
        thr = orig.thresO2 if threshold_list is None else threshold_list
        orig.param['threshold_list'] = thr
        self._loginfo('   - List of threshold = %s',
                      ' '.join("%.2f" % x for x in thr))
        self._loginfo('Compute greedy PCA on each zone')
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
    attrs = ('cube_correl', 'cube_profile', 'cube_local_min',
             'cube_local_max', 'maxmap')
    require = ('compute_greedy_PCA', )

    def run(self, orig, NbSubcube=1, neighbors=26, ncpu=1):
        if ncpu > 1:
            try:
                import mkl_fft  # noqa
            except ImportError:
                pass
            else:
                warnings.warn('using multiprocessing (ncpu>1) is not possible '
                              'with the mkl_fft package, it will crash')

        # TGLR computing (normalized correlations)
        self._loginfo('Correlation')
        inty, intx = Spatial_Segmentation(orig.Nx, orig.Ny, NbSubcube)
        correl, profile, correl_min = Correlation_GLR_test(
            orig.cube_faint._data, orig.var, orig.PSF, orig.wfields,
            orig.profiles, ncpu)

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
            correl, correl_min, orig.mask, intx, inty, NbSubcube, neighbors)
        self._loginfo('Save self.cube_local_max from max correlations')
        self.store_cube('cube_local_max', cube_local_max)
        self._loginfo('Save self.cube_local_min from min correlations')
        self.store_cube('cube_local_min', cube_local_min)


class ComputePurityThreshold(Step):
    """Find the threshold  for a given purity

    Parameters
    ----------
    purity : float
        purity to automatically compute the threshold
    tol_spat : int
        spatial tolerance for the spatial merging (distance in pixels)
        TODO en fonction du FWHM
    tol_spec : int
        spectral tolerance for the spatial merging (distance in pixels)
    spat_size : int
        spatiale size of the spatiale filter
    spect_size : int
        spectral lenght of the spectral filter
    auto : tuple (npts1,npts2,pmargin)
        nb of threshold sample for iteration 1 and 2, margin in purity
        default (5,15,0.1
    threshlist : list
        list of thresholds to compute the purity

    Returns
    -------
    self.threshold_correl : float
        Estimated threshold
    self.Pval_r : array
        Purity curves
    self.index_pval : array
        Indexes of the purity curves
    self.Det_M : list
        Number of detections in +DATA
    self.Det_m : list
        Number of detections in -DATA

    """

    name = 'compute_purity_threshold'
    desc = 'Compute Purity threshold'
    attrs = ('Pval_r', 'index_pval', 'Det_M', 'Det_m')
    require = ('compute_TGLR', )

    def run(self, orig, purity=.9, tol_spat=3, tol_spec=5, spat_size=19,
            spect_size=10, auto=(5, 15, 0.1), threshlist=None):
        orig.param['purity'] = purity
        self._loginfo('Estimation of threshold with purity = %.2f', purity)
        threshold, orig.Pval_r, orig.index_pval, orig.Det_m, orig.Det_M = \
            Compute_threshold_purity(purity, orig.cube_local_max.data,
                                     orig.cube_local_min.data, orig.segmap.data,
                                     spat_size, spect_size, tol_spat, tol_spec,
                                     True, True, auto, threshlist)
        orig.param['threshold'] = threshold
        self._loginfo('Threshold: %.2f ', threshold)
        self.outputs['array'].extend(['Pval_r', 'index_pval', 'Det_M',
                                      'Det_m'])


class Detection(Step):
    """Detections on local maxima from max correlation + spatia-spectral
    merging in order to create the first catalog.

    Parameters
    ----------
    threshold : float
        User threshod if the estimated threshold is not good

    Returns
    -------
    self.Cat0 : astropy.Table
        First catalog. Columns: ID ra dec lbda x0 y0 z0 profile seg_label T_GLR
    self.det_correl_min : (array, array, array)
        3D positions of detections in correl_min

    """

    name = 'detection'
    desc = 'Thresholding and spatio-spectral merging'
    attrs = ('Cat0', 'det_correl_min')

    def run(self, orig, threshold=None):
        if threshold is not None:
            orig.param['threshold'] = threshold

        pur_params = orig.param['compute_purity_threshold']['params']
        orig.Cat0, orig.det_correl_min = Create_local_max_cat(
            orig.param['threshold'], orig.cube_local_max.data,
            orig.cube_local_min.data, orig.segmap.data,
            pur_params['spat_size'], pur_params['spect_size'],
            pur_params['tol_spat'], pur_params['tol_spec'],
            True, orig.cube_profile._data, orig.wcs, orig.wave
        )
        _format_cat(orig.Cat0)
        self._loginfo('Save the catalogue in self.Cat0 (%d sources %d lines)',
                      len(np.unique(orig.Cat0['ID'])), len(orig.Cat0))
        self.outputs['table'].append('Cat0')
        self.outputs['array'].append('det_correl_min')


class DetectionLost(Step):
    """Detections on local maxima of std cube + spatia-spectral
    merging in order to create an complementary catalog. This catalog is
    merged with the catalog Cat0 in order to create the catalog Cat1

    Parameters
    ----------
    purity : float
        purity to automatically compute the threshold
        If None, previous purity is used
    auto : tuple (npts1,npts2,pmargin)
        nb of threshold sample for iteration 1 and 2, margin in purity
        default (5,15,0.1)
    threshlist : list
        list of thresholds to compute the purity default None

    Returns
    -------
    self.threshold_correl : float
        Estimated threshold used to detect complementary
        lines on local maxima of std cube
    self.Pval_r_comp : array
        Purity curves
    self.index_pval_comp : array
        Indexes of the purity curves
    self.Det_M_comp : list
        Number of detections in +DATA
    self.Det_m_comp : list
        Number of detections in -DATA
    self.Cat1 : astropy.Table
        New catalog.
        Columns: ID ra dec lbda x0 y0 z0 profile seg_label T_GLR STD comp

    """

    name = 'detection_lost'
    desc = 'Thresholding and spatio-spectral merging'
    attrs = ('Cat1', 'Pval_r_comp', 'index_pval_comp', 'Det_M_comp',
             'Det_m_comp')
    require = ('detection', )

    def run(self, orig, purity=None, auto=(5, 15, 0.1), threshlist=None):
        self._loginfo('Compute local maximum of std cube values')
        NbSubcube = orig.param['compute_TGLR']['params']['NbSubcube']
        neighbors = orig.param['compute_TGLR']['params']['neighbors']
        inty, intx = Spatial_Segmentation(orig.Nx, orig.Ny, NbSubcube)
        cube_local_max_faint_dct, cube_local_min_faint_dct = \
            Compute_local_max_zone(orig.cube_std.data, orig.cube_std.data,
                                   orig.mask, intx, inty, NbSubcube, neighbors)

        pur_params = orig.param['compute_purity_threshold']['params']

        # complementary catalog
        cube_local_max_faint_dct, cube_local_min_faint_dct = CleanCube(
            cube_local_max_faint_dct, cube_local_min_faint_dct,
            orig.Cat0, orig.det_correl_min, orig.Nz, orig.Nx, orig.Ny,
            pur_params['spat_size'], pur_params['spect_size'])

        if purity is None:
            purity = pur_params['purity']
        orig.param['purity2'] = purity

        self._loginfo('Threshold computed with purity = %.1f', purity)

        orig.cube_local_max_faint_dct = cube_local_max_faint_dct
        orig.cube_local_min_faint_dct = cube_local_min_faint_dct

        threshold2, orig.Pval_r_comp, orig.index_pval_comp, orig.Det_m_comp, \
            orig.Det_M_comp = Compute_threshold_purity(
                purity,
                cube_local_max_faint_dct,
                cube_local_min_faint_dct,
                orig.segmap._data,
                pur_params['spat_size'],
                pur_params['spect_size'],
                pur_params['tol_spat'],
                pur_params['tol_spec'],
                True, False,
                auto, threshlist)
        orig.param['threshold2'] = threshold2
        self._loginfo('Threshold: %.2f ', threshold2)

        if threshold2 == np.inf:
            orig.Cat1 = orig.Cat0.copy()
            orig.Cat1['comp'] = 0
            orig.Cat1['STD'] = 0
        else:
            Catcomp, _ = Create_local_max_cat(threshold2,
                                              cube_local_max_faint_dct,
                                              cube_local_min_faint_dct,
                                              orig.segmap._data,
                                              pur_params['spat_size'],
                                              pur_params['spect_size'],
                                              pur_params['tol_spat'],
                                              pur_params['tol_spec'],
                                              True,
                                              orig.cube_profile._data,
                                              orig.wcs, orig.wave)
            Catcomp.rename_column('T_GLR', 'STD')
            # merging
            Cat0 = orig.Cat0.copy()
            Cat0['comp'] = 0
            Catcomp['comp'] = 1
            Catcomp['ID'] += (Cat0['ID'].max() + 1)
            orig.Cat1 = _format_cat(vstack([Cat0, Catcomp]))

        ns = len(np.unique(orig.Cat1['ID']))
        ds = ns - len(np.unique(orig.Cat0['ID']))
        nl = len(orig.Cat1)
        dl = nl - len(orig.Cat0)
        self._loginfo('Save the catalogue in self.Cat1'
                      ' (%d [+%s] sources %d [+%d] lines)', ns, ds, nl, dl)

        self.outputs['table'].append('Cat1')
        self.outputs['array'].extend(['Pval_r_comp', 'index_pval_comp',
                                      'Det_M_comp', 'Det_m_comp'])


class ComputeSpectra(Step):
    """Compute the estimated emission line and the optimal coordinates

    for each detected lines in a spatio-spectral grid (each emission line
    is estimated with the deconvolution model ::

        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))

    Via PCA LS or denoised PCA LS Method

    Parameters
    ----------
    grid_dxy : int
        Maximum spatial shift for the grid

    Returns
    -------
    self.Cat2 : astropy.Table
        Catalogue of parameters of detected emission lines.
        Columns: ra dec lbda x0 x y0 y z0 z T_GLR profile
        residual flux num_line purity
    self.spectra : list of `~mpdaf.obj.Spectrum`
        Estimated lines

    """

    name = 'compute_spectra'
    desc = 'Lines estimation'
    attrs = ('Cat2', 'spectra')
    require = ('detection_lost', )

    def run(self, orig, grid_dxy=0):
        orig.Cat2, Cat_est_line_raw_T, Cat_est_line_var_T = Estimation_Line(
            orig.Cat1, orig.cube_raw, orig.var, orig.PSF,
            orig.wfields, orig.wcs, orig.wave, size_grid=grid_dxy,
            criteria='flux', order_dct=30, horiz_psf=1, horiz=5
        )

        self._loginfo('Purity estimation')
        orig.Cat2 = Purity_Estimation(orig.Cat2,
                                      [orig.Pval_r, orig.Pval_r_comp],
                                      [orig.index_pval, orig.index_pval_comp])
        _format_cat(orig.Cat2)
        self._loginfo('Save the updated catalogue in self.Cat2 (%d lines)',
                      len(orig.Cat2))

        orig.spectra = [
            Spectrum(data=data, var=vari, wave=orig.wave, mask=np.ma.nomask)
            for data, vari in zip(Cat_est_line_raw_T, Cat_est_line_var_T)
        ]
        self._loginfo('Save estimated spectrum of each line in self.spectra')
        self.outputs['table'].append('Cat2')


class CleanResults(Step):
    """Clean the various results.

    This step does several things to “clean” the results of ORIGIN:

    - The Cat2 line table may contain several lines found at the very same
      x, y, z position in the cube. Only the line with the highest purity
      is kept in the table.
    - Some lines are associated to the same source but are very near
      considering their z positions.  The lines are all marked as merged in
      the brightest line of the group (but are kept in the line table).
    - The FITS file containing the spectra is cleaned to keep only the
      lines from the cleaned line table. The spectrum around each line
      is trimmed around the line position.
    - A table of unique sources is created.

    Attributes added to the ORIGIN object:
    - `Cat3_lines`: clean table of lines;
    - `Cat3_sources`: table of unique sources
    - `Cat3_spectra`: trimmed spectra. For a given <num_line>, the
      spectrum is in `DATA<num_line>` extension and the variance in
      the `STAT<num_line>` extension.

    Parameters
    ----------
    merge_lines_z_threshold: int
        z axis pixel threshold used when merging similar lines.
    spectrum_size_fwhm: float
        The length of the spectrum to keep around each line as a factor of
        the fitted line FWHM.

    """

    name = 'clean_results'
    desc = 'Results cleaning'
    attrs = ('Cat3_lines', 'Cat3_sources', 'Cat3_spectra')
    require = ('compute_spectra', )

    def run(self, orig, merge_lines_z_threshold=5, spectrum_size_fwhm=3):
        unique_lines = remove_identical_duplicates(orig.Cat2)
        orig.Cat3_lines = merge_similar_lines(unique_lines)
        orig.Cat3_sources = unique_sources(orig.Cat3_lines)

        self._loginfo('Save the unique source catalogue in self.Cat3_sources'
                      ' (%d lines)', len(orig.Cat3_sources))
        self._loginfo('Save the cleaned lines in self.Cat3_lines (%d lines)',
                      len(orig.Cat3_lines))

        orig.Cat3_spectra = trim_spectrum_list(
            orig.Cat3_lines, orig.spectra, orig.FWHM_profiles,
            size_fwhm=spectrum_size_fwhm)

        self.outputs['table'].extend(['Cat3_lines', 'Cat3_sources'])


class CreateMasks(Step):
    """Create source masks and sky masks.

    This step create the mask and sky mask for each source.

    Parameters
    ----------
    path : str
        Path where the masks will be saved.
    overwrite : bool
        Overwrite the folder if it already exists
    mask_size: int
        Widht in pixel for the square masks.
    seg_thres_factor: float
        Factor applied to the detection threshold to get the threshold used
        for mask creation.

    """

    name = 'create_masks'
    desc = 'Mask creation'
    require = ('clean_results', )

    def run(self, orig, path=None, overwrite=True, mask_size=50,
            seg_thres_factor=.5):
        if path is None:
            out_dir = '%s/masks' % orig.outpath
        else:
            if os.path.exists(path):
                raise IOError("Invalid path: {0}".format(path))
            path = os.path.normpath(path)
            out_dir = '%s/%s/masks' % (path, orig.name)

        if overwrite:
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)

        orig.param['mask_filename_tpl'] = f"{out_dir}/source-mask-%0.5d.fits"
        orig.param['skymask_filename_tpl'] = f"{out_dir}/sky-mask-%0.5d.fits"

        create_masks(
            line_table=orig.Cat3_lines,
            source_table=orig.Cat3_sources,
            profile_fwhm=orig.FWHM_profiles,
            cube_correl=orig.cube_correl,
            threshold_correl=orig.threshold_correl,
            cube_std=orig.cube_std,
            threshold_std=orig.threshold_std,
            segmap=orig.segmap,
            out_dir=out_dir,
            mask_size=mask_size,
            seg_thres_factor=seg_thres_factor,
            plot_problems=True)


class SaveSources(Step):
    """Create the source file for each source.

    Parameters
    ----------
    version: str
        Version number of the source files.
    path: str
        Path where the sources will be saved.
    n_job: int
        Number of jobs for parallel processing.
    author: str
        Name of the author to add in the sources.
    nb_fwhm: float
        Factor multiplying the FWHM of a line to compute the width of the
        associated narrow band image.
    size: float
        Side of the square used for cut-outs around the source position
        (for images and sub-cubes) in arc-seconds.
    expmap_filename: str
        Name of the file containing the exposure map to add to the source.
    fieldmap_filename: str
        Name of the file containing the fieldmap.
    overwrite: bool
        Overwrite the folder if it already exists.

    """

    name = 'save_sources'
    desc = 'Save sources'

    def run(self, orig, version, *, path=None, n_jobs=1, author="",
            nb_fwhm=2, size=5, expmap_filename=None, fieldmap_filename=None,
            overwrite=True):

        if path is None:
            outpath = orig.outpath
        else:
            if not os.path.exists(path):
                raise IOError("Invalid path: {0}".format(path))
            outpath = os.path.join(os.path.normpath(path), orig.name)
        out_dir = os.path.join(outpath, 'sources')
        catname = os.path.join(outpath, '%s.fits' % orig.name)

        if overwrite:
            shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)

        # FIXME: We need to have the file containing the spectra saved for the
        # create_all_sources function.

        from .source_creation import create_all_sources
        create_all_sources(
            cat3_sources=orig.Cat3_sources,
            cat3_lines=orig.Cat3_lines,
            origin_params=orig.param,
            cube_cor_filename=os.path.join(outpath, 'cube_correl.fits'),
            mask_filename_tpl=orig.param['mask_filename_tpl'],
            skymask_filename_tpl=orig.param['skymask_filename_tpl'],
            spectra_fits_filename=os.path.join(outpath, 'Cat3_spectra.fits'),
            version=version,
            profile_fwhm=orig.FWHM_profiles,
            out_tpl=os.path.join(out_dir, 'source-%0.5d.fits'),
            n_jobs=n_jobs,
            author=author,
            nb_fwhm=nb_fwhm,
            size=size,
            expmap_filename=expmap_filename,
            fieldmap_filename=fieldmap_filename,
        )

        # create the final catalog
        self._loginfo('Create the final catalog...')
        catF = Catalog.from_path(out_dir, fmt='working')
        catF.write(catname, overwrite=overwrite)


"""This defines the list of all processing steps."""
STEPS = [
    Preprocessing,
    CreateAreas,
    ComputePCAThreshold,
    ComputeGreedyPCA,
    ComputeTGLR,
    ComputePurityThreshold,
    Detection,
    DetectionLost,
    ComputeSpectra,
    CleanResults,
    CreateMasks,
    SaveSources
]
