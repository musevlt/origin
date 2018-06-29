import inspect
import logging
import numpy as np
import os
import shutil
import time
import warnings

from astropy.io import fits
from astropy.table import vstack, Table
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from mpdaf.obj import Cube, Image, Spectrum
from mpdaf.sdetect import Catalog
from scipy import ndimage as ndi

from .lib_origin import (
    area_growing,
    area_segmentation_convex_fusion,
    area_segmentation_final,
    area_segmentation_sources_fusion,
    area_segmentation_square_fusion,
    Compute_GreedyPCA_area,
    compute_local_max,
    Compute_PCA_threshold,
    compute_segmap_gauss,
    Compute_threshold_purity,
    Correlation_GLR_test,
    create_local_max_cat,
    create_masks,
    dct_residual,
    estimation_line,
    merge_similar_lines,
    O2test,
    purity_estimation,
    unique_lines,
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
                if name in ('STD', 'T_GLR') and np.ma.is_masked(cat[name]):
                    cat[name].fill_value = np.nan
    return cat


def save_spectra(spectra, outname):
    """ The spectra are saved to a FITS file with two extension per
    spectrum, a DATA<ID> one and a STAT<ID> one.
    """
    hdulist = []
    for spec_id, sp in spectra.items():
        hdu = sp.get_data_hdu(name='DATA%d' % spec_id, savemask='nan')
        hdulist.append(hdu)
        hdu = sp.get_stat_hdu(name='STAT%d' % spec_id)
        if hdu is not None:
            hdulist.append(hdu)

    hdulist = fits.HDUList([fits.PrimaryHDU()] + hdulist)
    hdulist.writeto(outname, overwrite=True)


def load_spectra(filename):
    spectra = OrderedDict()
    with fits.open(filename) as hdul:
        for i in range(1, len(hdul), 2):
            spec_id = int(hdul[i].name[4:])
            spectra[spec_id] = Spectrum(hdulist=hdul, ext=(i, i+1))
    return spectra


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
    FAILED = 'failed'


class DataObj:
    """Descriptor used to load the data on demand.

    The "kind" of data is stored (cube, image, array, etc.), and used to load
    the data is the attribute is accessed. This requires that the path of the
    file associated to the data is set to its value.

    """
    def __init__(self, kind):
        # Note: self.label is set by the metaclass
        self.kind = kind

    def __get__(self, obj, owner=None):
        try:
            val = obj.__dict__[self.label]
        except KeyError:
            return

        if isinstance(val, str):
            if os.path.isfile(val):
                kind = self.kind
                if kind == 'cube':
                    val = Cube(val)
                if kind == 'image':
                    val = Image(val)
                elif kind == 'table':
                    val = _format_cat(Table.read(val))
                elif kind == 'array':
                    val = np.loadtxt(val, ndmin=1)
                elif kind == 'spectra':
                    val = load_spectra(val)
                obj.__dict__[self.label] = val
            else:
                val = None

        return val

    def __set__(self, obj, val):
        obj.__dict__[self.label] = val


class StepMeta(type):
    """Metaclass used to manage DataObj descriptors.

    The metaclass does two things:
    - It sets the "label" attribute on the descriptors, which allow to use
      ``cube_std = DataObj('cube')`` instead of ``cube_std =
      DataObj('cube_std', 'cube')`` if the label must be set manually.
    - It stores a list of all the descriptors in a ``_dataobjs`` attribute on
      the instance.

    """
    def __new__(cls, name, bases, attrs):
        descr = []
        for n, inst in attrs.items():
            if isinstance(inst, DataObj):
                inst.label = n
                descr.append((n, inst.kind))
        attrs['_dataobjs'] = descr
        return super(StepMeta, cls).__new__(cls, name, bases, attrs)


class Step(LogMixin, metaclass=StepMeta):
    """Define a processing step."""

    """Name of the function to run the step."""
    name = None

    """Description of the step."""
    desc = None

    """List of required steps (that must be run before)."""
    require = None

    def __init__(self, orig, idx, param):
        self.logger = logging.getLogger(__name__)
        self.orig = orig
        self.idx = idx
        self.method_name = 'step%02d_%s' % (idx, self.name)
        # when a session is reloaded, use its param dict (don't overwrite it)
        self.meta = param.setdefault(self.name, {})
        self.meta.setdefault('stepidx', idx)
        self.param = self.meta.setdefault('params', {})

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
        """Run a step, calling its ``run`` method.

        This method is the one that is called from the ORIGIN object. It calls
        the ``run`` method of the step and also does a few additional things
        like storing the parameters, execution time and date.

        """
        t0 = time.time()
        self._loginfo('Step %02d - %s', self.idx, self.desc)

        # Use the step's signature to 1) store the parameter values in the
        # `self.param` dict and 2) to print the default and actual values.
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

        # Manage dependencies between steps
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
        """Create a MPDAF Cube and store it as an attribute."""
        cube = Cube(data=data, wave=self.orig.wave, wcs=self.orig.wcs,
                    mask=np.ma.nomask, copy=False, **kwargs)
        setattr(self, name, cube)

    def store_image(self, name, data, **kwargs):
        """Create a MPDAF Image and store it as an attribute."""
        im = Image(data=data, wcs=self.orig.wcs, copy=False, **kwargs)
        setattr(self, name, im)

    def dump(self, outpath):
        """Save the attributes that have been created by the steps, and unload
        them to free memory.
        """
        if self.status is not Status.RUN:
            # If the step was not run, there is nothing to dump
            return

        self.logger.debug('%s - DUMP', self.method_name)
        for name, kind in self._dataobjs:
            obj = getattr(self, name)
            if obj is not None:
                outf = '{}/{}.{}'.format(
                    outpath, name, 'txt' if kind == 'array' else 'fits')
                self.logger.debug('   - %s [%s]', name, kind)
                if kind in ('cube', 'image'):
                    try:
                        obj.write(outf, convert_float32=False)
                    except TypeError:
                        warnings.warn('MPDAF version too old to support '
                                      'the new type conversion parameter, '
                                      'data will be saved as float32.')
                        obj.write(outf)
                elif kind in ('table', ):
                    obj.write(outf, overwrite=True)
                elif kind in ('array', ):
                    np.savetxt(outf, obj)
                elif kind in ('spectra', ):
                    save_spectra(obj, outf)

                # now set the value to the path of the written file, which
                # frees memory, and allow the object to reload automatically
                # the data if needed.
                setattr(self, name, outf)

        self.status = Status.DUMPED

    def load(self, outpath):
        """Recreate attributes of a step, not really loading them as just the
        file is set, and files are loaded in memory only if needed.
        """
        if self.status is not Status.DUMPED:
            # If the step was not dumped previously, there is nothing to load
            return

        self.logger.debug('%s - LOAD', self.method_name)
        for name, kind in self._dataobjs:
            # we just set the paths here, the data will be loaded only if
            # the attribute is accessed.
            outf = '{}/{}.{}'.format(outpath, name,
                                     'txt' if kind == 'array' else 'fits')
            setattr(self, name, outf)


class Preprocessing(Step):
    """Continuum subtraction, standardization and segmentation.

    Parameters
    ----------
    dct_order : int
        The number of atom to keep for the dct decomposition.
    dct_approx : bool
        if True, the DCT computation does not take the variance into account
        for the computation of the DCT coefficients.
    pfasegcont : float
        PFA for the segmentation based on the continuum.
    pfasegres : float
        PFA for the segmentation based on the residual.

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
    self.segmap : `~mpdaf.obj.Image`
        Segmentation map.

    """

    name = 'preprocessing'
    desc = 'Preprocessing'
    cube_std = DataObj('cube')
    cont_dct = DataObj('cube')
    ima_std = DataObj('image')
    ima_dct = DataObj('image')
    segmap = DataObj('image')

    def run(self, orig, dct_order=10, dct_approx=False, pfasegcont=0.01,
            pfasegres=0.01):
        self._loginfo('DCT computation')
        cont_dct = dct_residual(orig.cube_raw, dct_order, orig.var, dct_approx,
                                orig.mask)
        data = orig.cube_raw - cont_dct  # compute faint_dct
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
        cont_dct = cont_dct.astype(np.float32)
        self.store_cube('cont_dct', cont_dct)
        self.store_image('ima_dct', cont_dct.mean(axis=0))

        mean_fwhm = int(np.ceil(np.mean(self.orig.FWHM_PSF)))

        self._loginfo('Segmentation based on the continuum')
        # Test statistic for each spaxels - use log here as it gives a more
        # gaussian distribution
        map1 = np.log10(np.sum(cont_dct**2, axis=0))
        thresh, map_cont = compute_segmap_gauss(map1, pfasegcont, mean_fwhm)
        self._loginfo('Found %d regions, threshold=%.2f',
                      len(np.unique(map_cont)) - 1, thresh)

        self._loginfo('Segmentation based on the residual')
        map2 = O2test(data)
        thresh, map_res = compute_segmap_gauss(map2, pfasegres, mean_fwhm)
        self._loginfo('Found %d regions, threshold=%.2f',
                      len(np.unique(map_res)) - 1, thresh)

        self._loginfo('Merging both maps')
        segmap, nlabels = ndi.label((map_cont > 0) | (map_res > 0))
        self._loginfo('Segmap saved in self.segmap (%d regions)', nlabels)
        self.store_image('segmap', segmap)


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
    areamap = DataObj('image')

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
                orig.segmap._data, square_cut_fus, pfa, orig.Ny, orig.Nx)

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
    thresO2 = DataObj('array')
    meaO2 = DataObj('array')
    stdO2 = DataObj('array')
    # FIXME: test02, histO2 and binO2 are lists of arrays with variable
    # sizes so they cannot be managed by the Step class currently
    # testO2 = DataObj('array')
    # histO2 = DataObj('array')
    # binO2 = DataObj('array')
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

        (orig.testO2, orig.histO2, orig.binO2, self.thresO2, self.meaO2,
         self.stdO2) = zip(*results)


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
    cube_faint = DataObj('cube')
    mapO2 = DataObj('image')
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
    size : int
        Connectivity of contiguous voxels per axis, for the maximum filter.
    ncpu : int
        Number of CPUs used, defaults to 1.
    pcut : float
        Cut applied to the profiles to limit their width (default 1e-8).
    pmeansub : bool
        Subtract the mean of the profiles (default True).

    Returns
    -------
    self.cube_correl : `~mpdaf.obj.Cube`
        Cube of T_GLR values
    self.correl_min : `~mpdaf.obj.Cube`
        Cube of T_GLR values of minimum correlation
    self.cube_profile : `~mpdaf.obj.Cube` (type int)
        Number of the profile associated to the T_GLR
    self.maxmap : `~mpdaf.obj.Image`
        Map of maximum correlations along the wavelength axis
    self.minmap : `~mpdaf.obj.Image`
        Map of minimum correlations along the wavelength axis
    self.cube_local_max : `~mpdaf.obj.Cube`
        Local maxima from max correlation
    self.cube_local_min : `~mpdaf.obj.Cube`
        Local maxima from minus min correlation

    """

    name = 'compute_TGLR'
    desc = 'GLR test'
    cube_correl = DataObj('cube')
    cube_correl_min = DataObj('cube')
    cube_profile = DataObj('cube')
    cube_local_min = DataObj('cube')
    cube_local_max = DataObj('cube')
    maxmap = DataObj('image')
    minmap = DataObj('image')
    require = ('compute_greedy_PCA', )

    def run(self, orig, size=3, ncpu=1, pcut=1e-8, pmeansub=True):
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
        correl, profile, correl_min = Correlation_GLR_test(
            orig.cube_faint._data, orig.var, orig.PSF, orig.wfields,
            orig.profiles, nthreads=ncpu, pcut=pcut, pmeansub=pmeansub)

        self._loginfo('Save the TGLR value in self.cube_correl')
        correl[orig.mask] = 0
        self.store_cube('cube_correl', correl)
        self.store_cube('cube_correl_min', correl_min)

        self._loginfo('Save the number of profile associated to the TGLR'
                      ' in self.cube_profile')
        profile[orig.mask] = 0
        self.store_cube('cube_profile', profile)

        self._loginfo('Save the map of maxima in self.maxmap')
        self.store_image('maxmap', np.amax(correl, axis=0))
        self.store_image('minmap', np.amin(correl_min, axis=0))

        self._loginfo('Compute p-values of local maximum of correlation '
                      'values')
        cube_local_max, cube_local_min = compute_local_max(correl, correl_min,
                                                           orig.mask, size)
        self._loginfo('Save self.cube_local_max from max correlations')
        self.store_cube('cube_local_max', cube_local_max)
        self._loginfo('Save self.cube_local_min from min correlations')
        self.store_cube('cube_local_min', cube_local_min)


class ComputePurityThreshold(Step):
    """Find the threshold for a given purity.

    Parameters
    ----------
    purity : float
        Purity to automatically compute the threshold.
    threshlist : list
        List of thresholds to compute the purity.
    pfasegfinal : float
        PFA for the segmentation based on the maxmap.

    Returns
    -------
    self.threshold_correl : float
        Estimated threshold
    self.Pval : astropy.table.Table
        Table with the purity results for each threshold:
        - Pval_r : The purity function
        - Tval_r : index value to plot
        - Det_m : Number of detections (-DATA)
        - Det_M : Number of detections (+DATA)
    self.segmap_purity : `~mpdaf.obj.Image`
        Segmentation map.

    """

    name = 'compute_purity_threshold'
    desc = 'Compute Purity threshold'
    Pval = DataObj('table')
    segmap_purity = DataObj('image')
    require = ('compute_TGLR', )

    def run(self, orig, purity=.9, threshlist=None, pfasegfinal=1e-5):
        orig.param['purity'] = purity
        self._loginfo('Estimation of threshold with purity = %.2f', purity)

        # purity sur zone background avec adaptation de mapmerged
        # je prends la carte des correls max, je fais une approx gaussienne
        # et je seuille
        thresh, map_res = compute_segmap_gauss(self.orig.maxmap._data,
                                               pfasegfinal, 0)
        segmap, nlabels = ndi.label((map_res > 0) | (orig.segmap._data > 0))
        self.store_image('segmap_purity', segmap)

        threshold, self.Pval = Compute_threshold_purity(
            purity, orig.cube_local_max._data, orig.cube_local_min._data,
            segmap, threshlist=threshlist)
        orig.param['threshold'] = threshold
        self._loginfo('Threshold: %.2f ', threshold)


class Detection(Step):
    """Detections on local maxima from max correlation + spatia-spectral
    merging in order to create the first catalog.

    Parameters
    ----------
    threshold : float
        User threshold if the estimated threshold is not good.
    tol_spat : int
        Tolerance for the spatial merging (distance in pixels).
        TODO en fonction du FWHM
    tol_spec : int
        Tolerance for the spectral merging (distance in pixels).

    Returns
    -------
    self.Cat0 : astropy.Table
        Catalog with detections. Columns:
        ID ra dec lbda x0 y0 z0 profile seg_label T_GLR

    """

    name = 'detection'
    desc = 'Thresholding and spatio-spectral merging'
    Cat0 = DataObj('table')

    def det_correl_min(self, thresh=None):
        """3D positions of detections in correl_min."""
        thresh = thresh or self.orig.param['threshold']
        zm, ym, xm = np.where(self.orig.cube_local_min._data > thresh)
        return zm, ym, xm

    def run(self, orig, threshold=None, tol_spat=3, tol_spec=5):

        if threshold is not None:
            orig.param['threshold'] = threshold

        self.Cat0 = create_local_max_cat(
            orig.param['threshold'], orig.cube_local_max._data,
            orig.segmap._data, tol_spat, tol_spec, orig.cube_profile._data,
            orig.wcs, orig.wave)
        _format_cat(self.Cat0)
        self._loginfo('Save the catalog in self.Cat0 (%d sources %d lines)',
                      len(np.unique(self.Cat0['ID'])), len(self.Cat0))


class DetectionLost(Step):
    """Detections on local maxima of std cube + spatia-spectral
    merging in order to create an complementary catalog. This catalog is
    merged with the catalog Cat0 in order to create the catalog Cat1

    Parameters
    ----------
    purity : float
        purity to automatically compute the threshold
        If None, previous purity is used
    threshlist : list
        list of thresholds to compute the purity default None

    Returns
    -------
    self.threshold_correl : float
        Estimated threshold used to detect complementary
        lines on local maxima of std cube
    self.Pval_comp : astropy.table.Table
        Table with the purity results for each threshold:
        - Pval_r : The purity function
        - Tval_r : index value to plot
        - Det_m : Number of detections (-DATA)
        - Det_M : Number of detections (+DATA)
    self.Cat1 : astropy.Table
        New catalog.
        Columns: ID ra dec lbda x0 y0 z0 profile seg_label T_GLR STD comp

    """

    name = 'detection_lost'
    desc = 'Thresholding and spatio-spectral merging'
    Cat1 = DataObj('table')
    Pval_comp = DataObj('table')
    require = ('detection', )

    def run(self, orig, purity=None, threshlist=None):
        self._loginfo('Compute local maximum of std cube values')
        size = orig.param['compute_TGLR']['params']['size']
        local_max, local_min = compute_local_max(
            orig.cube_std._data, orig.cube_std._data, orig.mask, size)
        orig.cube_std_local_max = local_max
        orig.cube_std_local_min = local_min

        if purity is None:
            purity = orig.param['compute_purity_threshold']['params']['purity']
        orig.param['purity2'] = purity
        self._loginfo('Threshold computed with purity = %.2f', purity)

        # FIXME: use segmap or segmap_purity ?
        threshold2, self.Pval_comp = Compute_threshold_purity(
            purity, local_max, local_min, orig.segmap_purity._data,
            threshlist=threshlist)
        orig.param['threshold2'] = threshold2
        self._loginfo('Threshold: %.2f ', threshold2)

        Cat1 = orig.Cat0.copy()
        Cat1['comp'] = 0

        if threshold2 == np.inf:
            Cat1['STD'] = 0
        else:
            p = orig.param['detection']['params']
            Catcomp = create_local_max_cat(
                threshold2, local_max, orig.segmap._data, p['tol_spat'],
                p['tol_spec'], orig.cube_profile._data, orig.wcs, orig.wave)
            Catcomp.rename_column('T_GLR', 'STD')

            # merging
            Catcomp['comp'] = 1
            Catcomp['ID'] += (Cat1['ID'].max() + 1)
            Cat1 = _format_cat(vstack([Cat1, Catcomp]))
            # vstack creates a masked Table, masking the missing values. But
            # a bug with Astropy/Numpy 1.14 is causing the Cat1 table to be
            # modified later by Cat2 operations, if it was not dumped before.
            # So for now we transform the table to a non-masked one.
            Cat1 = Cat1.filled()

        ns = len(np.unique(Cat1['ID']))
        ds = ns - len(np.unique(orig.Cat0['ID']))
        nl = len(Cat1)
        dl = nl - len(orig.Cat0)
        self.Cat1 = Cat1
        self._loginfo('Save the catalog in self.Cat1'
                      ' (%d [+%s] sources %d [+%d] lines)', ns, ds, nl, dl)


class ComputeSpectra(Step):
    """Compute the estimated emission line and the optimal coordinates

    For each detected line in a spatio-spectral grid, the line
    is estimated with the deconvolution model::

        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))

    Via PCA LS or denoised PCA LS Method.

    Parameters
    ----------
    grid_dxy : int
        Maximum spatial shift for the grid
    spectrum_size_fwhm: float
        The length of the spectrum to keep around each line as a factor of
        the fitted line FWHM.

    Returns
    -------
    self.Cat2 : astropy.Table
        Catalog of parameters of detected emission lines.  Columns:
        ra dec lbda x0 x y0 y z0 z T_GLR profile residual flux num_line purity
    self.spectra : list of `~mpdaf.obj.Spectrum`
        Estimated lines

    """

    name = 'compute_spectra'
    desc = 'Lines estimation'
    Cat2 = DataObj('table')
    spectra = DataObj('spectra')
    require = ('detection_lost', )

    def run(self, orig, grid_dxy=0, spectrum_size_fwhm=6):
        self.Cat2, line_est, line_var = estimation_line(
            orig.Cat1, orig.cube_raw, orig.var, orig.PSF,
            orig.wfields, orig.wcs, orig.wave, size_grid=grid_dxy,
            criteria='flux', order_dct=30, horiz_psf=1, horiz=5
        )

        self._loginfo('Purity estimation')
        tmp_Cat2 = purity_estimation(self.Cat2, orig.Pval, orig.Pval_comp)

        # Remove duplicated lines
        unique_idx = unique_lines(tmp_Cat2)
        self.Cat2 = tmp_Cat2[unique_idx]
        line_est = np.array(line_est)[unique_idx]
        line_var = np.array(line_var)[unique_idx]

        _format_cat(self.Cat2)

        self._loginfo('Save the updated catalog in self.Cat2 (%d lines)',
                      len(orig.Cat2))
        nb_duplicated = len(tmp_Cat2) - len(orig.Cat2)
        if nb_duplicated:
            self._loginfo('%d lines were removed for being duplicates.',
                          nb_duplicated)

        # Radius for spectrum trimming
        radius = np.ceil(np.array(orig.FWHM_profiles) * spectrum_size_fwhm
                         / 2).astype(int)

        self.spectra = OrderedDict()
        for line_idx, (data, vari) in enumerate(zip(line_est, line_var)):
            profile, z, num_line = self.Cat2[line_idx]['profile', 'z',
                                                       'num_line']
            line_z_min = z - radius[profile]
            line_z_max = z + radius[profile]
            self.spectra[num_line] = Spectrum(
                data=data, var=vari, wave=orig.wave, mask=np.ma.nomask
            ).subspec(line_z_min, line_z_max, unit=None)

        self._loginfo('Save estimated spectrum of each line in self.spectra')


class CleanResults(Step):
    """Clean the various results.

    This step does several things to “clean” the results of ORIGIN:

    - Some lines are associated to the same source but are very near
      considering their z positions.  The lines are all marked as merged in
      the brightest line of the group (but are kept in the line table).
    - A table of unique sources is created.

    Attributes added to the ORIGIN object:
    - `Cat3_lines`: clean table of lines;
    - `Cat3_sources`: table of unique sources

    Parameters
    ----------
    merge_lines_z_threshold: int
        z axis pixel threshold used when merging similar lines.

    """

    name = 'clean_results'
    desc = 'Results cleaning'
    Cat3_lines = DataObj('table')
    Cat3_sources = DataObj('table')
    require = ('compute_spectra', )

    def run(self, orig, merge_lines_z_threshold=5):
        self.Cat3_lines = merge_similar_lines(
            orig.Cat2, z_pix_threshold=merge_lines_z_threshold)
        self.Cat3_sources = unique_sources(orig.Cat3_lines)

        self._loginfo('Save the unique source catalog in self.Cat3_sources'
                      ' (%d sources)', len(orig.Cat3_sources))
        self._loginfo('Save the cleaned lines in self.Cat3_lines (%d lines)',
                      len(orig.Cat3_lines))
        nb_line_merged = np.sum(orig.Cat3_lines['merged_in'] != -9999)
        if nb_line_merged:
            self._loginfo('%d lines were merged in nearby lines',
                          nb_line_merged)


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
    plot_problems: bool
        If true, when the mask or a source seems dubious, some diagnostic plots
        about the source and its lines is saved in the output folder. Note that
        this may be problematic when running the code inside a notebook, hence
        the option is False by default.

    """

    name = 'create_masks'
    desc = 'Mask creation'
    require = ('clean_results', )

    def run(self, orig, path=None, overwrite=True, mask_size=50,
            seg_thres_factor=.5, plot_problems=False):
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
            plot_problems=plot_problems)


class SaveSources(Step):
    """Create the source file for each source.

    Parameters
    ----------
    version: str
        Version number of the source files.
    path: str
        Path where the sources will be saved.
    n_jobs : int
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
    overwrite: bool
        Overwrite the folder if it already exists.

    """

    name = 'save_sources'
    desc = 'Save sources'

    def run(self, orig, version, *, path=None, n_jobs=1, author="",
            nb_fwhm=2, size=5, expmap_filename=None, overwrite=True):

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

        # We need the correlation cube and the spectrum FITS files saved to
        # disk.
        orig.write()

        from .source_creation import create_all_sources
        create_all_sources(
            cat3_sources=orig.Cat3_sources,
            cat3_lines=orig.Cat3_lines,
            origin_params=orig.param,
            cube_cor_filename=os.path.join(outpath, 'cube_correl.fits'),
            mask_filename_tpl=orig.param['mask_filename_tpl'],
            skymask_filename_tpl=orig.param['skymask_filename_tpl'],
            spectra_fits_filename=os.path.join(outpath, 'spectra.fits'),
            segmap_filename=os.path.join(outpath, 'segmap.fits'),
            version=version,
            profile_fwhm=orig.FWHM_profiles,
            out_tpl=os.path.join(out_dir, 'source-%0.5d.fits'),
            n_jobs=n_jobs,
            author=author,
            nb_fwhm=nb_fwhm,
            size=size,
            expmap_filename=expmap_filename,
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
