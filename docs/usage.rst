Usage
=====

The ORIGIN algorithm is computationally intensive, hence it is divided in
steps.  It is possible to save the outputs after each step and to reload
a session to continue the processing.

As ORIGIN uses intensively linear algebra (PCA) and FFT routines, using the
Intel MKL with Numpy may bring significant performance boost. This is the case
by default when using Anaconda or Miniconda, and it is also possible to use the
Numpy package from Intel `with pip`_. They also provide *mkl-fft* with
a parallelized version of the FFT.

The ORIGIN class
----------------

To run ORIGIN everything can be done through the `~muse_origin.ORIGIN` object.
To instantiate this object, we need to pass it a MUSE datacube. Here for the
example we will use the MUSE cube stored in the ``muse_origin`` package, which
is also used for the unit tests::

    >>> import muse_origin
    >>> import os
    >>> origdir = os.path.dirname(muse_origin.__file__)
    >>> CUBE = os.path.join(origdir, '..', 'tests', 'minicube.fits')

This supposes that you use a full copy of the git repository, the tests are
not included in the Python package because the file is quite big.

We also must give it a ``name``, this is the "session" name which is used as
the directory name in which outputs will be saved (inside the current directory
by default, which can be overridden with the ``path`` argument)::

    >>> testpath = getfixture('tmpdir')
    >>> orig = muse_origin.ORIGIN(CUBE, name='origtest', loglevel='INFO', path=testpath)
    INFO : Step 00 - Initialization (ORIGIN ...)
    INFO : Read the Data Cube ...
    INFO : Compute FSFs from the datacube FITS header keywords
    INFO : mean FWHM of the FSFs = 3.32 pixels
    INFO : 00 Done

Then this object allows to run the steps, which is described in more details in
the example notebook. It is possible to know the status of the processing with
`~muse_origin.status`::

    >>> orig.status()
    - 01, preprocessing: NOTRUN
    - 02, areas: NOTRUN
    - 03, compute_PCA_threshold: NOTRUN
    - 04, compute_greedy_PCA: NOTRUN
    - 05, compute_TGLR: NOTRUN
    - 06, compute_purity_threshold: NOTRUN
    - 07, detection: NOTRUN
    - 08, compute_spectra: NOTRUN
    - 09, clean_results: NOTRUN
    - 10, create_masks: NOTRUN
    - 11, save_sources: NOTRUN

FSF
---

ORIGIN needs some information about the FSF (*Field Spread Function*),
including its dependency on the wavelength.  The FSF model can be read from the
cube with MPDAF's `FSF models`_ (`mpdaf.MUSE.FSFModel`) or it can be provided
as parameter. It is also possible to use a Fieldmap_ for the case of mosaics
where the FSF varies on the field.

By default ORIGIN supposes that the cube contains an FSF model that can be
read with MPDAF::

    >>> from mpdaf.MUSE import FSFModel
    >>> fsfmodel = FSFModel.read(CUBE)
    >>> fsfmodel
    <OldMoffatModel(model=MOFFAT1)>
    >>> fsfmodel.to_header().cards
    ('FSFMODE', 'MOFFAT1', 'Old model with a fixed beta')
    ('FSF00BET', 2.8, '')
    ('FSF00FWA', 0.869, '')
    ('FSF00FWB', -3.401e-05, '')

This model is used to get the FWHM for each wavelength plane, otherwise the
full list must be provides as parameter::

    >>> import numpy as np
    >>> fsfmodel.get_fwhm(np.array([5000, 7000, 9000]))
    array([0.69... , 0.63..., 0.56...])

Profiles
--------

For the spectral axis ORIGIN uses a dictionary of line profiles, which will be
used to compute the spectral correlation. The ORIGIN comes with two FITS files
containing profiles:

- ``Dico_3FWHM.fits``: contains 3 profiles with FWHM of 2, 6.7 and 12 pixels.
  This is the one used by default.
- ``Dico_FWHM_2_12.fits``: contains 20 profiles with FWHM between 2 and 12
  pixels.

Session save and restore
------------------------

At any point it is possible to save the current state with
`~muse_origin.ORIGIN.write`::

    >>> orig.write()
    INFO : Writing...
    INFO : Current session saved in .../origtest

This uses the ``name`` of the object as output directory::

    >>> orig.name
    'origtest'

In this output directory, all the step outputs are saved, as well as a log file
(``{name}.log``) and a YAML file with all the parameters used in the various
steps (``{name}.yaml``).

A session can then be reloaded with `~muse_origin.ORIGIN.load`::

    >>> import muse_origin
    >>> orig = muse_origin.ORIGIN.load(os.path.join(testpath, 'origtest'))
    INFO : Step 00 - Initialization (ORIGIN ...)
    INFO : Read the Data Cube ...
    INFO : Compute FSFs from the datacube FITS header keywords
    INFO : mean FWHM of the FSFs = 3.32 pixels
    INFO : 00 Done

Another interesting point with the session feature is that saving the current
state will unload the data from the memory. When running the steps, various
data objects (cubes, images, tables) are added as attributes to the step
classes, and saving the session will dump these objects to disk and free the
memory.

Steps
-----

The steps are implemented are `~muse_origin.Step` sub-classes (described below),
which can be run with methods of the `~muse_origin.ORIGIN` object:

- ``orig.step01_preprocessing``
- ``orig.step02_areas``
- ``orig.step03_compute_PCA_threshold``
- ``orig.step04_compute_greedy_PCA``
- ``orig.step05_compute_TGLR``
- ``orig.step06_compute_purity_threshold``
- ``orig.step07_detection``
- ``orig.step08_compute_spectra``
- ``orig.step09_clean_results``
- ``orig.step10_create_masks``
- ``orig.step11_save_sources``

Each step can has several parameters, with default values that should be fine in
the general case. The most important parameters are mentioned below,

Step 1: `~muse_origin.Preprocessing`
    Preparation of the data for the following steps:

    - Nuisance removal with DCT. The estimated continuum cube is stored in
      ``cube_dct``. The order of the DCT is set with the ``dct_order`` keyword.

    - Standardization of the data (stored in ``cube_std``).

    - Computation of the local maxima and minima of ``cube_std``.

    - Segmentation based on the continuum (``segmap_cont``), with the threshold
      defined by ``pfasegcont``.

    - Segmentation based on the residual image (``ima_std``), with the
      threshold defined by ``pfasegres``, merged with the previous one which
      gives ``segmap_merged``.

Step 2: `~muse_origin.CreateAreas`
    Creation of areas for the PCA.

    The purpose of spatial segmentation is to locate regions where the sky
    contains "nuisance" sources, i.e., sources with continuum and / or bright
    emission lines, or regions exhibiting a particular statistical behaviour,
    caused by the presence of systematic residuals for instance.

    The merged segmap computed previously is used to avoid cutting objects. The
    size of the areas is controlled with the ``minsize`` and ``maxsize``
    keywords.

Step 3: `~muse_origin.ComputePCAThreshold`
    Loop on each area and estimate the threshold for the PCA, using the
    ``pfa_test`` parameter.

Step 4: `~muse_origin.ComputeGreedyPCA`
    Nuisance removal with iterative PCA.

    This is one of the most computationally intensive step in ORIGIN, with the
    following step.

    Loop on each area and compute the iterative PCA: iteratively locate and
    remove residual nuisance sources, i.e., any signal that is not the signa-
    ture of a faint, spatially unresolved emission line.  Use by default the
    thresholds computed in step 3.

Step 5: `~muse_origin.ComputeTGLR`
    Compute the cube of GLR test values (the "correlation" cube).

    The test is done on the cube containing the faint signal (``cube_faint``)
    and it uses the PSF and the spectral profiles. Then computes the local
    maximum and minima of correlation values and stores the maxmap and minmap
    images. It is possible to use multiprocessing to parallelize the work (with
    ``n_jobs``), but the best is to use the c-level parallelization with the
    *mkl-fft* package.

Step 6: `~muse_origin.ComputePurityThreshold`
    Find the thresholds for the given purity, for the correlation (faint)
    cube and the complementary (std) one.

Step 7: `~muse_origin.Detection`
    Detections on local maxima from the correlation and complementary cube,
    using the thresholds computed in step 5. It is also possible to provides
    thresholds with the corresponding parameters. This creates the ``Cat0``
    table.

    Then the detections are merged in sources, to create ``Cat1``. See
    :ref:`merging` below.

Step 8: `~muse_origin.ComputeSpectra`
    Compute the estimated emission line and the optimal coordinates.

    This computes ``Cat2`` with a refined position for sources.  And for each
    detected line in a spatio-spectral grid, the line is estimated with the
    deconvolution model::

        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))

    Via PCA LS or denoised PCA LS Method.

Step 9: `~muse_origin.CleanResults`
    This step does several things to “clean” the results of ORIGIN:

    - Some lines are associated to the same source but are very near
      considering their z positions.  The lines are all marked as merged in
      the brightest line of the group (but are kept in the line table).
    - A table of unique sources is created.
    - Statistical detection info is added on the 2 resulting catalogs.

    This step produces two tables:

    - `Cat3_lines`: clean table of lines;
    - `Cat3_sources`: table of unique sources.

Step 10: `~muse_origin.CreateMasks`
    This step create a source mask and a sky mask for each source. These masks
    are computed as the combination of masks on the narrow band images of each
    line.

Step 11: `~muse_origin.SaveSources`
    Create an `mpdaf.sdetect.Source` file for each source.

.. _merging:

Merging of lines in sources
---------------------------

Once we get the list of line detections, we need to group these detections in
"sources", where a given source can have multiple lines. It's a tricky step
because extended sources can have detections with different spatial positions.
To solve this problem we use the information from a segmentation map, that can
be provided or computed automatically on the continuum image, to identify the
regions of bright or extended sources. And we adopt a different method for
detections that are in these areas.

First, the detections are merged based on a spatial distance criteria (the
``tol_spat`` parameter). Starting from a given detection, the detections within
a distance of ``tol_spat`` are merged. Then looking iteratively at the
neighbors of the merged detections, these are merged in the group if their
distance to the seed detection is less than ``tol_spat``, or if the distance on
the wavelength axis is less than ``tol_spec``. And this process is repeated for
all detections that are not yet merged.

Then we take all the detections that belong to a given region of the
segmentation map, and if there is more than one group of lines from the
previous step we compute the distance on the wavelength axis between the groups
of lines. If the minimum distance in wavelength is less than ``tol_spec`` then
the groups are merged.







.. _FSF models: https://mpdaf.readthedocs.io/en/stable/muse.html#muse-fsf-models
.. _Fieldmap: https://mpdaf.readthedocs.io/en/stable/muse.html#muse-mosaic-field-map
.. _with pip: https://software.intel.com/en-us/articles/installing-the-intel-distribution-for-python-and-intel-performance-libraries-with-pip-and
