How it works
============

The ORIGIN algorithm is computationally intensive, hence it is divided in
steps.  It is possible to save the outputs after each step and to reload
a session to continue the processing.

From the user side, everything can be done through the `~origin.ORIGIN` object.
To Instantiate this object, we need to pass it a MUSE datacube. Here for the
example we will use the MUSE cube stored in the ``origin`` package, which is
used for the unit tests::

    >>> import origin
    >>> import os
    >>> origdir = os.path.dirname(origin.__file__)
    >>> CUBE = os.path.join(origdir, '..', 'tests', 'minicube.fits')

We also must give it a ``name``, this is the "session" name which is used as
the directory name in which outputs will be saved (in the current directory,
which can be overridden with the ``path`` argument)::

    >>> orig = origin.ORIGIN(CUBE, name='origtest', loglevel='INFO')
    INFO : Step 00 - Initialization (ORIGIN ...)
    INFO : Read the Data Cube ...
    INFO : Compute FSFs from the datacube FITS header keywords
    INFO : mean FWHM of the FSFs = 3.32 pixels
    INFO : 00 Done

During the instantiation it will also read the FSF information.

**TODO**: explain how to give this info

The steps are implemented as subclasses of the `~origin.Step` class, and can be
run directly through the `~origin.ORIGIN` object.

Step 1: `~origin.Preprocessing`
    Preparation of the data for the following steps:

    - Continuum subtraction with a DCT filter (the continuum cube is stored in
      ``cube_dct``)
    - Standardization of the data (stored in ``cube_std``).
    - Computation of the local maxima and minima of the std cube.
    - Segmentation based on the continuum (``segmap_cont``).
    - Segmentation based on the residual image (``ima_std``), merged with the
      previous one which gives ``segmap_merged``.

Step 2: `~origin.CreateAreas`
    Creation of areas to split the work.

    This allows to split the cube into sub-cubes to distribute the following
    steps on multiple processes. The merged segmap computed previously is used
    to avoid cutting objects.

Step 3: `~origin.ComputePCAThreshold`
    Loop on each sub-cube and estimate the threshold for the PCA.

Step 4: `~origin.ComputeGreedyPCA`
    Loop on each sub-cube and compute the greedy PCA.

Step 5: `~origin.ComputeTGLR`
    Compute the cube of GLR test values.

    The test is done on the cube containing the faint signal
    (``self.cube_faint``) and it uses the PSF and the spectral profiles.
    Then compute the p-values of local maximum of correlation values.

Step 6: `~origin.ComputePurityThreshold`
    Find the threshold for a given purity.

Step 7: `~origin.Detection`
    Detections on local maxima from correlation and std cube, and
    spatia-spectral merging in order to create the first catalog.

Step 8: `~origin.ComputeSpectra`
    Compute the estimated emission line and the optimal coordinates.

    For each detected line in a spatio-spectral grid, the line
    is estimated with the deconvolution model::

        subcube = FSF*line -> line_est = subcube*fsf/(fsf^2))

    Via PCA LS or denoised PCA LS Method.

Step 9: `~origin.CleanResults`
    This step does several things to “clean” the results of ORIGIN:

    - Some lines are associated to the same source but are very near
      considering their z positions.  The lines are all marked as merged in
      the brightest line of the group (but are kept in the line table).
    - A table of unique sources is created.
    - Statistical detection info is added on the 2 resulting catalogs.

Step 10: `~origin.CreateMasks`
    This step create the mask and sky mask for each source.

Step 11: `~origin.SaveSources`
    Create the source file for each source.
