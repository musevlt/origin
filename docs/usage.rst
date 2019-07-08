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

During the instantiation it will also read the FSF information. TODO: explain
how to give this info

The steps are implemented as subclasses of the `~origin.Step` class, and can be
run directly through the `~origin.ORIGIN` object.

- Step 1: `~origin.Preprocessing`
- Step 2: `~origin.CreateAreas`
- Step 3: `~origin.ComputePCAThreshold`
- Step 4: `~origin.ComputeGreedyPCA`
- Step 5: `~origin.ComputeTGLR`
- Step 6: `~origin.ComputePurityThreshold`
- Step 7: `~origin.Detection`
- Step 8: `~origin.ComputeSpectra`
- Step 9: `~origin.CleanResults`
- Step 10: `~origin.CreateMasks`
- Step 11: `~origin.SaveSources`
