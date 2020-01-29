Installation
============

ORIGIN requires Python 3.6+ and the following packages:

* Numpy
* Astropy
* SciPy
* MPDAF
* Joblib
* tqdm
* PyYAML
* Matplotlib
* Photutils

The last stable release of ORIGIN can be installed simply with pip::

    pip install origin

Or into the user path with::

    pip install --user origin

.. tip::

    As ORIGIN uses intensively linear algebra (PCA) and FFT routines, using the
    Intel MKL with Numpy may bring significant performance boost. This is the case
    by default when using Anaconda or Miniconda, and it is also possible to use the
    Numpy package from Intel with `with pip`_. They also provide *mkl-fft* with
    a parallelized version of the FFT.

.. _with pip: https://software.intel.com/en-us/articles/installing-the-intel-distribution-for-python-and-intel-performance-libraries-with-pip-and
