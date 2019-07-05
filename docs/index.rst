ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes
=========================================================

.. ifconfig:: 'dev' in release

    .. warning::

        This documentation is for the version of ORIGIN currently under
        development.

.. include:: ../README.rst

.. contents::

Installation
============

ORIGIN requires the following packages:

* Numpy
* Astropy
* SciPy
* MPDAF
* Joblib
* tqdm
* PyYAML
* Matplotlib

The last stable release of ORIGIN can be installed simply with pip::

    pip install origin

Or into the user path with::

    pip install --user origin

How it works
============

Brief description, TODO.

Usage
=====


1- we import the package and create an ORIGIN object::

    from origin import ORIGIN
    orig = ORIGIN.init('cube.fits', name='tmp')

2- we run the different steps::

    orig.step01_preprocessing()
    orig.step02_areas()
    orig.step03_compute_PCA_threshold()
    orig.step04_compute_greedy_PCA()
    orig.step05_compute_TGLR()
    orig.step06_compute_purity_threshold()
    orig.step07_detection()
    orig.step08_detection_lost()
    orig.step09_compute.spectra()
    nsources = orig.step12_save_sources()

3- Resulted detected sources can be load by using mpdaf::

    from mpdaf.sdetect import Source
    src = Source.from_file('./tmp/sources/tmp-00001.fits')

4- at each step, we can write the results in a folder::

    orig.write()

5- it is possible to create an ORIGIN object from a previous session by loading
the data stored in the folder::

    orig = ORIGIN.load('tmp')

Changelog
=========

.. include:: ../CHANGELOG

API
===
