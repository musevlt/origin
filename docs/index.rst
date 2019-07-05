ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes
=========================================================

.. ifconfig:: 'dev' in release

    .. warning::

        This documentation is for the version of ORIGIN currently under
        development.

.. include:: ../README.rst

This software was initially developed by Carole Clastres, under the supervision
of David Mary (Lagrange institute, University of Nice). It was then ported to
Python by Laure Piqueras (CRAL). From November 2016 to November 2107 the
software was developed by Antony Schutz (CRAL/Lagrange) and Laure. Then it was
developed by Simon Conseil (CRAL), in parallel with an Octave version by David,
and with contributions from Yannick Roehlly (CRAL). A lot of testing has been
done also by Roland Bacon (CRAL), which also produced simulated cubes.

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL).

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

.. automodapi:: origin
   :no-heading:
