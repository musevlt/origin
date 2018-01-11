"""Test interface on ORIGIN software."""

from __future__ import absolute_import, division

import numpy as np
import os
import shutil

from mpdaf.sdetect import Source, Catalog
from origin import ORIGIN

MINICUBE = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'minicube.fits')


def test_origin():
    """test ORIGIN"""
    # Number of subcubes for the spatial segmentation

    my_origin = ORIGIN.init(MINICUBE, name='tmp')
    my_origin.write()

    my_origin = ORIGIN.load('tmp')
    my_origin.step01_preprocessing()
    my_origin.write()

    my_origin = ORIGIN.load('tmp')
    my_origin.step02_areas()
    my_origin.write()

    my_origin = ORIGIN.load('tmp')
    my_origin.step03_compute_PCA_threshold()
    my_origin.write()

    my_origin = ORIGIN.load('tmp')
    my_origin.step04_compute_greedy_PCA()
    my_origin.write()

    # TGLR computing (normalized correlations)
    my_origin = ORIGIN.load('tmp')
    my_origin.step05_compute_TGLR(ncpu=1)
    my_origin.write()

    # segmap
    my_origin = ORIGIN.load('tmp')
    my_origin.step06_compute_segmentation_map(pfa=0.05)
    my_origin.write()

    # threshold applied on pvalues
    my_origin = ORIGIN.load('tmp')
    my_origin.step07_compute_purity_threshold()
    my_origin.write()

    my_origin = ORIGIN.load('tmp')
    my_origin.step08_detection()
    my_origin.write()

    my_origin = ORIGIN.load('tmp')
    my_origin.step09_detection_lost()
    my_origin.write()

    # estimation
    my_origin = ORIGIN.load('tmp', newname='tmp2')
    my_origin.step10_compute_spectra()
    my_origin.write()

    # list of source objects
    my_origin = ORIGIN.load('tmp2')
    cat = my_origin.step11_write_sources(ncpu=1)
    cat = my_origin.step11_write_sources(ncpu=2, overwrite=True)
    assert (len(cat) == 9)

    cat = Catalog.read('tmp2/tmp2.fits')
    assert (len(cat) == 9)

    # test returned sources are valid
    src = Source.from_file('./tmp2/sources/tmp2-00001.fits')
    Nz = np.array([sp.shape[0] for sp in src.spectra.values()])
    assert (len(np.unique(Nz)) == 1)
    Ny = np.array([ima.shape[0] for ima in src.images.values()])
    assert(len(np.unique(Ny)) == 1)
    Nx = np.array([ima.shape[1] for ima in src.images.values()])
    assert(len(np.unique(Nx)) == 1)
    Nz = np.unique(Nz)[0]
    Ny = np.unique(Ny)[0]
    Nx = np.unique(Nx)[0]
    cNz, cNy, cNx = src.cubes['MUSE_CUBE'].shape
    assert(cNy == Ny)
    assert(cNx == Nx)

    # Cleanup (try to close opened files)
    for h in my_origin._log_file.handlers:
        h.close()

    try:
        shutil.rmtree('tmp')
        shutil.rmtree('tmp2')
    except OSError:
        print('Failed to remove tmp directories')
