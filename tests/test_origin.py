"""Test interface on ORIGIN software."""

from __future__ import absolute_import, division

import os
import shutil

from mpdaf.sdetect import Source, Catalog
from origin import ORIGIN

MINICUBE = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'minicube.fits')
SEGMAP = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                      'segmap.fits')


def test_origin():
    """test ORIGIN"""
    # Number of subcubes for the spatial segmentation

    try:
        my_origin = ORIGIN.init(MINICUBE, SEGMAP, name='tmp')
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
        # my_origin.step05_compute_TGLR(ncpu=1, NbSubcube=2)
        my_origin.write()

        # threshold applied on pvalues
        my_origin = ORIGIN.load('tmp')
        my_origin.step06_compute_purity_threshold()
        my_origin.write()

        my_origin = ORIGIN.load('tmp')
        my_origin.step07_detection()
        my_origin.write()

        my_origin = ORIGIN.load('tmp')
        my_origin.step08_detection_lost()
        my_origin.write()

        # estimation
        my_origin = ORIGIN.load('tmp', newname='tmp2')
        my_origin.step09_compute_spectra()
        my_origin.write()

        # cleaned results
        my_origin = ORIGIN.load('tmp2')
        my_origin.step10_clean_results()
        my_origin.write()

        # list of source objects
        my_origin = ORIGIN.load('tmp2')
        cat = my_origin.step12_write_sources(ncpu=1)
        cat = my_origin.step12_write_sources(ncpu=2, overwrite=True)
        assert len(cat) == 8

        cat = Catalog.read('tmp2/tmp2.fits')
        assert len(cat) == 8

        # test returned sources are valid
        src = Source.from_file('./tmp2/sources/tmp2-00001.fits')
        assert set(sp.shape[0] for sp in src.spectra.values()) == {3681}
        assert set(ima.shape for ima in src.images.values()) == {(25, 25)}
        assert src.cubes['MUSE_CUBE'].shape == (3681, 25, 25)
    finally:
        # Cleanup (try to close opened files)
        for h in my_origin._log_file.handlers:
            h.close()

        try:
            shutil.rmtree('tmp')
            shutil.rmtree('tmp2')
        except OSError:
            print('Failed to remove tmp directories')
