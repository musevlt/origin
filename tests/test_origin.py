"""Test interface on ORIGIN software."""

import numpy as np
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
        my_origin = ORIGIN.init(MINICUBE, SEGMAP, name='tmp', loglevel='INFO')
        my_origin.write()

        my_origin = ORIGIN.load('tmp')
        # test that log level is correctly reloaded, then change it
        assert my_origin.logger.handlers[0].level == 20
        my_origin.set_loglevel('DEBUG')
        assert my_origin.logger.handlers[0].level == 10

        # FIXME: dct_approx=False does not work with the test dataset
        my_origin.step01_preprocessing(dct_approx=True)
        assert my_origin.ima_dct is not None
        assert my_origin.ima_std is not None
        my_origin.write()

        my_origin = ORIGIN.load('tmp')
        my_origin.step02_areas()
        assert my_origin.param['nbareas'] == 1
        assert list(np.unique(my_origin.areamap._data)) == [1]
        my_origin.write()

        my_origin = ORIGIN.load('tmp')
        assert my_origin.param['nbareas'] == 1
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

        # create masks
        my_origin = ORIGIN.load('tmp2')
        my_origin.step11_create_masks()
        my_origin.write()

        # list of source objects
        my_origin = ORIGIN.load('tmp2')
        my_origin.step12_save_sources("0.1")
        my_origin.step12_save_sources("0.1", n_jobs=2, overwrite=True)

        my_origin.info()

        cat = Catalog.read('tmp2/tmp2.fits')
        assert len(cat) == 8

        # test returned sources are valid
        src = Source.from_file('./tmp2/sources/source-00001.fits')
        assert set(sp.shape[0] for sp in src.spectra.values()) == {18, 3681}
        assert set(ima.shape for ima in src.images.values()) == {(25, 25)}
        assert src.cubes['MUSE_CUBE'].shape == (3681, 25, 25)
    finally:
        # Cleanup (try to close opened files)
        for h in my_origin.logger.handlers:
            h.close()

        try:
            shutil.rmtree('tmp')
            shutil.rmtree('tmp2')
        except OSError:
            print('Failed to remove tmp directories')
