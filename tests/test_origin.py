"""Test interface on ORIGIN software."""

import numpy as np
import os
import shutil

from astropy.io import fits
from astropy.table import Table
from mpdaf.sdetect import Source, Catalog
from origin import ORIGIN
from origin.lib_origin import spatiospectral_merging

MINICUBE = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                        'minicube.fits')
SEGMAP = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                      'segmap.fits')


def test_origin():
    """test ORIGIN"""
    # Number of subcubes for the spatial segmentation

    try:
        my_origin = ORIGIN.init(MINICUBE, name='tmp', loglevel='INFO')
        my_origin.write()

        # test that log level is correctly reloaded, then change it
        my_origin = ORIGIN.load('tmp')
        assert my_origin.logger.handlers[0].level == 20
        my_origin.set_loglevel('DEBUG')
        assert my_origin.logger.handlers[0].level == 10

        my_origin.step01_preprocessing()
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
        my_origin.step04_compute_greedy_PCA()

        # TGLR computing (normalized correlations)
        my_origin.step05_compute_TGLR(ncpu=1)
        # my_origin.step05_compute_TGLR(ncpu=1, NbSubcube=2)

        # threshold applied on pvalues
        my_origin.step06_compute_purity_threshold(purity=0.8)

        # FIXME: threshold is hardcoded for now
        my_origin.step07_detection(threshold=9.28, segmap=SEGMAP)

        # estimation
        my_origin.step08_compute_spectra()
        my_origin.write()

        cat = Catalog.read('tmp/Cat1.fits')
        subcat = cat[cat['comp'] == 0]
        assert np.all(np.isnan(subcat['STD']))
        # Test that the columns mask is correct. To be tested when we switch
        # back to a masked table
        # assert np.all(subcat['T_GLR'].mask == False)
        # assert np.all(subcat['STD'].mask == True)

        # cleaned results
        my_origin = ORIGIN.load('tmp', newname='tmp2')
        my_origin.step09_clean_results()
        my_origin.write()

        # check that the catalog version was saves
        assert "CAT3_TS" in Catalog.read('tmp2/Cat3_lines.fits').meta
        assert "CAT3_TS" in Catalog.read('tmp2/Cat3_sources.fits').meta

        # create masks
        my_origin = ORIGIN.load('tmp2')
        my_origin.step10_create_masks()
        my_origin.write()

        # list of source objects
        my_origin = ORIGIN.load('tmp2')
        my_origin.step11_save_sources("0.1")
        my_origin.step11_save_sources("0.1", n_jobs=2, overwrite=True)

        my_origin.info()

        cat = Catalog.read('tmp2/tmp2.fits')
        assert len(cat) == 8

        # test returned sources are valid
        src1 = Source.from_file('./tmp2/sources/source-00001.fits')
        src2 = Source.from_file('./tmp2/sources/source-00002.fits')
        # FIXME: check if this test is really useful
        # assert set(sp.shape[0] for sp in src.spectra.values()) == {22, 1100}
        assert set(ima.shape for ima in src1.images.values()) == {(25, 25)}
        assert src1.cubes['MUSE_CUBE'].shape == (1100, 25, 25)
        assert "SRC_TS" in src1.header
        assert src1.header["CAT3_TS"] == src2.header["CAT3_TS"]
        assert src1.header["SRC_TS"] == src2.header["SRC_TS"]
    finally:
        # Cleanup (try to close opened files)
        for h in my_origin.logger.handlers:
            h.close()

        try:
            shutil.rmtree('tmp')
            shutil.rmtree('tmp2')
        except OSError:
            print('Failed to remove tmp directories')


def test_merging():
    segmap = fits.getdata(SEGMAP)
    inputs = Table(rows=[
        # First source
        (72, 49, 545),
        (71, 49, 549),
        (71, 47, 751),
        (72, 45, 543),
        # close line, should be merged
        (74, 44, 546),

        (51, 44, 360),
        (51, 44, 564),

        (3, 15, 589),
        (3, 15, 597),
        (3, 15, 601),
        # in a segmap region
        (24, 12, 733),
        (24, 15, 736),
        (29, 11, 740),
        (20, 10, 749)
    ], names=['x0', 'y0', 'z0'])
    inputs['area'] = segmap[inputs['y0'], inputs['x0']]

    out = spatiospectral_merging(inputs, tol_spat=3, tol_spec=5)

    dt = [('x0', int), ('y0', int), ('z0', int),
          ('area', int), ('imatch', int), ('imatch2', int)]
    expected = np.array([
        (72, 49, 545, 0, 0, 0),
        (71, 49, 549, 0, 0, 0),
        (71, 47, 751, 0, 0, 0),
        (72, 45, 543, 0, 0, 0),
        (74, 44, 546, 0, 0, 0),
        (51, 44, 360, 0, 1, 1),
        (51, 44, 564, 0, 1, 1),
        (3,  15, 589, 0, 2, 2),
        (3,  15, 597, 0, 2, 2),
        (3,  15, 601, 0, 2, 2),
        (24, 12, 733, 1, 3, 3),
        (24, 15, 736, 1, 3, 4),
        (29, 11, 740, 1, 3, 5),
        (20, 10, 749, 1, 6, 6)
    ], dtype=dt)

    assert np.array_equal(out.as_array(), expected)
