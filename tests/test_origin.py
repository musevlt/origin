import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from mpdaf.sdetect import Catalog, Source

from muse_origin import ORIGIN
from muse_origin.lib_origin import spatiospectral_merging

CURDIR = os.path.dirname(os.path.abspath(__file__))
MINICUBE = os.path.join(CURDIR, 'minicube.fits')
SEGMAP = os.path.join(CURDIR, 'segmap.fits')


def test_attrs(tmpdir):
    orig = ORIGIN.init(MINICUBE, name='orig', path=str(tmpdir))

    attrs = dir(orig)
    # Check step attributes
    assert 'Cat3_sources' in attrs
    assert 'cube_faint' in attrs
    # Check step methods
    assert 'step03_compute_PCA_threshold' in attrs
    # Check params
    assert 'threshold_correl' in attrs
    # missing attributes
    with pytest.raises(AttributeError):
        orig.foo_bar_baz


def test_init_load(tmpdir):
    orig = ORIGIN.init(MINICUBE, name='orig', path=str(tmpdir))
    orig.write()
    assert tmpdir.join('orig', 'orig.yaml').exists()

    newpath = tmpdir.join('new')
    os.makedirs(newpath)
    orig.write(path=str(newpath), erase=True)
    orig = ORIGIN.load(str(newpath.join('orig')))
    assert newpath.join('orig', 'orig.yaml').exists()


def test_psf(caplog, tmpdir):
    path = str(tmpdir)
    orig = ORIGIN.init(MINICUBE, name='tmp', loglevel='INFO', path=path)

    psffile = str(tmpdir.join('psf.fits'))
    fits.writeto(psffile, orig.PSF)

    # To build an ORIGIN with a PSF not present in the header, we must
    # pass the PSF file
    orig2 = ORIGIN.init(
        MINICUBE,
        name='tmp2',
        loglevel='INFO',
        path=path,
        PSF=psffile,
        FWHM_PSF=orig.FWHM_PSF,
        LBDA_FWHM_PSF=orig.LBDA_FWHM_PSF,
    )

    assert orig.param['FWHM PSF'] == orig2.param['FWHM PSF']
    assert orig.param['LBDA FWHM PSF'] == orig2.param['LBDA FWHM PSF']


def test_origin(caplog, tmpdir):
    """Test the full ORIGIN process."""

    orig = ORIGIN.init(MINICUBE, name='tmp', loglevel='INFO', path=str(tmpdir))
    orig.write()

    origfolder = str(tmpdir.join('tmp'))

    # test that log level is correctly reloaded, then change it
    orig = ORIGIN.load(origfolder)
    assert orig.logger.handlers[0].level == 20
    orig.set_loglevel('DEBUG')
    assert orig.logger.handlers[0].level == 10

    orig.step01_preprocessing()
    assert orig.ima_dct is not None
    assert orig.ima_std is not None
    orig.write()

    orig = ORIGIN.load(origfolder)
    orig.step02_areas(minsize=30, maxsize=60)
    assert orig.param['nbareas'] == 4
    assert list(np.unique(orig.areamap._data)) == [1, 2, 3, 4]
    orig.write()

    orig = ORIGIN.load(origfolder)
    assert orig.param['nbareas'] == 4
    orig.step03_compute_PCA_threshold()
    orig.step04_compute_greedy_PCA()

    # TGLR computing (normalized correlations)
    orig.step05_compute_TGLR(ncpu=1)
    # orig.step05_compute_TGLR(ncpu=1, NbSubcube=2)

    # threshold applied on pvalues
    orig.step06_compute_purity_threshold(purity=0.8)

    # FIXME: threshold is hardcoded for now
    orig.step07_detection(threshold=9.28, segmap=SEGMAP)

    # estimation
    orig.step08_compute_spectra()
    orig.write()

    cat = Catalog.read(str(tmpdir.join('tmp', 'Cat1.fits')))
    subcat = cat[cat['comp'] == 0]
    assert np.all(np.isnan(subcat['STD']))
    # Test that the columns mask is correct. To be tested when we switch
    # back to a masked table
    # assert np.all(subcat['T_GLR'].mask == False)
    # assert np.all(subcat['STD'].mask == True)

    # cleaned results
    orig = ORIGIN.load(origfolder, newname='tmp2')
    orig.step09_clean_results()
    orig.write()

    # check that the catalog version was saves
    assert "CAT3_TS" in Catalog.read(str(tmpdir.join('tmp2', 'Cat3_lines.fits'))).meta
    assert "CAT3_TS" in Catalog.read(str(tmpdir.join('tmp2', 'Cat3_sources.fits'))).meta

    # create masks
    origfolder2 = str(tmpdir.join('tmp2'))
    orig = ORIGIN.load(origfolder2)
    orig.step10_create_masks()
    orig.write()

    # list of source objects
    orig = ORIGIN.load(origfolder2)
    orig.step11_save_sources("0.1")
    orig.step11_save_sources("0.1", n_jobs=2, overwrite=True)

    orig.info()
    with open(orig.logfile) as f:
        log = f.read().splitlines()
        assert '11 Done' in log[-1]

    tbl = orig.timestat(table=True)
    assert len(tbl) == 12
    assert tbl.colnames == ['Step', 'Exec Date', 'Exec Time']

    caplog.clear()
    orig.timestat()
    rec = caplog.records
    assert rec[0].message.startswith('step01_preprocessing executed:')
    assert rec[10].message.startswith('step11_save_sources executed:')
    assert rec[11].message.startswith('*** Total run time:')

    caplog.clear()
    orig.stat()
    assert [rec.message for rec in caplog.records] == [
        'ORIGIN PCA pfa 0.01 Back Purity: 0.80 Threshold: 9.28 '
        'Bright Purity 0.80 Threshold 5.46',
        'Nb of detected lines: 16',
        'Nb of sources Total: 6 Background: 3 Cont: 3',
        'Nb of sources detected in faint (after PCA): 4 in std (before PCA): 2',
    ]

    cat = Catalog.read(str(tmpdir.join('tmp2', 'Cat3_lines.fits')))
    assert len(cat) == 16
    assert max(cat['ID']) == 6

    # test returned sources are valid
    src1 = Source.from_file(str(tmpdir.join('tmp2', 'sources', 'source-00001.fits')))
    src2 = Source.from_file(str(tmpdir.join('tmp2', 'sources', 'source-00002.fits')))
    # FIXME: check if this test is really useful
    # assert set(sp.shape[0] for sp in src.spectra.values()) == {22, 1100}
    assert {ima.shape for ima in src1.images.values()} == {(25, 25)}
    assert src1.cubes['MUSE_CUBE'].shape == (1100, 25, 25)
    assert "SRC_TS" in src1.header
    assert src1.header["CAT3_TS"] == src2.header["CAT3_TS"]
    assert src1.header["SRC_TS"] == src2.header["SRC_TS"]

    # Cleanup (try to close opened files to avoid warnings)
    for h in orig.logger.handlers:
        h.close()


def test_merging():
    segmap = fits.getdata(SEGMAP)
    inputs = Table(
        rows=[
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
            (20, 10, 749),
        ],
        names=['x0', 'y0', 'z0'],
    )
    inputs['area'] = segmap[inputs['y0'], inputs['x0']]

    out = spatiospectral_merging(inputs, tol_spat=3, tol_spec=5)

    dt = [
        ('x0', int),
        ('y0', int),
        ('z0', int),
        ('area', int),
        ('imatch', int),
        ('imatch2', int),
    ]
    expected = np.array(
        [
            (72, 49, 545, 0, 0, 0),
            (71, 49, 549, 0, 0, 0),
            (71, 47, 751, 0, 0, 0),
            (72, 45, 543, 0, 0, 0),
            (74, 44, 546, 0, 0, 0),
            (51, 44, 360, 0, 1, 1),
            (51, 44, 564, 0, 1, 1),
            (3, 15, 589, 0, 2, 2),
            (3, 15, 597, 0, 2, 2),
            (3, 15, 601, 0, 2, 2),
            (24, 12, 733, 1, 3, 3),
            (24, 15, 736, 1, 3, 4),
            (29, 11, 740, 1, 3, 5),
            (20, 10, 749, 1, 6, 6),
        ],
        dtype=dt,
    )

    assert np.array_equal(out.as_array(), expected)
