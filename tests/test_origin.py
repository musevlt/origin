"""Test interface on ORIGIN software."""

from __future__ import absolute_import, division

import numpy as np
import os
import shutil

from mpdaf.sdetect import Source
from origin import ORIGIN

MINICUBE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                        'minicube.fits')

def test_origin():
    """test ORIGIN"""
    # Number of subcubes for the spatial segmentation
    NbSubcube = 1

    my_origin = ORIGIN(MINICUBE, NbSubcube, [2, 2, 1, 3])

    # Coefficient of determination for projection during PCA
    r0 = 0.67
    # PCA
    cube_faint, cube_cont = my_origin.compute_PCA(r0)

    # TGLR computing (normalized correlations)
    correl, profile = my_origin.compute_TGLR(cube_faint)

    # threshold applied on pvalues
    threshold = 8
    # compute pvalues
    cube_pval_correl, cube_pval_channel, cube_pval_final = \
                               my_origin.compute_pvalues(correl, threshold)

    # Connectivity of contiguous voxels
    neighboors = 26
    # Compute connected voxels and their referent pixels
    Cat0 = my_origin.compute_ref_pix(correl, profile, cube_pval_correl,
                              cube_pval_channel, cube_pval_final,
                              neighboors)

    # Number of the spectral ranges skipped to compute the controle cube
    nb_ranges = 3
    # Narrow band tests
    Cat1 = my_origin.compute_NBtests(Cat0, nb_ranges)
    # Thresholded narrow bands tests
    thresh_T1 = .2
    thresh_T2 = 2

    Cat1_T1, Cat1_T2 = my_origin.select_NBtests(Cat1, thresh_T1,
                                                   thresh_T2)

    # Estimation with the catalogue from the narrow band Test number 2
    Cat2_T2, Cat_est_line = \
    my_origin.estimate_line(Cat1_T2, profile, cube_faint)

    # Spatial merging
    Cat3 = my_origin.merge_spatialy(Cat2_T2)

    # Distance maximum between 2 different lines (in pixels)
    deltaz = 1
    # Spectral merging
    Cat4 = my_origin.merge_spectraly(Cat3, Cat_est_line, deltaz)

    # list of source objects
    nsources = my_origin.write_sources(Cat4, Cat_est_line, correl, ncpu=2, name='tmp')
    assert (nsources == 2)
    
    # test returned sources are valid
    src = Source.from_file('./tmp/tmp-00001.fits')
    Nz = np.array([sp.shape[0] for sp in src.spectra.values()])
    assert_equal(len(np.unique(Nz)), 1)
    Ny = np.array([ima.shape[0] for ima in src.images.values()])
    assert_equal(len(np.unique(Ny)), 1)
    Nx = np.array([ima.shape[1] for ima in src.images.values()])
    assert_equal(len(np.unique(Nx)), 1)
    Nz = np.unique(Nz)[0]
    Ny = np.unique(Ny)[0]
    Nx = np.unique(Nx)[0]
    cNz, cNy, cNx = src.cubes['MUSE_CUBE'].shape
    assert_equal(cNy, Ny)
    assert_equal(cNx, Nx)
    shutil.rmtree('./tmp')
