"""Test interface on ORIGIN software."""

from __future__ import absolute_import, division

import numpy as np
import os
import shutil

from mpdaf.sdetect import Source, Catalog
from origin import ORIGIN

MINICUBE = os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                        'minicube.fits')
EXPMAP = os.path.join(os.path.abspath(os.path.dirname(__file__)), 
                        'miniexpmap.fits')

def test_origin():
    """test ORIGIN"""
    # Number of subcubes for the spatial segmentation
    NbSubcube = 1

    my_origin = ORIGIN.init(MINICUBE, EXPMAP, NbSubcube, name='tmp')
    my_origin.write()

#    # Coefficient of determination for projection during PCA
#    r0 = 0.67
#    # PCA
#    my_origin = ORIGIN.load('tmp')
#    my_origin.step01_compute_PCA(r0)
#    my_origin.write()
    
    my_origin = ORIGIN.load('tmp')
    my_origin.step00_preprocessing()
    my_origin.write()
    
    my_origin = ORIGIN.load('tmp')
    my_origin.step01_compute_greedy_PCA()
    my_origin.write()

    # TGLR computing (normalized correlations)
    my_origin = ORIGIN.load('tmp')
    my_origin.step02_compute_TGLR()
    my_origin.write()

    # threshold applied on pvalues
    threshold = 8
    # compute pvalues
    my_origin = ORIGIN.load('tmp')
    my_origin.step03_compute_pvalues(threshold)
    my_origin.write()

    # Connectivity of contiguous voxels
    neighboors = 26
    # Compute connected voxels and their referent pixels
    my_origin = ORIGIN.load('tmp')
    my_origin.step04_compute_ref_pix(neighboors)
    my_origin.write()

#    # Number of the spectral ranges skipped to compute the controle cube
#    nb_ranges = 3
#    # Narrow band tests
    my_origin = ORIGIN.load('tmp', newname='tmp2')
#    my_origin.step05_compute_NBtests(nb_ranges)
#    my_origin.write()
#    
#    # Thresholded narrow bands tests
#    thresh_T1 = .2
#    thresh_T2 = 2
#    my_origin = ORIGIN.load('tmp2')
#    my_origin.step06_select_NBtests(thresh_T1, thresh_T2)
#    my_origin.write()

    # Estimation with the catalogue from the narrow band Test number 2
#    my_origin = ORIGIN.load('tmp2')
    my_origin.step07_compute_spectra()
    my_origin.write()

    # Spatial merging
    my_origin = ORIGIN.load('tmp2')
    my_origin.step08_spatial_merging()
    my_origin.write()

    # Distance maximum between 2 different lines (in pixels)
    deltaz = 1
    # Spectral merging
    my_origin = ORIGIN.load('tmp2')
    my_origin.step09_spectral_merging(deltaz)
    my_origin.write()
    
    # list of source objects
    my_origin = ORIGIN.load('tmp2')
    nsources = my_origin.step10_write_sources(ncpu=1)
    assert (nsources == 9) 
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
    shutil.rmtree('tmp')
    shutil.rmtree('tmp2')
