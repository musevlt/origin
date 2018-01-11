
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:37:27 2017

@author: antonyschutz
"""

#from mpdaf.sdetect import Catalog

#import numpy as np

from origin import ORIGIN
from mpdaf.obj import Cube
#from scipy import stats

#import sys
#%%

#meth = sys.argv[1]
#ncube = sys.argv[2]

# Methodes:
#    1 origin r0=,63 3*3 cubes
#    2 beta 1, criteria usual, 1*1 cube no mixing
#    3 beta 1, criteria usual, 1*1 cube with mixing
#    4 beta 1, criteria usual, 3*3 cube no mixing
#    5 beta 1, criteria usual, 3*3 cube with mixing

#%%

#expmapname = '/home/aschutz/ORIG/sdetect_origin/EXPMAP_UDF-10.fits'
#cubename   = '/home/aschutz/ORIG/sdetect_origin/DATACUBE_UDF-10.fits'

expmapname = 'EXPMAP_UDF-10.fits'
cubename = 'DATACUBE_UDF-10.fits'

#expmapname = '/Users/antonyschutz/Documents/ORIG/sdetect_origin/EXPMAP_UDF-10.fits'
#cubename   = '/Users/antonyschutz/Documents/ORIG/sdetect_origin/DATACUBE_UDF-10.fits'


threshold = 7.

#%%
expmap = Cube(expmapname).data.data
cube = Cube(cubename)

#expmap = expmap[:,-80:,:80]
#cube = cube[:,-80:,:80]

#%%


calname = 'cat_cal.fits'

for meth in ('1', '2', '3', '4', '5'):

    name = 'v5_beta2_V' + str(meth) + '.fits'

    if meth == '1':  # normal origin r0=.63 ncube = 3

        ncube = 3
        r0 = 0.63
        cube.write(name)
        orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
        print(meth)
        print('add')
        orig.step00_add_calibrator(name=calname)
        print('added - PCA')
        orig.step01_compute_PCA(r0)
        print('PCAed')
        orig.step02_compute_TGLR()
        orig.step03_compute_pvalues(threshold, sky=False)
        orig.step04_compute_ref_pix()
        orig.write()
        orig.step05_compute_NBtests()
        orig.step06_select_NBtests()
        orig.step07_compute_spectra(T=2)
        orig.step08_spatial_merging()
        orig.step09_spectral_merging()
        orig.step10_write_sources(author='', ncpu=24)
        orig.write()

    if meth == '2':  # beta 2 mixing False Ncube=1
        ncube = 1
        cube.write(name)
        orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
        orig.step00_add_calibrator(name=calname)
        orig.step00_preprocessing(expmap)
        orig.step01_compute_greedy_PCA(mixing=False)
        orig.step02_compute_TGLR()
        orig.step03_compute_pvalues(threshold, sky=False)
        orig.step04_compute_ref_pix()
        orig.write()
        orig.step05_compute_NBtests()
        orig.step06_select_NBtests()
        orig.step07_compute_spectra(T=2)
        orig.step08_spatial_merging()
        orig.step09_spectral_merging()
        orig.step10_write_sources(author='', ncpu=24)
        orig.write()

    if meth == '3':  # beta 2 mixing True Ncube=1
        ncube = 1
        cube.write(name)
        orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
        orig.step00_add_calibrator(name=calname)
        orig.step00_preprocessing(expmap)
        orig.step01_compute_greedy_PCA(mixing=True)
        orig.step02_compute_TGLR()
        orig.step03_compute_pvalues(threshold, sky=False)
        orig.step04_compute_ref_pix()
        orig.write()
        orig.step05_compute_NBtests()
        orig.step06_select_NBtests()
        orig.step07_compute_spectra(T=2)
        orig.step08_spatial_merging()
        orig.step09_spectral_merging()
        orig.step10_write_sources(author='', ncpu=24)
        orig.write()

    if meth == '4':  # beta 2 mixing False Ncube=3
        ncube = 3
        cube.write(name)
        orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
        orig.step00_add_calibrator(name=calname)
        orig.step00_preprocessing(expmap)
        orig.step01_compute_greedy_PCA(mixing=False)
        orig.step02_compute_TGLR()
        orig.step03_compute_pvalues(threshold, sky=False)
        orig.step04_compute_ref_pix()
        orig.write()
        orig.step05_compute_NBtests()
        orig.step06_select_NBtests()
        orig.step07_compute_spectra(T=2)
        orig.step08_spatial_merging()
        orig.step09_spectral_merging()
        orig.step10_write_sources(author='', ncpu=24)
        orig.write()

    if meth == '5':  # beta 2 mixing True Ncube=3
        ncube = 3
        cube.write(name)
        orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
        orig.step00_add_calibrator(name=calname)
        orig.step00_preprocessing(expmap)
        orig.step01_compute_greedy_PCA(mixing=True)
        orig.step02_compute_TGLR()
        orig.step03_compute_pvalues(threshold, sky=False)
        orig.step04_compute_ref_pix()
        orig.write()
        orig.step05_compute_NBtests()
        orig.step06_select_NBtests()
        orig.step07_compute_spectra(T=2)
        orig.step08_spatial_merging()
        orig.step09_spectral_merging()
        orig.step10_write_sources(author='', ncpu=24)
        orig.write()


# -*- coding: utf-8 -*-
#"""
# Created on Fri Mar 24 15:37:27 2017
#
#@author: antonyschutz
#"""
#
#from mpdaf.sdetect import Catalog
#
#import numpy as np
#
#from origin import ORIGIN
#from mpdaf.obj import Cube
#from scipy import stats
#
#import sys
# %%
#
#meth = sys.argv[1]
##ncube = sys.argv[2]
#
# Methodes:
# 1 origin r0=,63 3*3 cubes
# 2 beta 1, criteria usual, 1*1 cube no mixing
# 3 beta 1, criteria usual, 1*1 cube with mixing
# 4 beta 1, criteria usual, 3*3 cube no mixing
# 5 beta 1, criteria usual, 3*3 cube with mixing
#
# %%
#
# print(meth)
#
##expmapname = '/home/aschutz/ORIG/sdetect_origin/EXPMAP_UDF-10.fits'
##cubename   = '/home/aschutz/ORIG/sdetect_origin/DATACUBE_UDF-10.fits'
#
#expmapname = '/Users/antonyschutz/Documents/ORIG/sdetect_origin/EXPMAP_UDF-10.fits'
#cubename   = '/Users/antonyschutz/Documents/ORIG/sdetect_origin/DATACUBE_UDF-10.fits'
#
#
#threshold = 7.
#
# %%
#expmap = Cube(expmapname).data.data
#cube = Cube(cubename)
#
#expmap = expmap[:,-80:,:80]
#cube = cube[:,-80:,:80]
#
# %%
#
#name = 'v5_beta2_V'+str(meth)+'.fits'
#calname = 'cat_cal.fits'
#
# if meth=='1': # normal origin r0=.63 ncube = 3
#
#    ncube = 3
#    r0 = 0.63
#    cube.write(name)
#    orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
#    orig.step00_add_calibrator(name=calname)
#    orig.step01_compute_PCA(r0)
#
#
# if meth=='2': # beta 2 mixing False Ncube=1
#    ncube = 1
#    cube.write(name)
#    orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
#    orig.step00_add_calibrator(name=calname)
#    orig.step00_preprocessing(expmap)
#    orig.step01_compute_greedy_PCA(mixing=False)
#
# if meth=='3': # beta 2 mixing True Ncube=1
#    ncube = 1
#    cube.write(name)
#    orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
#    orig.step00_add_calibrator(name=calname)
#    orig.step00_preprocessing(expmap)
#    orig.step01_compute_greedy_PCA(mixing=True)
#
# if meth=='4': # beta 2 mixing False Ncube=3
#    ncube = 3
#    cube.write(name)
#    orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
#    orig.step00_add_calibrator(name=calname)
#    orig.step00_preprocessing(expmap)
#    orig.step01_compute_greedy_PCA(mixing=False)
#
# if meth=='5': # beta 2 mixing True Ncube=3
#    ncube = 3
#    cube.write(name)
#    orig = ORIGIN.init(name, ncube, [0, 0, 0, 0], name=name)
#    orig.step00_add_calibrator(name=calname)
#    orig.step00_preprocessing(expmap)
#    orig.step01_compute_greedy_PCA(mixing=True)
#
#
# orig.step02_compute_TGLR()
# orig.write()
# orig.step03_compute_pvalues(threshold,sky=False)
# orig.step04_compute_ref_pix()
# orig.write()
# orig.step05_compute_NBtests()
# orig.step06_select_NBtests()
# orig.step07_compute_spectra(T=2)
# orig.step08_spatial_merging()
# orig.step09_spectral_merging()
# orig.step10_write_sources(author='',ncpu=24)
# orig.write()
#
