#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:37:27 2017

@author: antonyschutz
"""
from origin import ORIGIN
#%%
# data path for Nexpmap and cube
pathdata = ''
savename = 'Dev_version'
# name of the catalogue of calibrators

#%%
threshold = 8.
NCUBE = 1

cubename = '/Users/antonyschutz/Documents/python_perso/Origin_beta_2/WithZeroes/mosaic_extract.fits'

# To work with self computed variance, expmap is for now necessary
expmapname = '/Users/antonyschutz/Documents/python_perso/Origin_beta_2/WithZeroes/expmap_mosaic_extract.fits'

# Ncube, same as before. Used to split cube in smaller spatiale cube. 
# as before it made the PCA more or less specialized (help more or less to
# modelize the content of data, as the sources of interest)
# as before it helps to reduce the computation time of the PCA
orig = ORIGIN.init(cubename, NCUBE, name=savename)

#%% Origin VARIANCE or self Computed VARIANCE with expmap

# Origin VARIANCE
orig.step00_preprocessing()

# self Computed VARIANCE
#orig.step00_preprocessing(expmap = expmapname)

#%%
# greedy PCA based on O2 test. 
# * Noise_population: factor (1/Noise_population %) of spectra to estimate the 
# background (Spectra for O2test<=threshold_test) similarly
# Spectra for which O2test>threshold_test are called nuisance and sources.
# Eigen vectors are learned on nuisance and sources orthogonalized to 
# background. Principal eigen vector is used to clean the cube or subcube of 
# data: Background+Nuisances+Sources
# * mixing: if output and input of pca are mixed accoring to a continuum test
# performed on the output of pca 
#%%
orig.step01_compute_greedy_PCA()

#%%
orig.step02_compute_TGLR()

#%%
orig.step03_compute_pvalues(threshold)
orig.step04_compute_ref_pix()
#%%
orig.step07_compute_spectra()
orig.step08_spatial_merging()
orig.step09_spectral_merging(deltaz=0)
orig.step10_write_sources(author='',ncpu=24)
orig.write()
