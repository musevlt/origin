#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:37:27 2017

@author: antonyschutz
"""

from origin import ORIGIN
from mpdaf.obj import Cube
import os
#%%
# data path for Nexpmap and cube
pathdata = ''
savename = 'beta2_V0'
# name of the catalogue of calibrators
calname = 'cat_cal_mini.fits'
#%%

expmapname = os.path.join(pathdata,'EXPMAP_UDF-10.fits')
cubename   = os.path.join(pathdata,'DATACUBE_UDF-10.fits')

expmapname = '/Users/antonyschutz/Documents/ORIG/sdetect_origin/EXPMAP_UDF-10.fits'
cubename   = '/Users/antonyschutz/Documents/ORIG/sdetect_origin/DATACUBE_UDF-10.fits'


#%%


expmap = Cube(expmapname).data.data
cube = Cube(cubename)

#%%
# reduce size of data

expmap = expmap[:,-80:,:80]
cube = cube[:,-80:,:80]
cube.write(savename+'.fits')

#%%
threshold = 7.
NCUBE = 1

# Ncube, same as before. Used to split cube in smaller spatiale cube. 
# as before it made the PCA more or less specialized (help more or less to
# modelize the content of data, as the sources of interest)
# as before it helps to reduce the computation time of the PCA
orig = ORIGIN.init(savename+'.fits', NCUBE, [0, 0, 0, 0], name=savename)
nl,ny,nx = orig.cube_raw.shape
#%%
# adding calibrators, one is inside a source 5 are radomly added
# the catalogue of calibrators is created and saved for later
orig.step00_init_calibrator( x = 20, y = 62, z = 2000, amp = 2, profil = 6 )
orig.step00_init_calibrator( random = 5, Cat_cal = 'add', 
                            save = True, name = calname )

# self.cube_raw is changed according to calibrators
orig.step00_add_calibrator(name = calname)

# pre processing of data, taking into account the Nexpmap X2=X*sqrt(Nexpmap)
# a dct of dct_order=order is applied on X2
# mean and variance are computed for each wavelenght (MX_imean(X2_i))
# data are now standardized   (SX_i = std(X2_i))
# X2_i = X2_i - MX_i
# X2_i = X2_i / SX_i
orig.step00_preprocessing(expmap, dct_order=10)

# greedy PCA based on O2 test. 
# * Noise_population: factor (1/Noise_population %) of spectra to estimate the 
# background (Spectra for O2test<=threshold_test) similarly
# Spectra for which O2test>threshold_test are called nuisance and sources.
# Eigen vectors are learned on nuisance and sources orthogonalized to 
# background. Principal eigen vector is used to clean the cube or subcube of 
# data: Background+Nuisances+Sources
# * mixing: if output and input of pca are mixed accoring to a continuum test
# performed on the output of pca 
orig.step01_compute_greedy_PCA(mixing = False, Noise_population = 50, 
                               threshold_test = 1)


#%%
orig.step02_compute_TGLR()
orig.step03_compute_pvalues(threshold,sky=False)
orig.step04_compute_ref_pix()
#orig.write()



#%%
orig.step05_compute_NBtests()
# shunt the T1 and T2 test
orig.step06_select_NBtests(thresh_T1=-1000, thresh_T2=-1000)
orig.step07_compute_spectra(T=2)
orig.step08_spatial_merging()
orig.step09_spectral_merging()
orig.step10_write_sources(author='',ncpu=24)
orig.write()



