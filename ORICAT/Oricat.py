#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:10:11 2017

@author: antonyschutz
"""
from mpdaf.sdetect import Catalog
import matplotlib.pyplot as plt
import DrawAndToDraw as DRAW
import CatTools as CT
import numpy as np
import os

#%%
#==============================================================================
# MAIN
#==============================================================================
plt.close('all')
#%% PARAMETERS
Radius = .5
snr_thresh = 3
peak_thresh = 8
band = 50


#%% PATH to catalog and data
path_cube = '/Users/antonyschutz/Documents/ORIG/DATACUBE_UDF-10.fits'

# Ground Truth Catalog
path_gt = 'GroundTruth/udf10_c031_e029_withz.fits'

# Main Path where data are
namedata1 = 'PCA063'
namedata2 = 'CLEANPCA063'
#namedata2 = 'AR10'
#namedata2 = 'Median51'

# Catalog files from Origin
path_method_1 = os.path.join('results', namedata1, namedata1 + '.fits')
path_method_2 = os.path.join('results', namedata2, namedata2 + '.fits')

# Correlation files from Origin
path_Correl1 = os.path.join('results', namedata1, 'cube_correl.fits')
path_Correl2 = os.path.join('results', namedata2, 'cube_correl.fits')

#%% DATA CUBE
if not 'cube_std_a' in locals():
    cube_std_a = CT.create_cubestd(path_cube)
    wcube_std = np.sum(cube_std_a, axis=0)
if not 'max_map' in locals():
    max_map = np.sum(cube_std_a, axis=0)
if not 'correl1' in locals():
    correl1, freq, sky2pix = CT.create_correl_data(path_Correl1)
    wcorrel1 = np.amax(correl1, axis=0)
    correl2, freq, sky2pix = CT.create_correl_data(path_Correl2)
    wcorrel2 = np.amax(correl2, axis=0)

Nl, Ny, Nx = cube_std_a.shape

#%%
# read catalog files and reduction to useful data
catgt = Catalog.read(path_gt)
cato1 = Catalog.read(path_method_1)
cato2 = Catalog.read(path_method_2)

print('')
print('Lines name for Lines selection')
[print(f) for f in catgt.colnames if f.rfind('LBDA_OBS') != -1]
print('')


LineList = []
LineList.append('LYALPHA_LBDA_OBS')
LineList.append('OII3726_LBDA_OBS')
LineList.append('OII3729_LBDA_OBS')

cat0, ID_thrash = CT.GroundTruth_selection(catgt, snr_thresh=snr_thresh, LineList=LineList)

# Matching ID in N*3 Tab
ID, DIST = CT.CREATE_ID(cat0, cato1, cato2, Radius=Radius)

# reduce catalog size
redcat_gt, keyslbd = CT.Reduce_catGT(cat0)
redcat_o1 = CT.Reduce_catOR(cato1)
redcat_o2 = CT.Reduce_catOR(cato2)

# Compute all y,x pixels from all Ra Dec (ID with pixels and tab pixel)
position, IDfromyx, radec = CT.RaDec2Pix(ID, redcat_gt, redcat_o1, redcat_o2, sky2pix)

# SNR Selection by thresholding after thresholding
###ID,ID_thrash = SNR_selection(ID,redcat_gt,snr_thresh=snr_thresh)

# Assign Lines from Ground Truth to catalog
Line, Name, SNR = CT.Match_Lines_Name(ID, redcat_gt,
                                      redcat_o1, redcat_o2, freq, snr_thresh)

# Find Peak and find closer candidate to give a name to the founded peak
Name, Line = CT.LineNameSearch(ID, correl1, correl2, freq, Line, Name, position,
                               Line_Band=band, Peak_Thresh=peak_thresh)

match = CT.dicomatch(ID)

# Counter number of matched lines et information
Counter = CT.CountName(ID, Name, keyslbd)

#%%
# draw general statistics for GT and both methods
path = os.path.join('results', namedata2, 'result_' + namedata1 + '_' + namedata2)
if not os.path.isdir(path):
    os.mkdir(path)


plt.close("all")

if True:
    DRAW.plotIDstat(ID, ID_thrash, save=True,
                    path=path, show=True)
    DRAW.bothplotbarGT(Counter, snr_thresh=snr_thresh, save=True,
                       path=path, show=True)
    DRAW.bothplotbarGTnm(Counter, snr_thresh=snr_thresh, save=True,
                         path=path, show=True)
    DRAW.matchimg(ID, position, max_map, wcorrel1, wcorrel2, wcube_std,
                  sat_tresh_list=(.05, .05), save=True, path=path,
                  show=True, colormap='jet')

if False:
    DRAW.drawsources(ID, correl1, correl2, freq, Line, Name, position,
                     SNR, DIST, IDfromyx, match, cube_std_a, radec, max_map,
                     Line_Band=band, Peak_Thresh=peak_thresh, snr_thresh=snr_thresh,
                     save=True, path=path, colormap='jet')

#%% DEV PART

# plt.close('all')
#
# DRAW.drawsources(ID[0:1],correl1,correl2,freq,Line,Name,position,
#                 SNR, DIST, IDfromyx, match, cube_std_a, radec, max_map,
#            Line_Band=band,Peak_Thresh=peak_thresh,snr_thresh=snr_thresh,
#            save=True,show=False,path=path,colormap='jet')
