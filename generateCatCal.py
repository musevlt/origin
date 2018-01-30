# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:59:36 calx117

@author: antonyschutz
"""

import matplotlib.pyplot as plt

from mpdaf.sdetect import Catalog

import numpy as np

from origin import ORIGIN
from mpdaf.obj import Cube
from scipy import stats
import os

#%%

cubename = '/Users/antonyschutz/Documents/ORIG/sdetect_origin/DATACUBE_UDF-10.fits'


caly1 = 116
calx1 = 213
calz1 = 2000

caly2 = 302
calx2 = 27
calz2 = 1700

caly3 = 4
calx3 = 67
calz3 = 1600

caly4 = 180
calx4 = 104
calz4 = 1500

caly5 = 318
calx5 = 185
calz5 = 1500
#%%
name = 'tmp'
cube = Cube(cubename)

calname = 'cat_cal.fits'
#%%

NCUBE = 1
orig = ORIGIN.init(cubename, NCUBE, [0, 0, 0, 0], name=name)
orig.step00_init_calibrator(x=calx1, y=caly1, z=calz1, amp=2, profil=6)
orig.step00_init_calibrator(x=calx2, y=caly2, z=calz2, amp=1, profil=6, Cat_cal='add')
orig.step00_init_calibrator(x=calx3, y=caly3, z=calz3, amp=1, profil=6, Cat_cal='add')
orig.step00_init_calibrator(x=calx4, y=caly4, z=calz4, amp=1, profil=6, Cat_cal='add')
orig.step00_init_calibrator(x=calx5, y=caly5, z=calz5, amp=1, profil=6, Cat_cal='add')
orig.step00_init_calibrator(random=20, Cat_cal='add', save=True, name=calname)
