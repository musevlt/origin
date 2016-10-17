# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:05:06 2016

@author: piqueras
"""

from astropy.io import fits
import numpy as np
from scipy.io import loadmat

Dico = loadmat('origin/Dico_FWHM_2_12.mat')['Dico']
hdulist = fits.HDUList([fits.PrimaryHDU()])
fwhm = np.linspace(2, 12, 20)
for i in range(Dico.shape[1]):
    hdu = fits.ImageHDU(name='PROF%02d'%i, data=Dico[:,i])
    hdu.header['FWHM'] = (fwhm[i], 'FWHM in pixels')
    hdulist.append(hdu)
hdulist.writeto('origin/Dico_FWHM_2_12.fits')