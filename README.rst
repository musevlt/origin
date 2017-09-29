=========================================================
ORIGIN: detectiOn and extRactIon of Galaxy emIssion liNes
=========================================================

This software has been developped by Carole Clastres and Antony Schutz
under the supervision of David Mary (Lagrange institute, University of Nice)
and ported to python by Laure Piqueras (CRAL).

The project is funded by the ERC MUSICOS (Roland Bacon, CRAL). Please contact
David for more info at david.mary@univ-lyon1.fr

origin.py contains an oriented-object interface to run the ORIGIN software.


Installation
============

Requires python packages mpdaf, yaml, joblib


Usage
=====

1- we import the package and create an ORIGIN object

 > from origin import ORIGIN
 
 > my_origin = ORIGIN.init('cube.fits', name='tmp')
 
 
2- we run the different steps

 > my_origin.step01_preprocessing()
    
 >  my_origin.step02_areas()
 
 >  my_origin.step03_compute_PCA_threshold()
    
 >  my_origin.step04_compute_greedy_PCA()
    
 >  my_origin.step05_compute_TGLR()
 
 >  my_origin.step06_threshold_pval()
    
 >  my_origin.step07_compute_spectra()
    
 >  my_origin.step08_spatiospectral_merging(deltaz)
    
 >  nsources = my_origin.step08_write_sources(ncpu=1)
 
 
3- Resulted detected sources can be load by using mpdaf

 > from mpdaf.sdetect import Source
 
 > src = Source.from_file('./tmp/sources/tmp-00001.fits')
 
 
4- at each step, we can write the results in a folder

 > my_origin.write()
 
 
5- it is possible to create an ORIGIN object from a previous session by loading
the data stored in the folder 

 > my_origin = ORIGIN.load('tmp')
 