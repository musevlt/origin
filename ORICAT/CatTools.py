#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:57:20 2016

@author: antonyschutz
"""
from mpdaf.obj import Cube
import funpeakdet as fpd
import numpy as np

#%%
#==============================================================================
# CREATE WHOLE ID, reduced Catalog in terms of SNR and Radius for matching
#==============================================================================


def CREATE_ID(cat0, cat1, cat2, Radius=.5):
    ''' Create ID correspondance between catalogs ground truth for 2 methods. 
    Return ID with columns corresponding to : 
        ID1: ID in Ground truth (cat0) 
        ID2: ID in first method (cat1)
        ID3: ID in second method (cat2)

        If cat0, cat1 and cat2 match (catalog.match): 
            they have source which are closer than "radius" 
            ID = [ID1,ID2,ID3]
        If a ground truth source is not found by any methods: 
            ID = [ID1,0,0] 
        or similarly if only one method find the ground truth:
            ID = [ID1,ID2,0] or ID = [ID1,0,ID2]
        If there is correspondance between the two method, if they find 
        the same source which is not in the ground truth:
            ID = [0,ID2,ID3]
        If only one method find a source which is not in the ground truth: 
            ID = [0,ID2,0] or ID = [0,0,ID3]'''

    cat1cat0match, cat1cat0nomatch1, cat1cat0nomatch2 = cat1.match(cat0, Radius)
    cat2cat0match, cat2cat0nomatch1, cat2cat0nomatch2 = cat2.match(cat0, Radius)
    cat1cat2match, cat1cat2nomatch1, cat1cat2nomatch2 = cat1cat0nomatch1.match(
        cat2cat0nomatch1, Radius)

    ID_gt = [f for f in cat0['ID']]

    ID_cat1 = np.array([f for f in cat1cat0match['ID_1']])
    Dist1 = np.array([f for f in cat1cat0match['Distance']])
    ID_cat0incat1 = np.array([f for f in cat1cat0match['ID_2']])

    ID_cat2 = np.array([f for f in cat2cat0match['ID_1']])
    Dist2 = np.array([f for f in cat2cat0match['Distance']])
    ID_cat0incat2 = np.array([f for f in cat2cat0match['ID_2']])

    ID1 = [f for f in cat1cat2nomatch1['ID']]
    ID2 = [f for f in cat1cat2nomatch2['ID']]

    ID_cat1cat2 = [f for f in cat1cat2match['ID_1']]
    ID_cat1incat2 = [f for f in cat1cat2match['ID_2']]

    ID = []
    # INFO: contain all additional information from matching
    # 1°) Distance
    Distance = []
    for gt in ID_gt:

        n = np.where(ID_cat0incat1 == gt)[0]
        o1 = ID_cat1[n][0] if len(ID_cat1[n]) > 0 else 0
        d1 = Dist1[n][0] if len(ID_cat1[n]) > 0 else '-'

        n = np.where(ID_cat0incat2 == gt)[0]
        o2 = ID_cat2[n][0] if len(ID_cat2[n]) > 0 else 0
        d2 = Dist2[n][0] if len(ID_cat2[n]) > 0 else '-'

        ID.append((gt, o1, o2))
        Distance.append((d1, d2))

    for n, ID_cat in enumerate(ID_cat1cat2):
        ID.append((0, ID_cat, ID_cat1incat2[n]))
        Distance.append(('-', '-'))

    for ID_cat in ID1:
        ID.append((0, ID_cat, 0))
        Distance.append(('-', '-'))

    for ID_cat in ID2:
        ID.append((0, 0, ID_cat))
        Distance.append(('-', '-'))

    return ID, Distance

# return reduced catalog


def Reduce_catGT(cat):
    ''' reduce the ground truth catalog to ID RA DEC and lines informations
    including SNR'''
    keyslbd = [f for f in cat.colnames if f.rfind('LBDA_OBS') != -1]
    keyssnr = [f for f in cat.colnames if f.rfind('_SNR') != -1]
    keys = ['ID', 'RA', 'DEC'] + keyslbd + keyssnr
    return cat[keys], keyslbd


def Reduce_catOR(cat):
    ''' reduce Origin catalog to ID RA DEC and lines informations'''
    keyslbd = [f for f in cat.colnames if f.rfind('LBDA_') != -1]
    keys = ['ID', 'RA', 'DEC'] + keyslbd
    return cat[keys]

# Reduction from SNR thresholding of Lines AFTER MATCHING
# def SNR_selection(ID,redcat_gt,snr_thresh=3):
#    keyssnr = [f for f in redcat_gt.colnames if f.rfind('_SNR') != -1]
#    ID_out = []
#    ID_trash = []
#    for n in range(len(ID)):
#        if ID[n][0] == 0:
#            ID_out.append(ID[n])
#        else:
#            source = redcat_gt[redcat_gt['ID']==ID[n][0]]
#            snr = [source[f].data.data[0] for f in keyssnr
#                   if not np.isnan(source[f].data.data[0])]
#            snr = [f for f in snr if f>snr_thresh]
#            if len(snr)>0:
#                ID_out.append(ID[n])
#            else:
#                ID_trash.append(ID[n])
#    print('')
#    print(' len ID input: '+str(len(ID))+' len ID output: '+str(len(ID_out)))
#    print('')
#    return ID_out , ID_trash

# Reduction from SNR thresholding of Lines before matching


def GroundTruth_selection(cat, snr_thresh=3, LineList=None):
    ''' Copy the ground truth catalog without Sources which have no lines
    greater than a defined snr.
    input:
        cat: ground truth catalog
        snr_thresh: snr threshold (default=3)
    output:
        cat: new catalof
        ID_trash: ID of the ground truth catalog which have been removed'''
    print('Remove from Ground Truth Catalog sources with Lines SNR <= '
          + str(snr_thresh))
    cat_out, ID_trash = SNR_selection(cat, snr_thresh)
    print('Remove from Ground Truth Catalog sources without the Lines :')
    print(LineList)
    print('')
    if LineList is not None:
        cat_out, ID_trash = Line_selection(cat_out, LineList, ID_trash)

    return cat_out, ID_trash


def Line_selection(cat, LineList, ID_trash):
    ID_trash
    indice = []
    for n, source in enumerate(cat):
        ID = source['ID']
        source = cat[cat['ID'] == ID]
        lin = [source[f].data.data[0] for f in LineList
               if not np.isnan(source[f].data.data[0])]
        if len(lin) > 0:
            indice.append(n)
        else:
            ID_trash.append(ID)

    return cat[indice], ID_trash


def SNR_selection(cat, snr_thresh):
    keyssnr = [f for f in cat.colnames if f.rfind('_SNR') != -1]
    ID_trash = []
    indice = []
    for n, source in enumerate(cat):

        ID = source['ID']
        source = cat[cat['ID'] == ID]

        snr = [source[f].data.data[0] for f in keyssnr
               if not np.isnan(source[f].data.data[0])]
        snr = [f for f in snr if f > snr_thresh]

        if len(snr) > 0:
            indice.append(n)
        else:
            ID_trash.append(ID)

    return cat[indice], ID_trash
#%%
#==============================================================================
# RELATED TO STD CUBE and to correl
#==============================================================================


def create_cubestd(path_cube):
    cub = Cube(path_cube)
    cube_raw = cub.data.filled(fill_value=0)  # Flux - set to 0 the Nan
    var = cub.var
    var[np.isnan(var)] = np.inf  # variance - set to Inf the Nan
    cube_std = cube_raw / np.sqrt(var)
    cube_std = cube_std.data
#    cube_std_a = np.array(cube_std)
    return cube_std


def create_correl_data(path_Correl):
    correl = Cube(path_Correl)
    W = correl.wave.get_range()
    sky2pix = correl.wcs.sky2pix
    freq = np.arange(W[0], W[1] + correl.wave.get_step(), correl.wave.get_step())
    correl = correl.data
#    correl = np.array(correl)
    return correl, freq, sky2pix

#%%
#==============================================================================
# Assign line from Origin to Ground Truth or not, extract line informations
#==============================================================================


def namelinegt(ID, cat, keywords, keywordssnr, snr_thresh=3):
    ''' extract from catalog the line name and frequencies'''
    source = cat[cat['ID'] == ID]

    lin = [source[f].data.data[0] for f in keywords
           if not np.isnan(source[f].data.data[0])]

    nam = [source[f].name for f in keywords
           if not np.isnan(source[f].data.data[0])]

    snr = [source[f].data.data[0] for f in keywordssnr
           if not np.isnan(source[f].data.data[0])]

    lin = [lin[f] for f in range(len(lin)) if snr[f] > snr_thresh]
    nam = [nam[f] for f in range(len(nam)) if snr[f] > snr_thresh]
    snr = [snr[f] for f in range(len(snr)) if snr[f] > snr_thresh]
    return lin, nam, snr

#%%


def nameline(ID, cat, keywords):
    ''' extract from catalog the line name and frequencies'''
    source = cat[cat['ID'] == ID]

    lin = [source[f].data.data[0] for f in keywords
           if not np.isnan(source[f].data.data[0])]

    nam = [source[f].name for f in keywords
           if not np.isnan(source[f].data.data[0])]

    return lin, nam

# def namelinegt(ID,cat,keywords):
#    ''' extract from catalog the line name and frequencies'''
#    source = cat[cat['ID']==ID]
#
#    lin = [source[f].data.data[0] for f in keywords
#           if not np.isnan(source[f].data.data[0])]
##    nam = [source[f].name[:-9] for f in keywords if not np.isnan(source[f].data.data[0])]
#    nam = [source[f].name for f in keywords
#           if not np.isnan(source[f].data.data[0])]
#    return lin, nam
#
# def namelineor(ID,cat,keywords):
#    source = cat[cat['ID']==ID]
#
#    lin = [source[f].data.data[0] for f in keywords
#           if not np.isnan(source[f].data.data[0])]
#    nam = [source[f].name[5:] for f in keywords
#           if not np.isnan(source[f].data.data[0])]
#    return lin, nam


def findclosername(lin_gt, nam_gt, lin_or, nam_or, gap=50):
    ltmp = np.array(lin_or)
    nam_out = nam_or.copy()
    for n, lin in enumerate(lin_gt):
        dist = np.abs(lin - ltmp)
        ii = np.argmin(dist)
        if dist[ii] < gap / 2:
            nam_out[ii] = nam_gt[n]
            ltmp[ii] = 0

    return nam_out

#    return Line_dico, Name_dico


def Match_Lines_Name(ID, redcat_gt, redcat_o1,
                     redcat_o2, freq, snr_thresh=3):

    key_gt = [f for f in redcat_gt.colnames if f.rfind('LBDA_OBS') != -1]
    key_o1 = [f for f in redcat_o1.colnames if f.rfind('LBDA_') != -1]
    key_o2 = [f for f in redcat_o2.colnames if f.rfind('LBDA_') != -1]
    keysnr_gt = [f[:-8] + 'SNR' for f in key_gt]

    SNR_dico = {}
    Name_dico = {}
    Name_dico['gt'] = {}
    Name_dico['o1'] = {}
    Name_dico['o2'] = {}
    Line_dico = {}
    Line_dico['gt'] = {}
    Line_dico['o1'] = {}
    Line_dico['o2'] = {}
    for n in ID:

        ID_gt, ID_o1, ID_o2 = n

        # check if GT
        if ID_gt:
            lin_gt, nam_gt, snr_gt = namelinegt(ID_gt, redcat_gt, key_gt,
                                                keysnr_gt, snr_thresh=snr_thresh)

            Name_dico['gt'][ID_gt] = {}
            Name_dico['gt'][ID_gt][0] = nam_gt
            Name_dico['gt'][ID_gt][1] = nam_gt
#            Name_dico['gt'][ID_gt] = nam_gt
            Line_dico['gt'][ID_gt] = {}
            Line_dico['gt'][ID_gt][0] = lin_gt
            Line_dico['gt'][ID_gt][1] = freq2ind(freq, lin_gt)
            SNR_dico[ID_gt] = snr_gt
        # check if ID_o1 match GT, if yes give good name to line
        if ID_o1:
            lin_o1, nam_o1 = nameline(ID_o1, redcat_o1, key_o1)
            # initialize dico
            Line_dico['o1'][ID_o1] = {}
            Line_dico['o1'][ID_o1][0] = lin_o1
            Line_dico['o1'][ID_o1][1] = freq2ind(freq, lin_o1)
            Name_dico['o1'][ID_o1] = {}
            Name_dico['o1'][ID_o1][0] = nam_o1
            if ID_gt:
                Name_dico['o1'][ID_o1][1] = findclosername(lin_gt, nam_gt,
                                                           lin_o1, nam_o1)
            else:
                Name_dico['o1'][ID_o1][1] = nam_o1

        # check if ID_o1 match GT, if yes give good name to line
        if ID_o2:
            lin_o2, nam_o2 = nameline(ID_o2, redcat_o2, key_o2)
            # initialize dico
            Line_dico['o2'][ID_o2] = {}
            Line_dico['o2'][ID_o2][0] = lin_o2
            Line_dico['o2'][ID_o2][1] = freq2ind(freq, lin_o2)
            Name_dico['o2'][ID_o2] = {}
            Name_dico['o2'][ID_o2][0] = nam_o2
            if ID_gt:
                Name_dico['o2'][ID_o2][1] = findclosername(lin_gt, nam_gt,
                                                           lin_o2, nam_o2)
            else:
                Name_dico['o2'][ID_o2][1] = nam_o2
    return Line_dico, Name_dico, SNR_dico

#==============================================================================
# Conversion Tools
#==============================================================================
#%% Convert all RA DEC to Pixels for sources in ID


def pixFromRaDec(cat, ID, sky2pix):
    source = cat[cat['ID'] == ID]
    ra = source['RA'].data.data[0]
    dec = source['DEC'].data.data[0]
    y, x = np.rint(sky2pix([dec, ra])[0])
    return int(y), int(x), ra, dec


def RaDec2Pix(ID, redcat_gt, redcat_o1, redcat_o2, sky2pix):
    ''' return all the y,x pixels position for Ground truth and from 
    catalogs for the two methods. y,x are computed from Ra Dec data'''
    position = {}
    position['gt'] = {}
    position['o1'] = {}
    position['o2'] = {}
    radec = {}
    radec['gt'] = {}
    radec['o1'] = {}
    radec['o2'] = {}

    list_gt = []
    list_o1 = []
    list_o2 = []
    for n in ID:
        ID_gt, ID_o1, ID_o2 = n

        if ID_gt:
            y, x, ra, dec = pixFromRaDec(redcat_gt, ID_gt, sky2pix)
            position['gt'][ID_gt] = y, x
            radec['gt'][ID_gt] = ra, dec
            list_gt.append((y, x, ID_gt))
        if ID_o1:
            y, x, ra, dec = pixFromRaDec(redcat_o1, ID_o1, sky2pix)
            position['o1'][ID_o1] = y, x
            radec['o1'][ID_gt] = ra, dec
            list_o1.append((y, x, ID_o1))
        if ID_o2:
            y, x, ra, dec = pixFromRaDec(redcat_o2, ID_o2, sky2pix)
            position['o2'][ID_o2] = y, x
            radec['o2'][ID_gt] = ra, dec
            list_o2.append((y, x, ID_o2))

    IDfromyx = {}
    IDfromyx['gt'] = np.array(list_gt)
    IDfromyx['o1'] = np.array(list_o1)
    IDfromyx['o2'] = np.array(list_o2)

    return position, IDfromyx, radec
#%% Convert Angstrom to sample


def freq2ind(freq, ref):
    ''' return the frequency index of ref in freq'''
    if type(ref) == np.float64 or type(ref) == np.int64:
        out = np.argmin((freq - ref)**2)
    else:
        out = [np.argmin((freq - f)**2) for f in ref]
    return out
#==============================================================================
# For Plot
#==============================================================================
#%% Counter of Matched Lines


def InitCounter(keyslbd):
    Counter = {}
    Counter['nm'] = 0
    for n in keyslbd:
        Counter[n] = 0
    return Counter


def CountName(ID, Name, keyslbd):

    # All Ground Truth Lines (GT Lines)
    Counter_total_gt = InitCounter(keyslbd)
    # # GT Lines in catalogs when o1 or o2 match
    Counter_match_gto1 = InitCounter(keyslbd)
    Counter_match_gto2 = InitCounter(keyslbd)
    # # GT Lines in catalogs when o1 or o2 don't match
    Counter_match_gtnm1 = InitCounter(keyslbd)
    Counter_match_gtnm2 = InitCounter(keyslbd)
    # Origin Lines in GT matching when peak distance is less than a band (50A)
    # for method 1 (o1) or 2 (o2) < band
    Counter_match_o1 = InitCounter(keyslbd)
    Counter_match_o2 = InitCounter(keyslbd)
    # A posteriori peak when there is matching for o1 or o2 with lines < band
    Counter_match_p1 = InitCounter(keyslbd)
    Counter_match_p2 = InitCounter(keyslbd)
    # A posteriori peak when there is no matching for o1 or o2
    # peak wich could have been detected because > threshold with lines < band
    Counter_match_g1 = InitCounter(keyslbd)
    Counter_match_g2 = InitCounter(keyslbd)

    Counter = {}
    for n in Name['gt']:  # count all the lines from gt
        for l in Name['gt'][n][1]:
            Counter_total_gt[l] += 1

    ID_fromgt = [f for f in ID if f[0] > 0]

    for TheIDs in ID_fromgt:
        ID_gt, ID_o1, ID_o2 = TheIDs
        if ID_o1:  # if o1 match gt count gt and o1 known lines and o1 no match
            for l in Name['gt'][ID_gt][1]:
                Counter_match_gto1[l] += 1
            for l in Name['o1'][ID_o1][1]:
                if l[:8] == 'LBDA_ORI':
                    Counter_match_o1['nm'] += 1
                else:
                    Counter_match_o1[l] += 1
            # count a posteriori found peaks
            for l in Name['p1'][ID_o1][1]:
                if l == 'Peak':
                    Counter_match_p1['nm'] += 1
                else:
                    Counter_match_p1[l] += 1

        if ID_o2:  # if o2 match gt count gt and o2 known lines and o2 no match
            for l in Name['gt'][ID_gt][1]:
                Counter_match_gto2[l] += 1
            for l in Name['o2'][ID_o2][1]:
                if l[:8] == 'LBDA_ORI':
                    Counter_match_o2['nm'] += 1
                else:
                    Counter_match_o2[l] += 1
            # count a posteriori found peaks
            for l in Name['p2'][ID_o2][1]:
                if l == 'Peak':
                    Counter_match_p2['nm'] += 1
                else:
                    Counter_match_p2[l] += 1

        if ID_o2 and not ID_o1:  # o1 don't match count the missed lines in gt
            for l in Name['gt'][ID_gt][1]:
                Counter_match_gtnm1[l] += 1
            # count a posteriori found peaks
            for l in Name['g1'][ID_gt][1]:
                if l == 'Peak':
                    Counter_match_g1['nm'] += 1
                else:
                    Counter_match_g1[l] += 1

        if ID_o1 and not ID_o2:  # o2 don't match count the missed lines in gt
            for l in Name['gt'][ID_gt][1]:
                Counter_match_gtnm2[l] += 1
            # count a posteriori found peaks
            for l in Name['g2'][ID_gt][1]:
                if l == 'Peak':
                    Counter_match_g2['nm'] += 1
                else:
                    Counter_match_g2[l] += 1

        # o1 and o2 don't match count the missed lines in gt
        if not ID_o2 and not ID_o1:
            for l in Name['gt'][ID_gt][1]:
                Counter_match_gtnm1[l] += 1
                Counter_match_gtnm2[l] += 1
            # count a posteriori found peaks
            for l in Name['g1'][ID_gt][1]:
                if l == 'Peak':
                    Counter_match_g1['nm'] += 1
                else:
                    Counter_match_g1[l] += 1
            # count a posteriori found peaks
            for l in Name['g2'][ID_gt][1]:
                if l == 'Peak':
                    Counter_match_g2['nm'] += 1
                else:
                    Counter_match_g2[l] += 1

    Counter['gt_all'] = Counter_total_gt
    Counter['gt_nm_o1'] = Counter_match_gto1
    Counter['gt_nm_o2'] = Counter_match_gto2
    Counter['gt_o1'] = Counter_match_o1
    Counter['gt_o2'] = Counter_match_o2
    Counter['gt_p1'] = Counter_match_p1
    Counter['gt_p2'] = Counter_match_p2
    Counter['gt_g1'] = Counter_match_g1
    Counter['gt_g2'] = Counter_match_g2
    Counter['gt_nm_g1'] = Counter_match_gtnm1
    Counter['gt_nm_g2'] = Counter_match_gtnm2
    return Counter
#%% create dico of matching info from ID


def dicomatch(ID):
    match = {}
    match['g1'] = {}
    match['g2'] = {}
    match['o1'] = {}
    match['o2'] = {}
    for n, TheIDs in enumerate(ID):
        ID_gt, ID_o1, ID_o2 = TheIDs

        if ID_gt and ID_o1:
            match['g1'][ID_gt] = 'm'
            match['o1'][ID_o1] = 'm'
        else:
            match['g1'][ID_gt] = 'nm'
            match['o1'][ID_o1] = 'nm'

        if ID_gt and ID_o2:
            match['g2'][ID_gt] = 'm'
            match['o2'][ID_o2] = 'm'
        else:
            match['g2'][ID_gt] = 'nm'
            match['o2'][ID_o2] = 'nm'

    return match
#%%
#==============================================================================
# Peak search in Methods 1 and 2 with Name
#==============================================================================


def searchPeaks(x, thresh=8):
    ''' in a correl "x" find peaks which are greater than "thresh"'''
    index, tmp = fpd.peakdet(x, 3)
    tmp = [int(f) for f in index[:, 0]]
    # peaks above thresh
    return [f for f in tmp if x[f] > thresh]


def redsearchpeaks(peak, freq, lin, band=50):
    ''' If origin found a line closer than "band" remove the peak '''
    out = []
    for n in peak:
        pic = freq[n]
        cc = np.argmin(np.abs(lin - pic))
        dif = np.abs(lin[cc] - pic)
        if dif > band / 2:
            out.append(n)

    return out


def LineNameSearch(ID, correl1, correl2, freq, Line, Name, position,\
                   Line_Band=50, Peak_Thresh=8):

    # define new entry in the dictionnary
    Name['p1'] = {}
    Name['p2'] = {}
    Name['g1'] = {}
    Name['g2'] = {}

    Line['p1'] = {}
    Line['p2'] = {}
    Line['g1'] = {}
    Line['g2'] = {}
    # from catalog 1
    for n, TheIDs in enumerate(ID):
        ID_gt, ID_o1, ID_o2 = TheIDs

        if ID_gt:
            lin_gt = Line['gt'][ID_gt][0]
            nam_gt = Name['gt'][ID_gt][1]
            y, x = position['gt'][ID_gt]
            if not ID_o1:
                spectre = correl1[:, y, x]
                peak_line = searchPeaks(spectre, thresh=Peak_Thresh)
                peak_freq = freq[peak_line]
                peak_name = ['Peak' for f in range(len(peak_line))]

                Line['g1'][ID_gt] = {}
                Line['g1'][ID_gt][0] = peak_freq
                Line['g1'][ID_gt][1] = peak_line
                Name['g1'][ID_gt] = {}
                Name['g1'][ID_gt][0] = peak_name
                Name['g1'][ID_gt][1] = peak_name

                if peak_name:
                    peak_name = findclosername(lin_gt, nam_gt,
                                               peak_freq, peak_name)
                    Name['g1'][ID_gt][1] = peak_name

            if not ID_o2:
                spectre = correl2[:, y, x]
                peak_line = searchPeaks(spectre, thresh=Peak_Thresh)
                peak_freq = freq[peak_line]
                peak_name = ['Peak' for f in range(len(peak_line))]

                Line['g2'][ID_gt] = {}
                Line['g2'][ID_gt][0] = peak_freq
                Line['g2'][ID_gt][1] = peak_line
                Name['g2'][ID_gt] = {}
                Name['g2'][ID_gt][0] = peak_name
                Name['g2'][ID_gt][1] = peak_name

                if peak_name:
                    peak_name = findclosername(lin_gt, nam_gt,
                                               peak_freq, peak_name)
                    Name['g2'][ID_gt][1] = peak_name

        if ID_o1:
            y, x = position['o1'][ID_o1]
            lin = Line['o1'][ID_o1][0]
            spectre = correl1[:, y, x]

            peak_line = searchPeaks(spectre, thresh=Peak_Thresh)
            peak_line = redsearchpeaks(peak_line, freq, lin, band=Line_Band)
            peak_freq = freq[peak_line]
            peak_name = ['Peak' for f in range(len(peak_line))]

            Line['p1'][ID_o1] = {}
            Line['p1'][ID_o1][0] = peak_freq
            Line['p1'][ID_o1][1] = peak_line
            Name['p1'][ID_o1] = {}
            Name['p1'][ID_o1][0] = peak_name
            Name['p1'][ID_o1][1] = peak_name

            if ID_gt and peak_name:
                peak_name = findclosername(lin_gt, nam_gt, peak_freq, peak_name)
                Name['p1'][ID_o1][1] = peak_name

        if ID_o2:
            y, x = position['o2'][ID_o2]
            lin = Line['o2'][ID_o2][0]
            spectre = correl2[:, y, x]

            peak_line = searchPeaks(spectre, thresh=Peak_Thresh)
            peak_line = redsearchpeaks(peak_line, freq, lin, band=Line_Band)
            peak_freq = freq[peak_line]
            peak_name = ['Peak' for f in range(len(peak_line))]

            Line['p2'][ID_o2] = {}
            Line['p2'][ID_o2][0] = peak_freq
            Line['p2'][ID_o2][1] = peak_line
            Name['p2'][ID_o2] = {}
            Name['p2'][ID_o2][0] = peak_name
            Name['p2'][ID_o2][1] = peak_name

            if ID_gt and peak_name:
                peak_name = findclosername(lin_gt, nam_gt, peak_freq, peak_name)
                Name['p2'][ID_o2][1] = peak_name

    return Name, Line
