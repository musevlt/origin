# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 10:15:00 2016

@author: antonyschutz
"""
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
#%%
#==============================================================================
# PLOT
#==============================================================================
# plot basic statistics


def plotIDstat(ID, ID_thrash, save=False, path='', show=True):

    IDtmp = np.array(ID.copy())
    IDtmp[IDtmp > 0] = 1
    N_gt, N_o1, N_o2 = IDtmp.sum(axis=0)

    Ntotal = N_gt + len(ID_thrash)

    tmp = (IDtmp[:, 0] + IDtmp[:, 1]) == 2
    M_gt_o1 = tmp.sum(axis=0)

    tmp = (IDtmp[:, 0] + IDtmp[:, 2]) == 2
    M_gt_o2 = tmp.sum(axis=0)

    noM_o1 = N_gt - M_gt_o1
    noM_o2 = N_gt - M_gt_o2
    noM_gt_o1 = N_o1 - M_gt_o1
    noM_gt_o2 = N_o2 - M_gt_o2

    xTickMarks = ['', 'Match GT', 'Missed sources', 'additional DET']

    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()

    # necessary variables
    ind = np.arange(4)                # the x locations for the groups
    width = 0.4                      # the width of the bars

    x = N_gt
    rects0 = ax.bar(2 * width, x, width, color='blue')
    score = '     ' + str(x) + ' / ' + str(Ntotal) + '\n(>SNR / 100%)'
    plt.text(width / 3 + 2 * width, x + 10, score,
             fontsize=10, color='black', rotation=0)

    x = M_gt_o1
    rects1 = ax.bar(1 + width, x, width, color='yellow')
    score = str(x) + ' / ' + str(N_gt) + '\n(' + str(round(x / N_gt * 100, 2)) + '%)'
    plt.text(1 + width / 3 + width, N_gt + 10, score,
             fontsize=10, color='black', rotation=0)

    x = M_gt_o2
    rects3 = ax.bar(1 + width + width, x, width, color='green')
    score = str(x) + ' / ' + str(N_gt) + '\n(' + str(round(x / N_gt * 100, 2)) + '%)'
    plt.text(1 + width / 3 + 2 * width, N_gt + 10, score,
             fontsize=10, color='black', rotation=0)

    x = noM_o1
    rects1 = ax.bar(2 + width, x, width, color='yellow')
    score = str(x) + ' / ' + str(N_gt) + '\n(' + str(round(x / N_gt * 100, 2)) + '%)'
    plt.text(2 + width / 3 + width, N_gt + 10, score,
             fontsize=10, color='black', rotation=0)

    x = noM_o2
    rects3 = ax.bar(2 + width + width, x, width, color='green')
    score = str(x) + ' / ' + str(N_gt) + '\n(' + str(round(x / N_gt * 100, 2)) + '%)'
    plt.text(2 + width / 3 + 2 * width, N_gt + 10, score,
             fontsize=10, color='black', rotation=0)

    x = noM_gt_o1
    rects1 = ax.bar(3 + width, x, width, color='yellow')
    score = str(x)
    plt.text(3 + width / 3 + width, N_gt + 10, score,
             fontsize=10, color='black', rotation=0)

    x = noM_gt_o2
    rects3 = ax.bar(3 + width + width, x, width, color='green')
    score = str(x)
    plt.text(3 + width / 3 + 2 * width, N_gt + 10, score,
             fontsize=10, color='black', rotation=0)

    # axes and labels
    ax.set_xlim(0, len(ind) + width)
#    ax.set_ylim(0,MAX+15)
    ax.set_ylabel('Scores')

    ax.set_xticks(ind + width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=20, fontsize=12)

    plt.title('Matching Informations')
    plt.suptitle('Catalog size (Ground Truth) len: ' + str(N_gt) +
                 '\nCAT1 (Initial Method) len: ' + str(N_o1) +
                 '  ||  CAT2 (Modificated Method) len: ' + str(N_o2))

    plt.legend([rects0, rects1, rects3],
               ['DET GT', 'DET Method 1', 'DET Method2'], loc='upper left')

    if save:
        fig.savefig(os.path.join(path, '00_matchInfo.pdf'), format="pdf")
    if not show:
        plt.close()
#%%


def plotbarnm(GTRes0, GTRes, PKRes, Ningt, finpk, xTickMarks, titre, MAX, snr_thresh=False):

    N = len(xTickMarks)
    ax = plt.gca()

    # necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars

    # the bars
    rects0 = ax.bar(ind + width, GTRes0, width, color='white')
    rectsa = ax.bar(ind + width, GTRes, width, color='blue')

    rects2 = ax.bar(ind, PKRes, width, color='black')

    for n in ind:
        score = str(PKRes[n]) + '/' + str(GTRes[n]) + '/' + str(GTRes0[n])
        plt.text(n, GTRes0[n] + 5, score, fontsize=9, color='black', rotation=0)

    # axes and labels
    MMax = MAX + 15
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylim(0, MMax)
    ax.set_ylabel('Scores')

    ax.set_yticks(np.arange(0, MMax, 20))
    ax.set_xticks(ind + width * 1.5)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=8)

    # add a legend
    if snr_thresh == False:
        t1 = '1) DET in ground truth'
    else:
        t1 = '1) DET in ground truth (S/N>' + str(snr_thresh) + ')'
    t2 = '3) Missed DET'
    t3 = '4) Missed DET with right peak in correl'
    ax.legend((rects0[0], rectsa[0], rects2[0]), (t1, t2, t3))
    titre = ' (' + '{0:d}'.format(np.array(GTRes).sum()) + '/'
    titre += '{0:d}'.format(np.array(GTRes0).sum()) + '='
    titre += '{0:.2f}'.format(100 * np.array(GTRes).sum() / np.array(GTRes0).sum())
    titre += '% ) || '
    titre += 'Additional line traces (in Peak search only): '
    titre += '{0:d}'.format(np.array(finpk).sum())
    plt.title(titre, fontsize=12)

    ax.yaxis.grid(which='major')  # horizontal lines
    plt.show()

# Bar Plot of population of Found compared to GT and no match
# (GT which are not found )


def bothplotbarGTnm(Ct, snr_thresh=False, save=False, path='', show=True):

    titre1 = 'Method 1'
    titre2 = 'Method 2'

    keywords = sorted(Ct['gt_all'].keys())[:-1]
    GTRes = [Ct['gt_all'][f] for f in keywords]

    Ningt1 = Ct['gt_all']['nm']
    finpk1 = Ct['gt_g1']['nm']
    PKRes1 = [Ct['gt_g1'][f] for f in keywords]
    GTRes1 = [Ct['gt_nm_g1'][f] for f in keywords]
    xTickMarks1 = [f[:-9] for f in sorted(Ct['gt_all'].keys()) if f != 'nm']

    MAX1 = np.maximum(np.array(PKRes1).max(), np.array(GTRes).max())

    Ningt2 = Ct['gt_all']['nm']
    finpk2 = Ct['gt_g2']['nm']
    PKRes2 = [Ct['gt_g2'][f] for f in keywords]
    GTRes2 = [Ct['gt_nm_g2'][f] for f in keywords]

    xTickMarks2 = xTickMarks1

    MAX2 = np.maximum(np.array(PKRes2).max(), np.array(GTRes).max())

    MAX = np.maximum(MAX1, MAX2)

    fig = plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plotbarnm(GTRes, GTRes1, PKRes1, Ningt1, finpk1, xTickMarks1, titre1,
              MAX, snr_thresh=snr_thresh)
    plt.subplot(212)
    plotbarnm(GTRes, GTRes2, PKRes2, Ningt2, finpk2, xTickMarks2, titre2,
              MAX, snr_thresh=snr_thresh)

    titresup = 'No match Score for GT:'
    titresup += '\n Method name (% of unmatched Ground truth lines).'
    titresup += '\n Sum of significant peak in correl '
    titresup += 'which are not Origin or ground truth lines'
    plt.suptitle(titresup)

    if save:
        fig.savefig(os.path.join(path, '00_score_no_match.pdf'), format="pdf")
    if not show:
        plt.close()


#%%
def plotbar(GTRes0, DTRes, GTRes, PKRes, Ningt, finpk, findt, xTickMarks,
            titre, MAX, snr_thresh=False):

    N = len(xTickMarks)
    ax = plt.gca()

    # necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.25                      # the width of the bars

    # the bars
    pkdt = [DTRes[f] + PKRes[f] for f in range(N)]
    rects0 = ax.bar(ind + width + width, GTRes0, width, color='white')
    rectsa = ax.bar(ind + width + width, GTRes, width, color='blue')
    rects3 = ax.bar(ind + width, pkdt, width, color='yellow')
    rects1 = ax.bar(ind + width, DTRes, width, color='red')
    rects2 = ax.bar(ind, PKRes, width, color='black')

    for n in ind:
        score = str(PKRes[n]) + '/' + str(DTRes[n]) + '/' + str(GTRes[n])
        score += '/' + str(GTRes0[n])
        plt.text(n, GTRes0[n] + 5, score, fontsize=9, color='black', rotation=0)

    # axes and labels
    MMax = MAX + 15
    ax.set_xlim(-width, len(ind) + width)
    ax.set_ylim(0, MMax)
    ax.set_ylabel('Scores')

    ax.set_yticks(np.arange(0, MMax, 20))
    ax.set_xticks(ind + width * 1.5)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=8)

    # add a legend
    if snr_thresh == False:
        t1 = '1) DET in ground truth'
        ta = '2) DET in match Origin/ground truth'
    else:
        t1 = '1) DET in ground truth (S/N>' + str(snr_thresh) + ')'
        ta = '2) DET in match Origin/ground truth (S/N>' + str(snr_thresh) + ')'
    t2 = '3) Total DET by Origin'
    t3 = '4) Missed DET with right peak in correl'
    t4 = '3) + 4)'

    ax.legend((rects0[0], rectsa[0], rects1[0], rects2[0], rects3[0]),
              (t1, ta, t2, t3, t4))

    titre += ' (' + '{0:d}'.format(np.array(DTRes).sum()) + '/'
    titre += '{0:d}'.format(np.array(GTRes0).sum()) + '='
    titre += '{0:.2f}'.format(100 * np.array(DTRes).sum() / np.array(GTRes0).sum())
    titre += '% ) || '

    tmp = (np.array(GTRes0).sum() - np.array(DTRes).sum())
    titre += 'Missed DET (=in GT only): ' + '{0:d}'.format(tmp) + '/'
    titre += '{0:d}'.format(np.array(GTRes0).sum()) + '='
    titre += '{0:.2f}'.format(100 * tmp / np.array(GTRes0).sum()) + ' || '
    titre += 'FA/New DET (=in Origin only): '
    titre += '{0:d}'.format(np.array(findt).sum()) + ' || '
    titre += 'Additional line traces (in Peak search only): '
    titre += '{0:d}'.format(np.array(finpk).sum())
    plt.title(titre, fontsize=12)

    ax.yaxis.grid(which='major')  # horizontal lines
    plt.show()

# Bar Plot of population of Found compared to GT and no match
# (GT which are not found )


def bothplotbarGT(Ct, snr_thresh=False, save=False, path='', show=True):

    titre1 = 'Method 1'
    titre2 = 'Method 2'
    keywords = sorted(Ct['gt_all'].keys())[:-1]
    GTRes = [Ct['gt_all'][f] for f in keywords]

    Ningt1 = Ct['gt_all']['nm']
    findt1 = Ct['gt_o1']['nm']
    finpk1 = Ct['gt_p1']['nm']
    DTRes1 = [Ct['gt_o1'][f] for f in keywords]
    PKRes1 = [Ct['gt_p1'][f] for f in keywords]
    GTRes1 = [Ct['gt_nm_o1'][f] for f in keywords]
    xTickMarks1 = [f[:-9] for f in sorted(Ct['gt_all'].keys()) if f != 'nm']

    MAX1 = np.maximum(np.array(DTRes1).max(), np.array(GTRes).max())

    Ningt2 = Ct['gt_all']['nm']
    findt2 = Ct['gt_o2']['nm']
    finpk2 = Ct['gt_p2']['nm']
    DTRes2 = [Ct['gt_o2'][f] for f in keywords]
    PKRes2 = [Ct['gt_p2'][f] for f in keywords]
    GTRes2 = [Ct['gt_nm_o2'][f] for f in keywords]
    xTickMarks2 = xTickMarks1

    MAX2 = np.maximum(np.array(DTRes2).max(), np.array(GTRes).max())

    MAX = np.maximum(MAX1, MAX2)

    fig = plt.figure(figsize=(20, 10))
    plt.subplot(211)
    plotbar(GTRes, DTRes1, GTRes1, PKRes1, Ningt1, finpk1, findt1, xTickMarks1, titre1,
            MAX, snr_thresh=snr_thresh)
    plt.subplot(212)
    plotbar(GTRes, DTRes2, GTRes2, PKRes2, Ningt2, finpk2, findt2, xTickMarks2, titre2,
            MAX, snr_thresh=snr_thresh)

    titresup = 'Score for two methods: '
    titresup += '\n Method name (sum of sources found by Origin assigned to a '
    titresup += 'ground truth line / sum of detected lines in ground truth).'
    titresup += '\nSum of ground truth lines missed by Origin. '
    titresup += 'Sum of lines found by origin which are not in ground truth. '
    titresup += 'Sum of significant peak in correl which are not Origin or '
    titresup += 'ground truth lines'
    plt.suptitle(titresup)

    if save:
        fig.savefig(os.path.join(path, '00_score_match.pdf'), format="pdf")
    if not show:
        plt.close()


#%%
# Draw Sources correlation and lines informations
def find_ym_yM(ID, correl1, correl2, position, freq):
    ''' check all spectra to plot to return min and max value and at same time
    freq range.'''
    # find min and max for whole fig
    ym = 0
    yM = 0
    for n, TheIDs in enumerate(ID):
        ID_gt, ID_o1, ID_o2 = TheIDs
        if ID_gt:  # plot Ground Truth information
            y_gt, x_gt = position['gt'][ID_gt]
            spectre1_gt = correl1[:, y_gt, x_gt]
            spectre2_gt = correl2[:, y_gt, x_gt]
            ym = np.minimum(spectre1_gt.min(), spectre2_gt.min())
            yM = np.maximum(spectre1_gt.max(), spectre2_gt.max())

        if ID_o1:  # plot Ground Truth information
            y_o1, x_o1 = position['o1'][ID_o1]
            spectre1_o1 = correl1[:, y_o1, x_o1]
            ym = np.minimum(ym, spectre1_o1.min())
            yM = np.maximum(yM, spectre1_o1.max())

        if ID_o2:  # plot Ground Truth information
            y_o2, x_o2 = position['o2'][ID_o2]
            spectre1_o2 = correl1[:, y_o2, x_o2]
            ym = np.minimum(ym, spectre1_o2.min())
            yM = np.maximum(yM, spectre1_o2.max())
    return (ym, yM), (freq[0], freq[-1])
# for one source:


def draw_1_ID_Corr(TheIDs, correl1, correl2, freq, Line, Name, position, ym_yM_xm_xM,
                   SNR, DIST, radec, Line_Band, Peak_Thresh, snr_thresh):

    ID_gt, ID_o1, ID_o2 = TheIDs
    handles = []
    legends = []

    y_gt, x_gt, y_o1, x_o1, y_o2, x_o2 = ('-', '-', '-', '-', '-', '-')
    ra_gt, dec_gt, ra_o1, dec_o1, ra_o2, dec_o2 = ('-', '-', '-', '-', '-', '-')
    dist_o1, dist_o2 = DIST

    ym_yM, xm_xM = ym_yM_xm_xM
    ym, yM = ym_yM
    xm, xM = xm_xM

    cor_fig = plt.figure(figsize=(20, 10))

    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)

    thresh_h, = ax1.plot([xm, xM], [Peak_Thresh, Peak_Thresh],
                         linestyle='-', color='gray', alpha=.1, lw=2)
    legend_h = 'threshold (>' + str(Peak_Thresh) + ')'
    handles.append(thresh_h)
    legends.append(legend_h)

    ax2.plot([xm, xM], [Peak_Thresh, Peak_Thresh], linestyle='-',
             color='gray', alpha=.1, lw=2)
    if ID_gt:  # plot Ground Truth information
        y_gt, x_gt = position['gt'][ID_gt]
        ra_gt, dec_gt = radec['gt'][ID_gt]
        spectre1_gt = correl1[:, y_gt, x_gt]
        spectre2_gt = correl2[:, y_gt, x_gt]

    if ID_o1:  # plot Ground Truth information
        y_o1, x_o1 = position['o1'][ID_o1]
        ra_o1, dec_o1 = radec['o1'][ID_gt]
        spectre1_o1 = correl1[:, y_o1, x_o1]

    if ID_o2:  # plot Ground Truth information
        y_o2, x_o2 = position['o2'][ID_o2]
        ra_o2, dec_o2 = radec['o2'][ID_gt]
        spectre1_o2 = correl1[:, y_o2, x_o2]

    isbk = 0
    isbl = 0
    if ID_gt:  # plot Ground Truth information
        lin = Line['gt'][ID_gt][0]
        spl = Line['gt'][ID_gt][1]
        nam = Name['gt'][ID_gt][1]
        snr = SNR[ID_gt]
        # plot correlation and line at Ground Truth Position
        gt_h, = ax1.plot(freq, spectre1_gt, 'b', lw=2, alpha=.75)
        ax2.plot(freq, spectre2_gt, 'b', lw=2, alpha=.75)

        # plot line of ground truth
        sort_index = np.argsort(lin)
        m = .5
        for ind in sort_index:
            ax = ax1
            thename = nam[ind][0:-9] + ' (' + '{0:.2f}'.format(snr[ind]) + ')'
            ax.annotate(thename,
                        xy=(lin[ind], spectre1_gt[spl[ind]]),
                        xytext=(xm + ((xM - xm) / (sort_index.max() + 2)) * m, .9 * yM),
                        fontsize=8, color='blue',
                        arrowprops=dict(arrowstyle='->', facecolor='blue',
                                        alpha=.25, color='blue'))
            li_gt_h, = ax.plot([lin[ind], lin[ind]], [ym, yM],
                               color='blue', lw=4, alpha=.15)
            bd_gt_h, = ax.plot([lin[ind] - Line_Band / 2, lin[ind] - Line_Band / 2],
                               [ym, yM], linestyle='--', color='blue', lw=1, alpha=.25)
            ax.plot([lin[ind] + Line_Band / 2, lin[ind] + Line_Band / 2],
                    [ym, yM], linestyle='--', color='blue', lw=1, alpha=.25)
            ax = ax2
            ax.annotate(thename,
                        xy=(lin[ind], spectre1_gt[spl[ind]]),
                        xytext=(xm + ((xM - xm) / (sort_index.max() + 2)) * m, .9 * yM),
                        fontsize=8, color='blue',
                        arrowprops=dict(arrowstyle='->', facecolor='blue',
                                        alpha=.25, color='blue'))

            m += 1

            ax.plot([lin[ind], lin[ind]], [ym, yM], color='blue', lw=4, alpha=.15)
            ax.plot([lin[ind] - Line_Band / 2, lin[ind] - Line_Band / 2], [ym, yM],
                    linestyle='--', color='blue', lw=1, alpha=.25)
            ax.plot([lin[ind] + Line_Band / 2, lin[ind] + Line_Band / 2], [ym, yM],
                    linestyle='--', color='blue', lw=1, alpha=.25)

        legend_h = 'band DET in GT (' + str(Line_Band) + ' Angstroms)'
        handles.append(bd_gt_h)
        legends.append(legend_h)

        legend_h = 'Correlation (GT position)'
        handles.append(gt_h)
        legends.append(legend_h)
        legend_h = 'DET in GT (SNR>' + str(snr_thresh) + ')'
        handles.append(li_gt_h)
        legends.append(legend_h)

        if not ID_o1:
            peak_freq = Line['g1'][ID_gt][0]
            peak_smpl = Line['g1'][ID_gt][1]
            peak_name = Name['g1'][ID_gt][1]
            # plot peak found in method 1 at ground truth
            sort_index = np.argsort(peak_freq)
            ax = ax1
            m = .5
            for ind in sort_index:
                if peak_name[ind] == 'Peak':
                    peak_name2 = peak_name[ind]
                    isbk_h, = ax.plot(peak_freq[ind],
                                      spectre1_gt[peak_smpl[ind]],
                                      color='black', marker='o', linestyle='')
                    isbk = 1
                else:
                    peak_name2 = peak_name[ind][:-9]
                    isbl_h, = ax.plot(peak_freq[ind],
                                      spectre1_gt[peak_smpl[ind]],
                                      color='blue', marker='o', linestyle='')
                    isbl = 1
                ax.annotate(peak_name2,
                            xy=(peak_freq[ind], spectre1_gt[peak_smpl[ind]]),
                            xytext=(xm + ((xM - xm) / (sort_index.max() + 2)) * m, .7 * yM),
                            fontsize=8, color='black',
                            arrowprops=dict(arrowstyle='->', facecolor='black',
                                            alpha=.25, color='black'))

                m += 1

        if not ID_o2:
            peak_freq = Line['g2'][ID_gt][0]
            peak_smpl = Line['g2'][ID_gt][1]
            peak_name = Name['g2'][ID_gt][1]
            # plot peak found in method 1 at ground truth
            sort_index = np.argsort(peak_freq)
            ax = ax2
            m = .5
            for ind in sort_index:
                if peak_name[ind] == 'Peak':
                    peak_name2 = peak_name[ind]
                    isbk_h, = ax.plot(peak_freq[ind],
                                      spectre2_gt[peak_smpl[ind]],
                                      color='black', marker='o', linestyle='')
                    isbk = 1
                else:
                    peak_name2 = peak_name[ind][:-9]
                    isbl_h, = ax.plot(peak_freq[ind],
                                      spectre2_gt[peak_smpl[ind]],
                                      color='blue', marker='o', linestyle='')
                    isbl = 1

                ax.annotate(peak_name2,
                            xy=(peak_freq[ind], spectre2_gt[peak_smpl[ind]]),
                            xytext=(xm + ((xM - xm) / (sort_index.max() + 2)) * m, .7 * yM),
                            fontsize=8, color='black',
                            arrowprops=dict(arrowstyle='->', facecolor='black',
                                            alpha=.25, color='black'))
                m += 1

    if ID_o1:  # plot Method1 information
        idm = ID_o1
        num = '1'
        spectre = spectre1_o1
        ax = ax1

        lin = Line['o' + num][idm][0]
        spl = Line['o' + num][idm][1]
        nam = Name['o' + num][idm][1]
        peak_freq = Line['p' + num][idm][0]
        peak_smpl = Line['p' + num][idm][1]
        peak_name = Name['p' + num][idm][1]
        # plot correlation and line at Ground Truth Position
        o1_h, = ax.plot(freq, spectre, 'r', lw=2, alpha=.75)
        legend_h = 'Correlation (Origin Position)'
        handles.append(o1_h)
        legends.append(legend_h)
        # plot line of ground truth
        sort_index = np.argsort(lin)
        m = .5
        fact = (xM - xm) / (sort_index.max() + 2)
        for ind in sort_index:
            nam2 = nam[ind][9:] if nam[ind][0:8] == 'LBDA_ORI' else nam[ind][:-9]
            ax.annotate(nam2, xy=(lin[ind], spectre[spl[ind]]),
                        xytext=(xm + fact * m, .8 * yM), fontsize=8, color='red',
                        arrowprops=dict(arrowstyle='->', facecolor='red',
                                        alpha=.25, color='red'))
            m += 1
            li_o1_h, = ax.plot([lin[ind], lin[ind]], [ym, yM],
                               color='red', lw=2, alpha=.25)
#            ax.plot([lin[ind]-band/2,lin[ind]-band/2],[ym,yM],linestyle=':',color='red',lw=1 ,alpha=.5)
#            ax.plot([lin[ind]+band/2,lin[ind]+band/2],[ym,yM],linestyle=':',color='red',lw=1 ,alpha=.5)
        legend_h = 'DET in Origin'
        handles.append(li_o1_h)
        legends.append(legend_h)
        isb = 0
        isr = 0
        if peak_name:
            sort_index = np.argsort(peak_freq)
            m = .5
            fact = (xM - xm) / (sort_index.max() + 2)
            for ind in sort_index:
                if peak_name[ind] == 'Peak':
                    peak_name2 = peak_name[ind]
                    peakcolor = 'black'
                    pk_o1_hb, = ax.plot(peak_freq[ind], spectre[peak_smpl[ind]],
                                        linestyle='', color=peakcolor, marker='o')
                    isb = 1
                else:
                    peak_name2 = peak_name[ind][:-9]
                    peakcolor = 'red'
                    pk_o1_hr, = ax.plot(peak_freq[ind], spectre[peak_smpl[ind]],
                                        linestyle='', color=peakcolor, marker='o')
                    isr = 1
                ax.annotate(peak_name2,
                            xy=(peak_freq[ind], spectre[peak_smpl[ind]]),
                            xytext=(xm + fact * m, .7 * yM), fontsize=8, color='black',
                            arrowprops=dict(arrowstyle='->', facecolor='black',
                                            alpha=.25, color='black'))

                m += 1
            if isr == 1:
                legend_h = 'Missed DET in Origin'
                handles.append(pk_o1_hr)
                legends.append(legend_h)
            if isb == 1 and isbk == 0:
                legend_h = 'FA/New DET'
                handles.append(pk_o1_hb)
                legends.append(legend_h)

    if ID_o2:  # plot Method2 information
        idm = ID_o2
        num = '2'
        spectre = spectre1_o2
        ax = ax2

        lin = Line['o' + num][idm][0]
        spl = Line['o' + num][idm][1]
        nam = Name['o' + num][idm][1]
        peak_freq = Line['p' + num][idm][0]
        peak_smpl = Line['p' + num][idm][1]
        peak_name = Name['p' + num][idm][1]
        # plot correlation and line at Ground Truth Position
        o1_h, = ax.plot(freq, spectre, 'r', lw=2, alpha=.75)
        if not ID_o1:
            legend_h = 'Correlation (Origin Position)'
            handles.append(o1_h)
            legends.append(legend_h)
        # plot line of ground truth
        sort_index = np.argsort(lin)
        m = .5
        fact = (xM - xm) / (sort_index.max() + 2)
        for ind in sort_index:
            nam2 = nam[ind][9:] if nam[ind][0:8] == 'LBDA_ORI' else nam[ind][:-9]
            ax.annotate(nam2, xy=(lin[ind], spectre[spl[ind]]),
                        xytext=(xm + fact * m, .8 * yM), fontsize=8, color='red',
                        arrowprops=dict(arrowstyle='->', facecolor='red',
                                        alpha=.25, color='red'))
            m += 1
            li_o1_h, = ax.plot([lin[ind], lin[ind]], [ym, yM],
                               color='red', lw=2, alpha=.25)
#            ax.plot([lin[ind]-band/2,lin[ind]-band/2],[ym,yM],linestyle=':',color='red',lw=1 ,alpha=.5)
#            ax.plot([lin[ind]+band/2,lin[ind]+band/2],[ym,yM],linestyle=':',color='red',lw=1 ,alpha=.5)
        if not ID_o1:
            legend_h = 'DET in Origin'
            handles.append(li_o1_h)
            legends.append(legend_h)
        isb = 0
        isr = 0
        if peak_name:
            sort_index = np.argsort(peak_freq)
            m = .5
            fact = (xM - xm) / (sort_index.max() + 2)
            for ind in sort_index:
                if peak_name[ind] == 'Peak':
                    peak_name2 = peak_name[ind]
                    peakcolor = 'black'
                    pk_o1_hb, = ax.plot(peak_freq[ind], spectre[peak_smpl[ind]],
                                        linestyle='', color=peakcolor, marker='o')
                    isb = 1
                else:
                    peak_name2 = peak_name[ind][:-9]
                    peakcolor = 'red'
                    pk_o1_hr, = ax.plot(peak_freq[ind], spectre[peak_smpl[ind]],
                                        linestyle='', color=peakcolor, marker='o')
                    isr = 1
                ax.annotate(peak_name2,
                            xy=(peak_freq[ind], spectre[peak_smpl[ind]]),
                            xytext=(xm + fact * m, .7 * yM), fontsize=8, color='black',
                            arrowprops=dict(arrowstyle='->', facecolor='black',
                                            alpha=.25, color='black'))

                m += 1
            if not ID_o1 and isr == 1:
                legend_h = 'Missed DET in Origin'
                handles.append(pk_o1_hr)
                legends.append(legend_h)
            if not ID_o1 and isb == 1 and isbk == 0:
                legend_h = 'FA/New DET'
                handles.append(pk_o1_hb)
                legends.append(legend_h)
    if isbl:
        legend_h = 'right peak in correl'
        handles.append(isbl_h)
        legends.append(legend_h)

    if isbk == 1:
        legend_h = 'FA/New DET'
        handles.append(isbk_h)
        legends.append(legend_h)

    ax1.set_ylim(ym_yM)
    ax1.set_xlim(xm_xM)
    ax2.set_ylim(ym_yM)
    ax2.set_xlim(xm_xM)

    pst = ax1.get_position()

    plt.legend(handles, legends, bbox_to_anchor=(pst.x0, 0., pst.x1 - pst.x0, 1),
               loc=3, bbox_transform=plt.gcf().transFigure,
               ncol=4, mode="expand", borderaxespad=0.)

    titresup = 'ID Ground Truth: ' + str(ID_gt)
    if not y_gt == '-':
        titresup += ' - Position: '
        titresup += '( y = ' + str(y_gt)
        titresup += ', x = ' + str(x_gt) + ', '
        titresup += ' ra = ' + '{0:.5f}'.format(ra_gt)
        titresup += ', dec = ' + '{0:.5f}'.format(dec_gt) + ' )'
    titresup += '\nsnr_thresh: ' + str(snr_thresh)
    titresup += ' - '
    titresup += 'Peak_Thresh: ' + str(Peak_Thresh)
    cor_fig.suptitle(titresup, fontsize=12)

    titreax1 = 'ID method 1: ' + str(ID_o1)
    if not y_o1 == '-':
        titreax1 += ', position: ( y = ' + str(y_o1)
        titreax1 += ', x = ' + str(x_o1) + ','
        titreax1 += ' ra = ' + '{0:.5f}'.format(ra_o1)
        titreax1 += ', dec = ' + '{0:.5f}'.format(dec_o1) + ' )'
    if not dist_o1 == '-':
        titreax1 += ', distance = ' + '{0:.3f}'.format(dist_o1) + ' as'
    ax1.set_title(titreax1)

    titreax2 = 'ID method 2: ' + str(ID_o2)
    if not y_o2 == '-':
        titreax2 += ', position: ( y=' + str(y_o2)
        titreax2 += ', x=' + str(x_o2) + ','
        titreax2 += ' ra=' + '{0:.5f}'.format(ra_o2)
        titreax2 += ', dec=' + '{0:.5f}'.format(dec_o2) + ' )'
    if not dist_o2 == '-':
        titreax2 += ', distance = ' + '{0:.3f}'.format(dist_o2) + ' as'
    ax2.set_title(titreax2)

    return cor_fig
#%% Draw all matching and not matching sources


def MatchNoMatch(img1, img2, ID, position, sat_tresh, colormap, name=''):

    figmatch = plt.figure(figsize=(20, 10))
    ax0 = plt.subplot(1, 1, 1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.imshow(img1, cmap=colormap, vmin=img1.min(),
               vmax=img1.min() + (img1.max() - img1.min()) * sat_tresh,
               interpolation='nearest')
    ax2.imshow(img2, cmap=colormap, vmin=img2.min(),
               vmax=img2.min() + (img2.max() - img1.min()) * sat_tresh,
               interpolation='nearest')
    xm = ax1.get_xlim()
    ym = ax1.get_ylim()
    colorgt = 'b'
    coloror = 'r'

    for n, TheIDs in enumerate(ID):
        ID_gt, ID_o1, ID_o2 = TheIDs
        if ID_gt:
            y_gt, x_gt = position['gt'][ID_gt]
            if ID_o1:
                markergt = 'o'
                ogt, = ax1.plot(x_gt, y_gt, marker=markergt, color=colorgt, ls='')
            else:
                markergt = 'x'
                xgt, = ax1.plot(x_gt, y_gt, marker=markergt, color=colorgt, ls='')
            if ID_o2:
                markergt = 'o'
                ogt, = ax2.plot(x_gt, y_gt, marker=markergt, color=colorgt, ls='')
            else:
                markergt = 'x'
                xgt, = ax2.plot(x_gt, y_gt, marker=markergt, color=colorgt, ls='')

        if ID_o1:
            y_o1, x_o1 = position['o1'][ID_o1]
            if ID_gt:
                markeror = '.'
                oor, = ax1.plot(x_o1, y_o1, marker=markeror, color=coloror, ls='')
            else:
                markeror = 'x'
                xor, = ax1.plot(x_o1, y_o1, marker=markeror, color=coloror, ls='')

        if ID_o2:
            y_o2, x_o2 = position['o2'][ID_o2]
            if ID_gt:
                markeror = '.'
                oor, = ax2.plot(x_o2, y_o2, marker=markeror, color=coloror, ls='')
            else:
                markeror = 'x'
                xor, = ax2.plot(x_o2, y_o2, marker=markeror, color=coloror, ls='')

        for ax in (ax1, ax2):
            ax.set_xlim(xm)
            ax.set_ylim(ym)
            ax.set_xticklabels(())
            ax.set_yticklabels(())
    handles = (ogt, xgt, oor, xor)
    legends = []
    legends.append('Ground Truth sources with match')
    legends.append('Ground Truth sources without match')
    legends.append('Origin sources with match')
    legends.append('Origin sources without match')
    pst = ax0.get_position()
    plt.legend(handles, legends, bbox_to_anchor=(pst.x0, 0., pst.x1 - pst.x0, 1),
               loc=3, bbox_transform=plt.gcf().transFigure,
               ncol=4, mode="expand", borderaxespad=0.)

    ax1.set_title('Method 1')
    ax2.set_title('Method 2')
    suptitre = 'Match and unmatch sources from catalogs for two methods'
    if name:
        suptitre += '\n drawn on : ' + name

    suptitre += '\n saturated at : ' + str(sat_tresh)
    figmatch.suptitle(suptitre, fontsize=14)

    return figmatch


def matchimg(ID, position, max_map, wcorrel1, wcorrel2, wcube_std,
             sat_tresh_list=(1, 1), save=False, path='', show=True,
             colormap='gray_r'):

    im1l = (max_map, wcorrel1)
    im2l = (max_map, wcorrel2)
    naml = ('White Std Cube', 'Max Correlation')
    for n in range(2):
        figmatch = MatchNoMatch(im1l[n], im2l[n], ID, position,
                                sat_tresh_list[n], colormap, name=naml[n])

        filename = 'match_' + naml[n].replace(" ", "")
        if save:
            figmatch.savefig(os.path.join(path, filename + '.pdf'), format="pdf")
        if not show:
            plt.close()


def reducedsearch(red, y1, y2, x1, x2, cuID=False):
    out = False
    yred = np.where(red[:, 0] > y1)[0]
    if len(yred) > 0:
        red = red[yred, :]
        yred = np.where(red[:, 0] < y2)[0]
        if len(yred) > 0:
            red = red[yred, :]
            xred = np.where(red[:, 1] > x1)[0]
            if len(xred) > 0:
                red = red[xred, :]
                xred = np.where(red[:, 1] < x2)[0]
                if len(xred) > 0:
                    red = red[xred, :]
                    if cuID:
                        idred = np.where(red[:, 2] != cuID)[0]
                        red = red[idred, :]
                    if len(red) > 0:
                        out = red[:, 2]

    return out


def redname(nam):
    if nam[:5] == 'LBDA_':
        return nam[5:]
    else:
        return nam[:-9]


def drawfromcand(num, ax, cand, dl, cu_position, position, Name, Line, match,
                 size_inpix, colorsrc, method, sy, h, t):

    gtor = 'Origin' if colorsrc == 'red' else 'Ground Truth'
    meth = '1' if method == 1 else '2'
    t1 = gtor + ' source with match (Method ' + meth + ')'
    t2 = gtor + ' source without match (Method ' + meth + ')'
    pst = ax.get_position()
    pp = np.zeros(3)
    pp[1] = pst.x0
    pp[-1] = pst.x1 - pst.x0
    markersrc = ('s', 'x') if method == 1 else ('d', '+')
    for n, c in enumerate(cand):
        lin = []
        nam = []
        for ii, f in enumerate(Line[c][1]):
            if f < dl[1] and f > dl[0]:
                lin.append(f)
                nam.append(Name[c][1][ii])
        if len(lin) > 0:
            mrk = markersrc[0] if match[c] == 'm' else markersrc[1]
            y, x = position[c]
            y = y - cu_position[0] + size_inpix / 2
            x = x - cu_position[1] + size_inpix / 2
            ht = ax.scatter(x, y, marker=mrk, facecolor=colorsrc,
                            edgecolor='white', s=120, alpha=.8)
            tt = t1 if match[c] == 'm' else t2
            h.append(ht)
            t.append(tt)
            for ii, li in enumerate(lin):
                rp = (1 - np.maximum(0., (-1)**num), 0.5)
                linname = redname(nam[ii])
                info = '(M' + str(method) + ' - ID: ' + str(c) + ') ' + linname

                ax.annotate(info, xy=(x, y), textcoords='figure fraction',
                            xytext=(pp[(-1)**num], pst.y1 * (1 - num / 20)),
                            fontsize=14,
                            color=colorsrc, arrowprops=dict(arrowstyle='->',
                                                            facecolor=colorsrc, alpha=1, color=colorsrc, relpos=rp))
                num += 1

    return num, h, t


def from1line(gtor, cu_ID, cu_position, layer, cu_line, cu_name, candgt,
              cando1, cando2, position, Line, Name, match, band_inpix,
              size_inpix, cube_std_a, max_map, correl1, correl2, hgt,
              colormap):

    titlesupi = ', ID: ' + str(cu_ID)
    titlesupm = 'Ground Truth' if gtor == 'gt' else 'Origin with Method 1' \
        if gtor == 'o1' else 'Origin with Method 2'

    titlesup = 'Image Centered on '
    titlesup += titlesupm
    titlesup += ' source '
    titlesup += titlesupi
    titlesup += ' '
    titlesup += '(y,x) interval: ' + str(layer)

    h = []
    t = []
    Df = 1.25
    df = 25
    s = .15

    nl = cube_std_a.shape[0]
    y1, y2, x1, x2 = layer
    colsrc = 'blue' if gtor == 'gt' else 'red'

    zoomfig = plt.figure(figsize=(20, 10))
    ax1 = plt.subplot(1, 1, 1)
    img = max_map
    ax1.imshow(img, interpolation='nearest', cmap=colormap)
    ax1.plot([x1, x1], [y1, y2], 'white')
    ax1.plot([x2, x2], [y1, y2], 'white')
    ax1.plot([x1, x2], [y1, y1], 'white')
    ax1.plot([x1, x2], [y2, y2], 'white')
    ax1.set_xlim((0, img.shape[1] - 1))
    ax1.set_ylim((0, img.shape[0] - 1))
    ax1.set_title('White image - zone', color=colsrc)
    hgt.append(zoomfig)

    maxmapfig = plt.figure(figsize=(20, 10))
    maxmapfig.suptitle(titlesup, fontsize=18)
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    img = max_map[y1:y2, x1:x2]
    ax1.imshow(img, interpolation='nearest', cmap=colormap)
    ax1.set_xlim((0, img.shape[1] - 1))
    ax1.set_ylim((0, img.shape[0] - 1))
    ax1.set_title('Max Map', color=colsrc)

    img = np.max(correl1[:, y1:y2, x1:x2], axis=0)
    ax2.imshow(img, interpolation='nearest', cmap=colormap)
    ax2.set_xlim((0, img.shape[1] - 1))
    ax2.set_ylim((0, img.shape[0] - 1))
    ax2.set_title('Correlation max, Method 1', color=colsrc)

    img = np.max(correl2[:, y1:y2, x1:x2], axis=0)
    ax3.imshow(img, interpolation='nearest', cmap=colormap)
    ax3.set_xlim((0, img.shape[1] - 1))
    ax3.set_ylim((0, img.shape[0] - 1))
    ax3.set_title('Correlation max, Method 2', color=colsrc)

    hgt.append(maxmapfig)
    sy = img.shape[0] - 1
    for n, lin in enumerate(cu_line[1]):

        imagette = plt.figure(figsize=(20, 10))
        ax1 = plt.subplot(1, 1, 1)

        l1 = int(np.maximum(0, lin - band_inpix / 2))
        l2 = int(np.minimum(nl, lin + band_inpix / 2))

        img = np.sum(cube_std_a[l1:l2, y1:y2, x1:x2], axis=0)
        cax1 = ax1.imshow(img, cmap=colormap, interpolation='nearest')
#        ax1.imshow( img ,cmap=colormap, interpolation='nearest')
        titlelin = redname(cu_name[n]) + ' ( ' + str(cu_line[0][n]) + \
            r' $\mathring{A}$ ' + '+/-' + str(band_inpix / 2 * Df) + ' ) '
        ax1.set_title(titlelin, color=colsrc, fontsize=16)
        cbaxes = imagette.add_axes([s, 0.1, 1 - 2 * s, 0.05])
        plt.colorbar(cax1, cax=cbaxes, orientation='horizontal')

        # figure suptitle
        imgsupt = titlesupm + ' Source (data white)' + titlesupi
        imagette.suptitle(imgsupt, fontsize=18)

        num = 0
        if candgt is not False:
            xsrc = 'gt'
            colorsrc = 'blue'
            method = 1
            num, h, t = drawfromcand(num, ax1, candgt, (l1, l2), cu_position,
                                     position[xsrc], Name[xsrc], Line[xsrc],
                                     match['g1'], size_inpix, colorsrc, method,
                                     sy, h, t)

            xsrc = 'gt'
            colorsrc = 'blue'
            method = 2
            num, h, t = drawfromcand(num, ax1, candgt, (l1, l2), cu_position,
                                     position[xsrc], Name[xsrc], Line[xsrc],
                                     match['g2'], size_inpix, colorsrc, method,
                                     sy, h, t)

        if cando1 is not False:
            if len(cando1) > 0:
                xsrc = 'o1'
                colorsrc = 'red'
                method = 1
                num, h, t = drawfromcand(num, ax1, cando1, (l1, l2), cu_position,
                                         position[xsrc], Name[xsrc], Line[xsrc],
                                         match[xsrc], size_inpix, colorsrc, method,
                                         sy, h, t)
        if cando2 is not False:
            if len(cando2) > 0:
                xsrc = 'o2'
                colorsrc = 'red'
                method = 2
                num, h, t = drawfromcand(num, ax1, cando2, (l1, l2), cu_position,
                                         position[xsrc], Name[xsrc], Line[xsrc],
                                         match[xsrc], size_inpix, colorsrc, method,
                                         sy, h, t)

        ax1.set_xlim((0, img.shape[1] - 1))
        ax1.set_ylim((0, sy))
        pst = ax1.get_position()
        ax1.set_position((.3, pst.y0, .4, pst.y1))
        ax1.set_yticks(())
        ax1.set_xticks(())
        if t:
            h, t = rmht(h, t)
            ax1.legend(h, t, bbox_to_anchor=(0.05, 0., .9, 1),
                       loc=3, bbox_transform=plt.gcf().transFigure, fontsize=12,
                       ncol=4, mode="expand", borderaxespad=0.)

        # second axis
        rect = -.015, (.35 - s) / 2, s, s
        ax2 = imagette.add_axes(rect)
        l1 = int(np.maximum(0, lin - df - band_inpix / 2))
        l2 = int(np.minimum(nl, lin - df + band_inpix / 2))

        if l2 > l1:
            img = np.sum(cube_std_a[l1:l2, y1:y2, x1:x2], axis=0)
            cax2 = ax2.imshow(img, cmap=colormap, interpolation='nearest')
            titlelin = str(cu_line[0][n] - df * Df) + r' $\mathring{A}$'
            ax2.set_title(titlelin, color=colsrc, fontsize=8)
            ax2.set_xticklabels(())
            ax2.set_yticklabels(())
            ax2.set_xlim((0, img.shape[1] - 1))
            ax2.set_ylim((0, img.shape[0] - 1))
            plt.colorbar(cax2)

        # third axis
        rect = .985 - s, (.35 - s) / 2, s, s
        ax3 = imagette.add_axes(rect)
        l1 = int(np.maximum(0, lin + df - band_inpix / 2))
        l2 = int(np.minimum(nl, lin + df + band_inpix / 2))

        if l2 > l1:
            img = np.sum(cube_std_a[l1:l2, y1:y2, x1:x2], axis=0)
            cax3 = ax3.imshow(img, cmap=colormap, interpolation='nearest')
            titlelin = str(cu_line[0][n] + df * Df) + r' $\mathring{A}$'
            ax3.set_title(titlelin, color=colsrc, fontsize=8)
            ax3.set_xticklabels(())
            ax3.set_yticklabels(())
            ax3.set_xlim((0, img.shape[1] - 1))
            ax3.set_ylim((0, img.shape[0] - 1))
            plt.colorbar(cax3)

        hgt.append(imagette)

    return hgt


#%%

def rmht(h, t):
    List = [(h[f], t[f]) for f in range(len(h))]
    n = 0
    out_h = []
    out_t = []
    while True:
        out_h.append(List[n][0])
        out_t.append(List[n][1])
        cand = [List[f] for f in range(len(List)) if List[f][1] == List[n][1]]
        [List.remove(f) for f in cand]
        if len(List) == 0:
            break
    return out_h, out_t


def y12x12(y, x, ny, nx, size_inpix):
    y1 = int(np.maximum(y - size_inpix / 2, 0))
    y2 = int(np.minimum(y + size_inpix / 2, ny))
    x1 = int(np.maximum(x - size_inpix / 2, 0))
    x2 = int(np.minimum(x + size_inpix / 2, nx))
    return y1, y2, x1, x2


def candfor1line(IDfromyx, y1, y2, x1, x2):
    candgt = reducedsearch(IDfromyx['gt'].copy(), y1, y2, x1, x2)
    cando1 = reducedsearch(IDfromyx['o1'].copy(), y1, y2, x1, x2)
    cando2 = reducedsearch(IDfromyx['o2'].copy(), y1, y2, x1, x2)
    return candgt, cando1, cando2


def speccontent(TheIDs, Line, Name, position, IDfromyx, match, cube_std_a,
                correl1, correl2, max_map, band_inpix, size_inpix, colormap):

    ID_gt, ID_o1, ID_o2 = TheIDs
    nl, ny, nx = cube_std_a.shape

    hgt = []

    if ID_gt > 0:  # draw around the position of ID_gt
        idsrc = ID_gt
        src = 'gt'
        y, x = position[src][idsrc]
        y1, y2, x1, x2 = y12x12(y, x, ny, nx, size_inpix)
        # search source in y1:y2 and x1:x2
        candgt, cando1, cando2 = candfor1line(IDfromyx, y1, y2, x1, x2)
        # search in lambda1:lambda2 and draw the content
        hgt = from1line(src, idsrc, (y, x), (y1, y2, x1, x2), Line[src][idsrc],
                        Name[src][idsrc][1], candgt, cando1, cando2, position, Line,
                        Name, match, band_inpix, size_inpix, cube_std_a, max_map,
                        correl1, correl2, hgt, colormap)

    if ID_o1 > 0:  # draw around the position of ID_o1
        idsrc = ID_o1
        src = 'o1'
        y, x = position[src][idsrc]
        y1, y2, x1, x2 = y12x12(y, x, ny, nx, size_inpix)
        # search source in y1:y2 and x1:x2
        candgt, cando1, cando2 = candfor1line(IDfromyx, y1, y2, x1, x2)
        # search in lambda1:lambda2 and draw the content
        hgt = from1line(src, idsrc, (y, x), (y1, y2, x1, x2), Line[src][idsrc],
                        Name[src][idsrc][1], candgt, cando1, cando2, position, Line,
                        Name, match, band_inpix, size_inpix, cube_std_a, max_map,
                        correl1, correl2, hgt, colormap)

    if ID_o2 > 0:  # draw around the position of ID_o2
        idsrc = ID_o2
        src = 'o2'
        y, x = position[src][idsrc]
        y1, y2, x1, x2 = y12x12(y, x, ny, nx, size_inpix)
        # search source in y1:y2 and x1:x2
        candgt, cando1, cando2 = candfor1line(IDfromyx, y1, y2, x1, x2)
        # search in lambda1:lambda2 and draw the content
        hgt = from1line(src, idsrc, (y, x), (y1, y2, x1, x2), Line[src][idsrc],
                        Name[src][idsrc][1], candgt, cando1, cando2, position, Line,
                        Name, match, band_inpix, size_inpix, cube_std_a, max_map,
                        correl1, correl2, hgt, colormap)

    return hgt

#%%


def drawsources(ID, correl1, correl2, freq, Line, Name, position,
                SNR, DIST, IDfromyx, match, cube_std_a, radec, max_map,
                Line_Band=50, Peak_Thresh=8, snr_thresh=3, size_inpix=32,
                save=False, path='', show=True, colormap='gray_r'):
    ''' Draw all sources and save them in one pdf file per sources'''

    df = freq[1] - freq[0]
    band_inpix = int(Line_Band / df)
    ym_yM = find_ym_yM(ID, correl1, correl2, position, freq)
    for n, TheIDs in enumerate(ID):

        cor_fig = draw_1_ID_Corr(TheIDs, correl1, correl2, freq, Line,
                                 Name, position, ym_yM, SNR, DIST[n], radec,
                                 Line_Band, Peak_Thresh, snr_thresh)

        img_fig = speccontent(TheIDs, Line, Name, position, IDfromyx, match,
                              cube_std_a, correl1, correl2, max_map,
                              band_inpix, size_inpix, colormap)

        if save:
            # Create the PdfPages object to which we will save the pages:
            pdfname = os.path.join(path, 'ID_' + str(n))
            pdfname += '_gt_' + str(TheIDs[0])
            pdfname += '_o1_' + str(TheIDs[1])
            pdfname += '_o2_' + str(TheIDs[2])

            keywordsforpdf = '_Line_Band_' + str(Line_Band)
            keywordsforpdf += '_Peak_Thresh_' + str(Peak_Thresh)
            keywordsforpdf += '_snr_thresh_' + str(snr_thresh)

            with PdfPages(pdfname + '.pdf') as pdf:
                pdf.savefig(cor_fig)  # saves the current figure into a pdf page
                plt.close(cor_fig)
                for handlefig in img_fig:
                    pdf.savefig(handlefig)  # saves the current figure into a pdf page
                    if not show:
                        plt.close(handlefig)

                    # We can also set the file's metadata via the PdfPages object:
                    d = pdf.infodict()
                    d['Title'] = 'Multipage PDF Example'
                    d['Author'] = u'Antony Schutz'
                    d['Subject'] = 'Origin results Analysis'
                    d['Keywords'] = keywordsforpdf
                    d['CreationDate'] = datetime.datetime(2016, 12, 26)
                    d['ModDate'] = datetime.datetime.today()
        if not show:
            plt.close()
