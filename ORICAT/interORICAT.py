#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 09:34:46 2017

@author: antonyschutz
"""

# ==============================================================================
# INTERACTIVE MAPPING
# ==============================================================================
from matplotlib.widgets import Slider, RadioButtons, Button, RectangleSelector
from pylab import *


def testentry(dico, keyword):
    test = len([f for f in dico.keys() if f == keyword]) > 0
    if not test:
        dico[keyword] = {}
    return dico
# spectral dictionnary: for each frequency sample save the ID of source ``
# which have a peak


def SpecDict(Line, position):
    Spectre_tmp = {}
    gtor_list = ('gt', 'o1', 'o2')
    for gtor in gtor_list:
        Spectre_tmp[gtor] = {}
        for ID in Line[gtor].keys():  # for each sources in gt
            lin = Line[gtor][ID][1]  # take frequency in sample
            for l in lin:
                Spectre_tmp[gtor] = testentry(Spectre_tmp[gtor], l)
                Spectre_tmp[gtor][l][ID] = position[gtor][ID]
    return Spectre_tmp


def reduceSpecDict(SpectreInfo, l1, l2):
    ''' DeltaLambda is already between 0 and Max frequency'''
    gtor_list = ('gt', 'o1', 'o2')
    SpectreIndex = {}
    for gtor in gtor_list:
        SpectreIndex[gtor] = False
        index = [f for f in SpectreInfo[gtor].keys()]
        index = np.array(index)
        ind = np.where(index >= l1)
        if len(ind) > 0:
            index = index[ind]
            ind = np.where(index <= l2)
            if len(ind) > 0:
                index = index[ind]
                SpectreIndex[gtor] = index

    return SpectreIndex


def DrawPoints(ax, nx, SpectreInfo, SpectreIndex, X, Y, match):
    x1, x2 = X
    y1, y2 = Y

    gtor_list = ('gt', 'o1', 'o2')
    colr = ('blue', 'red', 'red')
    mrkr = ('o', '+', 'x')

    numr = 0
    numl = 0

    pst = ax.get_position()
    pp = np.zeros(3)
    pp[1] = pst.x0 / 2
    pp[0] = pst.x1 + pst.x0 / 4

    for n, gtor in enumerate(gtor_list):
        for ind in SpectreIndex[gtor]:
            ID = [f for f in SpectreInfo[gtor][ind].keys()]
            for indid in ID:
                y, x = SpectreInfo[gtor][ind][indid]
                if y > y1 and y < y2 and x > x1 and x < x2:
                    y -= y1
                    x -= x1

                    ax.scatter(x, y, marker=mrkr[n], facecolor=colr[n],
                               edgecolor=colr[n])

                    if x < nx / 2:
                        rpn = 1
                        num = numl
                        numl += 1
                    else:
                        rpn = 0
                        num = numr
                        numr += 1

                    if gtor == 'gt':
                        info = '(' + gtor_list[n] + ' - ID: ' + str(indid)\
                            + 'm1: ' + match['g1'][indid] + ' ,m2: '\
                            + match['g2'][indid] + ')'
                    else:
                        info = '(' + gtor_list[n] + ' - ID: ' + str(indid)\
                            + ' ' + match[gtor][indid] + ')'

                    ax.annotate(info, xy=(x, y), textcoords='figure fraction',
                                xytext=(pp[rpn], pst.y1 * (1 - num / 20)),
                                fontsize=14,
                                color=colr[n], arrowprops=dict(arrowstyle='->',
                                                               facecolor=colr[n], alpha=1, color=colr[n], relpos=(rpn, .5)))


def DrawSlice(ax, cuby, SpectreInfo, match, lambda_central, Delta_lambda, thresh, freq, X, Y):
    nl, ny, nx = cuby.shape
    x1, x2 = X
    y1, y2 = Y
    l1 = int(np.maximum(0, lambda_central - Delta_lambda / 2))
    l2 = int(np.minimum(nl, lambda_central + Delta_lambda / 2))
    y1 = int(np.maximum(0, y1))
    y2 = int(np.minimum(ny, y2))
    x1 = int(np.maximum(0, x1))
    x2 = int(np.minimum(nx, x2))
    img = np.sum(cuby[l1:l2, y1:y2, x1:x2], axis=0)
    ny, nx = img.shape
    img = img - img.min()
    img = img / img.max()
    SpectreIndex = reduceSpecDict(SpectreInfo, l1, l2)
    titre = ''
    titre += r'$\lambda_0$: ' + str(freq[lambda_central])
    titre += ' (+/-) ' + r' $\Delta \lambda$: ' + str(.5 * Delta_lambda * (freq[1] - freq[0]))

    ax.cla()
    ax.imshow(img, cmap="gray_r", vmin=0, vmax=thresh)
    DrawPoints(ax, nx, SpectreInfo, SpectreIndex, X, Y, match)
    ax.set_xlim((0, nx - 1))
    ax.set_ylim((0, ny - 1))

    ax.set_title(titre)


plt.close('all')

SpectreInfo = SpecDict(Line, position)

fig = plt.figure(figsize=(20, 10))
axis_color = 'lightgoldenrodyellow'
# Draw the plot
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.25)

t = np.arange(0.0, 1.0, 0.001)
band_0 = 50
freq_0 = 25
thresh_0 = 100
nl, ny, nx = cube_std_a.shape
X = (0, nx - 1)
Y = (0, ny - 1)


band_slider_ax = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axis_color)
band_slider = Slider(band_slider_ax, 'sample band px', 1, 200, valinit=band_0)
freq_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axis_color)
freq_slider = Slider(freq_slider_ax, 'sample Freq px', 1, cube_std_a.shape[0], valinit=freq_0)

thresh_slider_ax = fig.add_axes([0.25, 0.05, 0.45, 0.03], axisbg=axis_color)
thresh_slider = Slider(thresh_slider_ax, 'Thresh %', 1, 100, valinit=thresh_0)


br_ax = fig.add_axes([0.025, 0.5, 0.05, 0.15], axisbg=axis_color)
data_radios = RadioButtons(br_ax, ('cube', 'c1', 'c2'), active=0)

br_ax2 = fig.add_axes([0.025, 0.25, 0.05, 0.15], axisbg=axis_color)
data_radios2 = RadioButtons(br_ax2, ('5%', '10%', '15%', '25%', '50%', '75%', '100%'), active=6)

x1_ax = fig.add_axes([0.05, 0.05, 0., 0.], axisbg=axis_color)
x1_slider = Slider(x1_ax, 'x1 :   ', 0, nx - 1, valinit=0)
y1_ax = fig.add_axes([0.05, 0.1, 0., 0.], axisbg=axis_color)
y1_slider = Slider(y1_ax, 'y1 :   ', 0, ny - 1, valinit=0)
x2_ax = fig.add_axes([0.05, 0.15, 0., 0.], axisbg=axis_color)
x2_slider = Slider(x2_ax, 'x2 :   ', 0, nx - 1, valinit=nx - 1)
y2_ax = fig.add_axes([0.05, 0.20, 0., 0.], axisbg=axis_color)
y2_slider = Slider(y2_ax, 'y2 :   ', 0, ny - 1, valinit=ny - 1)

x1 = x1_slider.val
y1 = y1_slider.val
x2 = x2_slider.val
y2 = y2_slider.val

DrawSlice(ax, cube_std_a, SpectreInfo, match, freq_0, band_0, 1, freq, (x1, x2), (y1, y2))


def drawall():
    hzdict = {'cube': cube_std_a, 'c1': correl1, 'c2': correl2}
    img = hzdict[data_radios.value_selected]
    thresh = thresh_slider.val / 100
    x1 = x1_slider.val
    y1 = y1_slider.val
    x2 = x2_slider.val
    y2 = y2_slider.val
#    print(x1,x2,y1,y2)
    DrawSlice(ax, img, SpectreInfo, match, freq_slider.val, band_slider.val, thresh, freq, (x1, x2), (y1, y2))
    plt.show()


def sliders_on_changed(val):
    drawall()


band_slider.on_changed(sliders_on_changed)
freq_slider.on_changed(sliders_on_changed)
thresh_slider.on_changed(sliders_on_changed)
data_radios.on_clicked(sliders_on_changed)


def clickthresh(mouse_event):
    htdict = {'5%': 5, '10%': 10, '15%': 15, '25%': 25, '50%': 50, '75%': 75, '100%': 100}
    thresh = htdict[data_radios2.value_selected]
    thresh_slider.set_val(thresh)
    sliders_on_changed()


data_radios2.on_clicked(clickthresh)


def clickcube(mouse_event):
    drawall()


data_radios.on_clicked(clickcube)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')


def reset_button_on_clicked(mouse_event):
    freq_slider.reset()
    band_slider.reset()
    thresh_slider.reset()
    x1_slider.reset()
    x2_slider.reset()
    y1_slider.reset()
    y2_slider.reset()


reset_button.on_clicked(reset_button_on_clicked)


def onselect(eclick, erelease):
    'eclick and erelease are matplotlib events at press and release'
#    print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
#    print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
    x1 = int(np.minimum(eclick.xdata, erelease.xdata))
    x2 = int(np.maximum(eclick.xdata, erelease.xdata))
    y1 = int(np.minimum(eclick.ydata, erelease.ydata))
    y2 = int(np.maximum(eclick.ydata, erelease.ydata))
    x1_slider.set_val(x1)
    x2_slider.set_val(x2)
    y1_slider.set_val(y1)
    y2_slider.set_val(y2)


def toggle_selector(event):
    print(' Key pressed.')


#zoom_button_ax = fig.add_axes([0.025, 0.8, 0.05, 0.04])
#zoom_button = Button(zoom_button_ax, 'reset ZOOM', color=axis_color, hovercolor='0.975')
# def zoom_button_on_clicked(mouse_event):
#    x1_slider.reset()
#    x2_slider.reset()
#    y1_slider.reset()
#    y2_slider.reset()
# zoom_button.on_clicked(zoom_button_on_clicked)


toggle_selector.RS = RectangleSelector(ax, onselect, drawtype='box')
toggle_selector.RS.set_active(True)
connect('key_press_event', toggle_selector)

#connect('key_press_event', toggle_selector)
# show()
