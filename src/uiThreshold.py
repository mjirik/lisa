# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 19:09:40 2012

@author: Pavel Volkovinsky
"""

import sys
sys.path.append("../src/")
sys.path.append("../extern/")

#=========
# 0
#=========

"""
"""

#=========
# 1
#=========

import pylab as pylab

# Vyzve uzivatele k zadani jmena souboru.
info = raw_input("Give me an order: ")
if(info == 'x'):
    fileName = 'morpho.png'
else:
    fileName = info

imgA = pylab.imread(fileName)
#generates a RGB image, so do
#pylab.imshow(imgA)
imgB = pylab.mean(imgA, 2) # to get a 2-D array
#pylab.imshow(imgB)
pylab.gray()

import matplotlib.pyplot as matpyplot
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons

fig = matpyplot.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.25)
min0 = 0
max0 = 20000

#im = max0 * np.random.random((10,10))
im1 = ax.imshow(imgB) # tady byl 'im' (o radek vyse)
fig.colorbar(im1)

axcolor = 'lightgoldenrodyellow'
axmin = fig.add_axes([0.15, 0.1, 0.65, 0.03], axisbg=axcolor)
axmax  = fig.add_axes([0.15, 0.15, 0.65, 0.03], axisbg=axcolor)

smin = Slider(axmin, 'Min', 0, 20000, valinit=min0)
smax = Slider(axmax, 'Max', 0, 20000, valinit=max0)

def update(val):
    pass
    im1.set_clim([smin.val,smax.val])
    fig.canvas.draw()
smin.on_changed(update)
smax.on_changed(update)

matpyplot.show()

"""
"""

#=========
# 2
#=========

"""

from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons

ax = subplot(111)
subplots_adjust(left=0.25, bottom=0.25)
t = arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0*sin(2*pi*f0*t)
l, = plot(t,s, lw=2, color='red')
axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axamp  = axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*sin(2*pi*freq*t))
    draw()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    draw()
radio.on_clicked(colorfunc)

show()

"""