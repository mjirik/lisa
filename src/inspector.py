#
# -*- coding: utf-8 -*-
"""
================================================================================
Name:        inspector
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky (volkovinsky.pavel@gmail.com)

Created:     08.11.2012
Copyright:   (c) Pavel Volkovinsky 2012
================================================================================
"""

import sys
sys.path.append("../src/")
sys.path.append("../extern/")

import logging
logger = logging.getLogger(__name__)

import numpy

import scipy.ndimage
import sys
"""
import scipy.misc
import scipy.io

import unittest
import argparse
"""

import matplotlib.pyplot as matpyplot
import matplotlib
from matplotlib.widgets import Slider, Button#, RadioButtons

"""
================================================================================
inspector
================================================================================
"""
class inspector:
    
    def __init__(self, data):
        
        self.data = data
    
    def showPlot(self):
        
        #matpyplot.show()
        
        return self.data
