#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Generator of histology report

"""
import logging
logger = logging.getLogger(__name__)


# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../../lar-cc/lib/py/"))
# import sys; sys.path.insert(0, 'lib/py/')
from larcc import *
from largrid import *
from scipy.spatial import Delaunay

class GTLar:
    def __init__(self, gtree=None):
        """
        gtree is information about input data structure. Not used here.
        """
# input of geometry and topology
        self.V = []
        self.CV = []
        pass

    def add_cylinder(self, nodeA, nodeB, radius):
        ln = len(self.V)
        self.V.append(nodeB)
        self.V.append((np.array(nodeB)+[2, 0, 0]).tolist())
        self.V.append((np.array(nodeB)+[2, 2, 0]).tolist())
        self.V.append((np.array(nodeB)+[2, 2, 2]).tolist())
        self.V.append((np.array(nodeA)+[0, 0, 0]).tolist())
        self.CV.append([ln, ln+1, ln+2, ln+3, ln+4])


    def show(self):
        V = self.V
        CV = self.CV


        # V = [[0,0,0],[5,5,1],[0,5,5],[5,5,5]]
        # CV = [[0,1,2,3]]
        # print 'V, CV ', V, CV
        VIEW(MKPOL([V,AA(AA(lambda k:k+1))(CV),[]]))
# characteristic matrices

    def get_output(self):
        pass
