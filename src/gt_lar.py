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
        print nodeB
        ln = len(self.V)
        self.V.append(nodeB)
        self.V.append((np.array(nodeB)+[2, 0, 0]).tolist())
        self.V.append((np.array(nodeB)+[2, 2, 0]).tolist())
        self.V.append((np.array(nodeB)+[2, 2, 2]).tolist())
        self.V.append((np.array(nodeA)+[0, 0, 0]).tolist())
        self.CV.append([ln, ln+1, ln+2, ln+3, ln+4])


    def show(self):
        self.__circle([30,30,30], [0,2,1], 10)
        V = self.V
        CV = self.CV


        # V = [[0,0,0],[5,5,1],[0,5,5],[5,5,5]]
        # CV = [[0,1,2,3]]
        # print 'V, CV ', V, CV
        VIEW(MKPOL([V,AA(AA(lambda k:k+1))(CV),[]]))
# characteristic matrices

    def get_output(self):
        pass

    def __add_tetr(self, nodeB):
        ln = len(self.V)
        self.V.append(nodeB)
        self.V.append((np.array(nodeB)+[2, 0, 0]).tolist())
        self.V.append((np.array(nodeB)+[2, 2, 0]).tolist())
        self.V.append((np.array(nodeB)+[2, 2, 2]).tolist())
        self.CV.append([ln, ln + 1, ln + 2, ln + 3])

    def __circle(self, center, perp_vect, radius):
        """
        perp_vect is vector perpendicular to plane of circle
        """
        tl = [0, 0.2,0.4,0.6,0.8, 1.0]

        # vector form center to edge of circle
        # u is a unit vector from the centre of the circle to any point on the circumference

        u = self.__perpendicular_vector(perp_vect)
        u = u / np.linalg.norm(u)

        n = perp_vect

        for t in tl:
            # u = np.array([0, 1, 0])
            # n = np.array([1, 0, 0])
            pt = radius * np.cos(t * 2 * np.pi) * u +\
                radius * np.sin(t * 2 * np.pi) * np.cross(u, n) +\
                center

            print 'pt ', pt

            self.__add_tetr(pt.tolist())

    def __perpendicular_vector(self, v):
        r""" Finds an arbitrary perpendicular vector to *v*."""
        if v[1] == 0 and v[2] == 0:
            if v[0] == 0:
                raise ValueError('zero vector')
            else:
                return np.cross(v, [0, 1, 0])
        return np.cross(v, [1, 0, 0])


