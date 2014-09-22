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
import numpy as np

from larcc import VIEW, MKPOL, AA, STRUCT, MKPOLS, PI
import mapper
# from largrid import *


class GTLar:

    def __init__(self, gtree=None):
        """
        gtree is information about input data structure.
        endDistMultiplicator: make cylinder shorter by multiplication of radius
        """
# input of geometry and topology
        self.V = []
        self.CV = []
        self.balls = {}
        self.gtree = gtree
        self.endDistMultiplicator = 1
        pass

    def add_cylinder(self, nodeA, nodeB, radius, cylinder_id):

        idA = self.gtree.tree_data[cylinder_id]['nodeIdA']
        idB = self.gtree.tree_data[cylinder_id]['nodeIdB']

        # vect = nodeA - nodeB
        # self.__draw_circle(nodeB, vect, radius)

        vect = (np.array(nodeA) - np.array(nodeB)).tolist()

# mov circles to center of cylinder by size of radius
        nodeA = self.__translate(nodeA, vect,
                                 -radius * self.endDistMultiplicator)
        nodeB = self.__translate(nodeB, vect,
                                 radius * self.endDistMultiplicator)

        CVlistA = []
        CVlistB = []
        # 1st base
        pts = self.__circle(nodeA, vect, radius)
        ln = len(self.V)

        for i, pt in enumerate(pts):
            self.V.append(pt)
            CVlistA.append(ln + i)

        try:
            self.balls[idA] = self.balls[idA] + CVlistA
        except:
            self.balls[idA] = CVlistA

        # 2nd base
        pts = self.__circle(nodeB, vect, radius)
        ln = len(self.V)

        for i, pt in enumerate(pts):
            self.V.append(pt)
            CVlistB.append(ln + i)
        try:
            self.balls[idB] = self.balls[idB] + CVlistB
            print 'used id ', idB, ' ', CVlistB, ' ', self.balls[idB]
        except:
            import traceback
            traceback.print_exc()

            self.balls[idB] = CVlistB

        CVlist = CVlistA + CVlistB

        self.CV.append(CVlist)

# lar add ball
#         ball0 = mapper.larBall(radius, angle1=PI, angle2=2*PI)([10, 16])
#         V, CV = ball0
#         # mapper.T
#         # ball = STRUCT(MKPOLS(ball0))
#
#         # mapper.T(1)(nodeA[0])(mapper.T(2)(nodeA[1])(mapper.T(3)(nodeA[1])(ball)))
#
#         lenV = len(self.V)
#
#         self.V = self.V + (np.array(V) + np.array(nodeA)).tolist()
#         self.CV = self.CV + (np.array(CV) + lenV).tolist()

    def __translate(self, point, vector, length=None):
        vector = np.array(vector)
        if length is not None:
            vector = length * vector / np.linalg.norm(vector)
        return (np.array(point) + vector).tolist()

    def __add_old_cylinder(self, nodeA, nodeB, radius):
        """
        deprecated simple representation of cylinder
        """
        nodeA = np.array(nodeA)
        nodeB = np.array(nodeB)

        # print nodeB
        ln = len(self.V)
        self.V.append(nodeB.tolist())
        self.V.append((nodeB + [2, 0, 0]).tolist())
        self.V.append((nodeB + [2, 2, 0]).tolist())
        self.V.append((nodeB + [2, 2, 2]).tolist())
        self.V.append((nodeA + [0, 0, 0]).tolist())
        self.CV.append([ln, ln + 1, ln + 2, ln + 3, ln + 4])

    def show(self):
        # print 'balls ', self.balls
        print 'len V ', len(self.V)
        print 'mx ', np.max(np.array(self.balls))
        # print self.balls[self.balls.keys()[0]]
        for joint in self.balls.values():
            print "joint ", joint, type(joint)
            self.CV.append(joint)
            # print 'joint id ', joint_id, " balls  ", self.balls[joint_id]

        V = self.V
        CV = self.CV

        # V = [[0,0,0],[5,5,1],[0,5,5],[5,5,5]]
        # CV = [[0,1,2,3]]
        # print 'V, CV ', V, CV
        VIEW(MKPOL([V, AA(AA(lambda k:k + 1))(CV), []]))

    def get_output(self):
        pass

    def __add_tetr(self, nodeB):
        """
        Creates tetrahedron in specified position.
        """
        try:
            nodeB = nodeB.tolist()
        except:
            pass

        ln = len(self.V)
        self.V.append(nodeB)
        self.V.append((np.array(nodeB) + [2, 0, 0]).tolist())
        self.V.append((np.array(nodeB) + [2, 2, 0]).tolist())
        self.V.append((np.array(nodeB) + [2, 2, 2]).tolist())
        self.CV.append([ln, ln + 1, ln + 2, ln + 3])

    def __add_cone(self, nodeA, nodeB, radius):
        vect = (np.array(nodeA) - np.array(nodeB)).tolist()
        pts = self.__circle(nodeA, vect, radius)

        ln = len(self.V)
        self.V.append(nodeB)
        # first object is top of cone
        CVlist = [ln]

        for i, pt in enumerate(pts):
            self.V.append(pt)
            CVlist.append(ln + i + 1)

        self.CV.append(CVlist)

    def __add_circle(self, center, perp_vect, radius, polygon_element_number=10):
        """
        Draw circle some circle points as tetrahedrons.
        """
        pts = self.__circle(center, perp_vect, radius,
                            polygon_element_number=polygon_element_number)
        for pt in pts:
            self.__add_tetr(pt)

    def __circle(self, center, perp_vect, radius, polygon_element_number=8):
        """
        Function computed the circle points. No drawing.
        perp_vect is vector perpendicular to plane of circle
        """
        # tl = [0, 0.2, 0.4, 0.6, 0.8]
        tl = np.linspace(0, 1, polygon_element_number)

        # vector form center to edge of circle
        # u is a unit vector from the centre of the circle to any point on the
        # circumference

        # normalized perpendicular vector
        n = perp_vect / np.linalg.norm(perp_vect)

        # normalized vector from the centre to point on the circumference
        u = self.__perpendicular_vector(n)
        u = u / np.linalg.norm(u)

        pts = []

        for t in tl:
            # u = np.array([0, 1, 0])
            # n = np.array([1, 0, 0])
            pt = radius * np.cos(t * 2 * np.pi) * u +\
                radius * np.sin(t * 2 * np.pi) * np.cross(u, n) +\
                center

            pt = pt.tolist()
            pts.append(pt)

        return pts

    def __perpendicular_vector(self, v):
        r""" Finds an arbitrary perpendicular vector to *v*."""
        if v[1] == 0 and v[2] == 0:
            if v[0] == 0:
                raise ValueError('zero vector')
            else:
                return np.cross(v, [0, 1, 0])
        return np.cross(v, [1, 0, 0])
