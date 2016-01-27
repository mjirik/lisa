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

from larcc import VIEW, MKPOL, AA, INTERVALS
from splines import all
# import mapper
# from largrid import *

import geometry3d as g3
# import warnings
# warnings.filterwarnings('error')

class GTLar:

    def __init__(self, gtree=None,
                 endDistMultiplicator=1,
                 use_joints=True
                 ):
        """
        gtree is information about input data structure.
        endDistMultiplicator: make cylinder shorter by multiplication of radius
        """
# input of geometry and topology
        self.V = []
        self.CV = []
        self.joints = {}
        self.joints_lar = []
        self.gtree = gtree
        self.endDistMultiplicator = endDistMultiplicator
        self.use_joints = use_joints
        pass

    def add_cylinder(self, nodeA, nodeB, radius, cylinder_id):

        try:
            idA = tuple(nodeA)  # self.gtree.tree_data[cylinder_id]['nodeIdA']
            idB = tuple(nodeB)  # self.gtree.tree_data[cylinder_id]['nodeIdB']
        except:
            idA = 0
            idB = 0
            self.use_joints = False

        # vect = nodeA - nodeB
        # self.__draw_circle(nodeB, vect, radius)

        vector = (np.array(nodeA) - np.array(nodeB)).tolist()

# mov circles to center of cylinder by size of radius because of joint
        nodeA = g3.translate(nodeA, vector,
                             -radius * self.endDistMultiplicator)
        nodeB = g3.translate(nodeB, vector,
                             radius * self.endDistMultiplicator)

        if all(nodeA == nodeB):
            logger.error("End points are on same place")

        ptsA, ptsB = g3.cylinder_circles(nodeA, nodeB, radius,
                                         element_number=30)
        CVlistA = self.__construct_cylinder_end(ptsA, idA)
        CVlistB = self.__construct_cylinder_end(ptsB, idB)

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

    def __construct_cylinder_end(self, pts, id):
        """
        creates end of cylinder and prepares for joints
        """
        CVlist = []
        # base
        ln = len(self.V)

        for i, pt in enumerate(pts):
            self.V.append(pt)
            CVlist.append(ln + i)

        try:
            self.joints[id].append(CVlist)
        except:
            self.joints[id] = [CVlist]

        return CVlist

    def __add_old_cylinder(self, nodeA, nodeB, radius):
        """
        deprecated simple representation of cylinder
        """
        nodeA = np.array(nodeA)
        nodeB = np.array(nodeB)

        ln = len(self.V)
        self.V.append(nodeB.tolist())
        self.V.append((nodeB + [2, 0, 0]).tolist())
        self.V.append((nodeB + [2, 2, 0]).tolist())
        self.V.append((nodeB + [2, 2, 2]).tolist())
        self.V.append((nodeA + [0, 0, 0]).tolist())
        self.CV.append([ln, ln + 1, ln + 2, ln + 3, ln + 4])

    def finish(self):
        if self.use_joints:
            for joint in self.joints.values():
                # There is more then just one circle in this joint, so it
                # is not end of vessel
                if len(joint) > 1:
                    self.__generate_joint(joint)

    def __generate_joint(self, joint):
        joint = (np.array(joint).reshape(-1)).tolist()
        self.CV.append(joint)
        # for circle in joint:
        #     for vertex_id in circle:
        #         print 'v id ', vertex_id
        #         self.V[vertex_id]
        # dom1D = INTERVALS(1)(32);
        # dom2D = TRIANGLE_DOMAIN(32, [[1,0,0],[0,1,0],[0,0,1]]);
        # Cab0 = BEZIER(S0)([[0,2,0],[-2,2,0],[-2,0,0],[-2,-2,0],[0,-2,0]]);
        # Cbc0 = BEZIER(S0)([[0,2,0],[-1,2,0],[-1,1,1],[-1,0,2],[0,0,2]]);
        # Cca0 = BEZIER(S0)([[0,0,2],[-1,0,2],[-1,-1,1],[-1,-2,0],[0,-2,0]]);
        # Cbc1 = BEZIER(S1)([[0,2,0],[-1,2,0],[-1,1,1],[-1,0,2],[0,0,2]]);
        # out1 = MAP(TRIANGULAR_COONS_PATCH([Cab0,Cbc1,Cca0]))(dom2D);
        # self.joints_lar.append(out1)

    def show(self):

        V = self.V
        CV = self.CV

        # V = [[0,0,0],[5,5,1],[0,5,5],[5,5,5]]
        # CV = [[0,1,2,3]]
        # print 'V, CV ', V, CV
        # DRAW(self.joints_lar[0])
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
        pts = g3.circle(center, perp_vect, radius,
                        polygon_element_number=polygon_element_number)
        for pt in pts:
            self.__add_tetr(pt)
