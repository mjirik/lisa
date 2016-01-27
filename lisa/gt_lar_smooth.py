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
sys.path.append(os.path.join(path_to_script, "../lisa/extern"))
sys.path.append(os.path.join(path_to_script, "../../pyplasm/src/pyplasm"))

import numpy as np

from scipy import mat, cos, sin

from larcc import VIEW, MKPOL, AA, INTERVALS, STRUCT, MAP, PROD
from larcc import UNITVECT, VECTPROD, PI, SUM, CAT, IDNT, UNITVECT
from splines import BEZIER, S1, S2, COONSPATCH
# from splines import *
# import mapper
#import hpc
#import pyplasm.hpc


import geometry3d as g3
import interpolation_pyplasm as ip


class GTLarSmooth:

    def __init__(self, gtree=None):
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
        self.endDistMultiplicator = 2
        self.use_joints = True
        #dir(splines)
        pass

    def add_cylinder(self, nodeA, nodeB, radius, cylinder_id):

        try:
            idA = tuple(nodeA)  # self.gtree.tree_data[cylinder_id]['nodeIdA']
            idB = tuple(nodeB)  # self.gtree.tree_data[cylinder_id]['nodeIdB']
        except:
            idA = 0
            idB = 0
            self.use_joints = False

        vect = np.array(nodeA) - np.array(nodeB)
        u = g3.perpendicular_vector(vect)
        u = u / np.linalg.norm(u)
        u = u.tolist()
        vect = vect.tolist()



        c1 = self.__circle(nodeA, radius, vect)
        c2 = self.__circle(nodeB, radius, vect)
        tube = BEZIER(S2)([c1,c2])
        domain = PROD([ INTERVALS(2*PI)(36), INTERVALS(1)(4) ])
        tube = MAP(tube)(domain)


        self.joints_lar.append(tube)


        #self.__draw_circle(nodeB, vect, radius)

        ##vector = (np.array(nodeA) - np.array(nodeB)).tolist()

# mov circles to center of cylinder by size of radius because of joint
        ##nodeA = g3.translate(nodeA, vector,
        ##                     -radius * self.endDistMultiplicator)
        ##nodeB = g3.translate(nodeB, vector,
        ##                     radius * self.endDistMultiplicator)

        ##ptsA, ptsB = g3.cylinder_circles(nodeA, nodeB, radius, element_number=32)
        ##CVlistA = self.__construct_cylinder_end(ptsA, idA, nodeA)
        ##CVlistB = self.__construct_cylinder_end(ptsB, idB, nodeB)

        ##CVlist = CVlistA + CVlistB

        ##self.CV.append(CVlist)

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

    def __circle(self, center=[0,0,0],radius=1,normal=[0,0,1],sign=1,shift=0):
        N = UNITVECT(normal)
        if N == [0,0,1] or N == [0,0,-1]: Q = mat(IDNT(3))
        else: 
            QX = UNITVECT((VECTPROD([[0,0,1],N])))
            QZ = N
            QY = VECTPROD([QZ,QX])
            Q = mat([QX,QY,QZ]).T
        def circle0(p):
            u = p[0]
            x = radius*cos(sign*u+shift)
            y = radius*sin(sign*u+shift)
            z = 0
            return SUM([ center, CAT((Q*[[x],[y],[z]]).tolist()) ])
        return circle0


    def __construct_cylinder_end(self, pts, id, node):
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
            self.joints[id].append([node, CVlist])
        except:
            self.joints[id] = [[node, CVlist]]

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
        print 'use joints? ', self.use_joints
        if self.use_joints:
            for joint in self.joints.values():
                # There is more then just one circle in this joint, so it
                # is not end of vessel
                if len(joint) > 1:
                    self.__generate_joint(joint)


    def __half_plane(self, perp, plane_point, point):
        cdf = (np.array(point) - np.array(plane_point))
        out = perp[0] * cdf[0] +\
            perp[1] * cdf[1] + \
            perp[2] * cdf[2]
        return  out > 0

    def __get_vessel_connection_curve(self, vessel_connection, perp, vec0, vec1):
        """
        perp is perpendicular to plane given by centers of circles
        vec1, vec0 are vectors from circle centers
        """
        curve_t = []
        curve_d = []
        curve_pts_indexes_t = []
        curve_pts_indexes_d = []
        brake_point_t = None
        brake_point_d = None
        center, circle = vessel_connection

        # left to right
        perp_lr = np.cross(perp, vec1)

        print 'center ', center
        print 'circle ', circle
        for vertex_id in circle:
            if ((len(curve_pts_indexes_t) > 0) and 
                (vertex_id - curve_pts_indexes_t[-1]) > 1):
                brake_point_t = len(curve_pts_indexes_t)
            if ((len(curve_pts_indexes_d) > 0) and 
                (vertex_id - curve_pts_indexes_d[-1]) > 1):
                brake_point_d = len(curve_pts_indexes_d)

            #hp = self.__half_plane(perp_lr, center, self.V[vertex_id])
            hp = self.__half_plane(perp, center, self.V[vertex_id])

            
            if(hp):
                curve_t.append(self.V[vertex_id])
                curve_pts_indexes_t.append(vertex_id)
            else:
                curve_d.append(self.V[vertex_id])
                curve_pts_indexes_d.append(vertex_id)

        ordered_curve_t = curve_t[brake_point_t:] + curve_t[:brake_point_t]
        ordered_pts_indexes_t = \
            curve_pts_indexes_t[brake_point_t:] +\
            curve_pts_indexes_t[:brake_point_t]

        ordered_curve_d = curve_d[brake_point_d:] + curve_d[:brake_point_d]
        ordered_pts_indexes_d = \
            curve_pts_indexes_d[brake_point_t:] +\
            curve_pts_indexes_d[:brake_point_d]
        #print '    hp v id ', curve_pts_indexes_t    
        #print 'ord hp v id ', ordered_pts_indexes_t

        #print 'hp circle ', curve_one

        # add point from oposit half-circle
        first_pt_d = ordered_curve_d[0]
        last_pt_d = ordered_curve_d[-1]
        first_pt_t = ordered_curve_t[0]
        last_pt_t = ordered_curve_t[-1]

        ordered_curve_t.append(first_pt_d)
        ordered_curve_t.insert(0, last_pt_d)

        ordered_curve_d.append(first_pt_t)
        ordered_curve_d.insert(0, last_pt_t)

        return ordered_curve_t, ordered_curve_d

    def __generate_joint(self, joint):
        #joint = (np.array(joint).reshape(-1)).tolist()
        #self.CV.append(joint)
        cc0 = np.array(joint[0][0])
        cc1 = np.array(joint[1][0])
        cc2 = np.array(joint[2][0])

        vec0 = cc0 - cc1
        vec1 = cc1 - cc2

        perp = np.cross(vec0, vec1)


        curvelistT = []
        curvelistD = []

        for vessel_connection in joint:
            ordered_curve_t, ordered_curve_d = self.__get_vessel_connection_curve(
                vessel_connection, perp, vec0, vec1)
            


            curvelistT.append(ordered_curve_t)
            curvelistD.append(ordered_curve_d)
                #print '  ', self.V[vertex_id], '  hp: ', hp

        Betacurve_id, Astart, Alphacurve_id, Bstart, Gammacurve_id, Cstart = self.__find_couples(curvelistT)
        
        #print 'ABC ', Betacurve_id, Astart, Alphacurve_id, Bstart

        dom2D = ip.TRIANGLE_DOMAIN(32, [[1,0,0],[0,1,0],[0,0,1]])
        Cab0 = BEZIER(S1)(self.__order_curve(curvelistT[Gammacurve_id][-1:0:-1], Cstart))
        Cbc0 = BEZIER(S1)(self.__order_curve(curvelistT[Alphacurve_id], Bstart))
        Cbc1 = BEZIER(S2)(self.__order_curve(curvelistT[Alphacurve_id], Bstart))
        Cca0 = BEZIER(S1)(self.__order_curve(curvelistT[Betacurve_id][-1:0:-1], Astart))
        
        out1 = MAP(ip.TRIANGULAR_COONS_PATCH([Cab0,Cbc1,Cca0]))(STRUCT(dom2D))
        self.joints_lar.append(out1)

        Betacurve_id, Astart, Alphacurve_id, Bstart, Gammacurve_id, Cstart = self.__find_couples(curvelistD)
        
        #print 'ABC ', Betacurve_id, Astart, Alphacurve_id, Bstart

        dom2D = ip.TRIANGLE_DOMAIN(32, [[1,0,0],[0,1,0],[0,0,1]])
        Cab0 = BEZIER(S1)(self.__order_curve(curvelistD[Gammacurve_id][-1:0:-1], Cstart))
        Cbc0 = BEZIER(S1)(self.__order_curve(curvelistD[Alphacurve_id], Bstart))
        Cbc1 = BEZIER(S2)(self.__order_curve(curvelistD[Alphacurve_id], Bstart))
        Cca0 = BEZIER(S1)(self.__order_curve(curvelistD[Betacurve_id][-1:0:-1], Astart))

        out2 = MAP(ip.TRIANGULAR_COONS_PATCH([Cab0,Cbc1,Cca0]))(STRUCT(dom2D))
        self.joints_lar.append(out2)

    def __find_couples(self, curvelist):
        """
        try find all posible couples with minimal energy. 
        Energy is defined like sum of distances
        """
        energy = None
        mn_ind = None
        output = None
        for i in range(0,3):
            Betacurve_id, Astart, dist0 = self.__find_nearest(
                curvelist, i, 0, [i])
            Alphacurve_id, Bstart, dist1 = self.__find_nearest(
                curvelist, i, -1, [i, Betacurve_id])
            this_energy = dist0 + dist1

            if energy is None or this_energy < energy:
                energy = this_energy
                mn_ind = i
                #Gammacurve_id = i
                output = Betacurve_id, Astart, Alphacurve_id, Bstart, i, 0


            Betacurve_id, Astart, dist0 = self.__find_nearest(
                curvelist, i, -1, [i])
            Alphacurve_id, Bstart, dist1 = self.__find_nearest(
                curvelist, i, 0, [i, Betacurve_id])
            this_energy = dist0 + dist1

            if energy is None or this_energy < energy:
                energy = this_energy
                mn_ind = i
                output = Betacurve_id, Astart, Alphacurve_id, Bstart, i, -1

        print 'output'
        print output

        return output


    def __order_curve(self, curve, start):
        if start is 0:
            return curve
        else:
            return curve[-1:0:-1]

    def __find_nearest(self, curvelist, this_curve_index, start, wrong_curve=None):
        """
        start: use 0 or -1
        """
        #if start:
        #    start_index = 0
        #else:
        #    start_index = -1
        if wrong_curve is None:
            wrong_curve = [this_curve_index]
        dist = None
        min_cv_ind = None
        min_cv_start = None

        for curve_index in range(0, len(curvelist)):
            if curve_index not in wrong_curve:
                pt0 = np.array(curvelist[this_curve_index][start])
                pt1 = np.array(curvelist[curve_index][0])
                this_dist = np.linalg.norm(pt0 - pt1)
                if (dist is None) or (this_dist < dist):
                    dist = this_dist
                    min_cv_ind = curve_index
                    min_cv_start = 0

                pt1 = np.array(curvelist[curve_index][-1])
                this_dist = np.linalg.norm(pt0 - pt1)
                if (dist is None) or (this_dist < dist):
                    dist = this_dist
                    min_cv_ind = curve_index
                    min_cv_start = -1

        return min_cv_ind, min_cv_start, dist





    def show(self):

        V = self.V
        CV = self.CV

        # V = [[0,0,0],[5,5,1],[0,5,5],[5,5,5]]
        # CV = [[0,1,2,3]]
        # print 'V, CV ', V, CV
        #for joint in self.joints_lar:

        # out = STRUCT([MKPOL([V, AA(AA(lambda k:k + 1))(CV), []])] + self.joints_lar)
        out = STRUCT(self.joints_lar)
        #VIEW(self.joints_lar[0])
        #VIEW(MKPOL([V, AA(AA(lambda k:k + 1))(CV), []]))
        VIEW(out)
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
