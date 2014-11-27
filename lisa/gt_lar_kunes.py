#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Surface model generator of vessel tree.
Used by gen_volume_tree.py
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
from splines import *
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

    def add_cylinder(self, nodeA, nodeB, radius, cylinder_id=None):
        """
        cylinder_id is not needed anymore
        """        
        
        try:
            idA = tuple(nodeA)  # self.gtree.tree_data[cylinder_id]['nodeIdA']
            idB = tuple(nodeB)  # self.gtree.tree_data[cylinder_id]['nodeIdB']
        except:
            idA = 0
            idB = 0
            self.use_joints = False

        # # mov circles to center of cylinder by size of radius because of joint
        # make cylinder shorter by multiplication of radius
        #vector = (np.array(nodeA) - np.array(nodeB)).tolist()
        #nodeA = g3.translate(nodeA, vector,
                             #-radius * self.endDistMultiplicator)
        #nodeB = g3.translate(nodeB, vector,
                             #radius * self.endDistMultiplicator)

        if all(nodeA == nodeB):
            logger.error("End points are on same place")
        
        # generate lists of points of two end circles
        ptsA, ptsB = g3.cylinder_circles(nodeA, nodeB, radius,
                                         element_number=30)
        
        # gives points unique global id. merges to global list of points.
        # self.joints[id] == list of lists of ids of cylinder points that 
        # belong to joint idA/idB.
        CVlistA = self.__construct_cylinder_end(ptsA, idA)
        CVlistB = self.__construct_cylinder_end(ptsB, idB)
        CVlist = CVlistA + CVlistB

        self.CV.append(CVlist)

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

    def finish(self):
        if self.use_joints:
            for joint in self.joints.values():
                # There is more then just one circle in this joint, so it
                # is not end of vessel
                if len(joint) > 1:
                    self.__generate_joint(joint)

    def __generate_joint(self, joint):
        # get cylinder info - get points of other side of cylinder, that is not 
        # connected to current joint
        cylinders = []
        for i, j in enumerate(joint):
            for cv_list in self.CV:
                if j[0] in cv_list:
                    far_points = []
                    for p in cv_list:
                        if p not in j:
                            far_points.append(p)
                                
                    cylinders.append({'near_points':j, 'far_points':far_points})               
                    break
        
        # get cylinder info - get radiuses, node positions and vectors of 
        # connected cylinders
        for c in cylinders:
            p1 = np.array(self.V[c['near_points'][0]])
            p2 = np.array(self.V[c['near_points'][len(c['near_points'])/2]])
            
            dist = np.linalg.norm(p1-p2) / 2.0
            c['radius'] = dist
            
            # get position of nodes
            c['near_node'] = (p2 + ((p1-p2) / 2.0)).tolist()
            
            p1 = np.array(self.V[c['far_points'][0]])
            p2 = np.array(self.V[c['far_points'][len(c['far_points'])/2]])
            c['far_node'] = (p2 + ((p1-p2) / 2.0)).tolist()
            
            c['vector'] = (np.array(c['near_node']) - 
                                np.array(c['far_node'])).tolist()
        
        
        # make cylinders shorter by max radius to create place for joint
        max_radius = 0
        for c in cylinders:
            if c['radius'] > max_radius:
                max_radius = c['radius'] 
         
        for c in cylinders:
            start_id = c['near_points'][0]
            end_id = c['near_points'][len(c['near_points'])-1]
            
            for p_id in range(start_id, end_id+1):
                self.V[p_id] = g3.translate(self.V[p_id], c['vector'],
                                    -max_radius)
        
        
        # Takes all lists of points of circles that belong to joint and 
        # merge-copy them to one new list.
        # Points in list are covered with surface => this creates joint.
        joint = (np.array(joint).reshape(-1)).tolist()
        self.CV.append(joint)
        

    def show(self):
        V = self.V
        CV = self.CV

        VIEW(MKPOL([V, AA(AA(lambda k:k + 1))(CV), []]))

    def get_output(self):
        pass

    #def __add_tetr(self, nodeB):
        #"""
        #Creates tetrahedron in specified position.
        #"""
        #try:
            #nodeB = nodeB.tolist()
        #except:
            #pass

        #ln = len(self.V)
        #self.V.append(nodeB)
        #self.V.append((np.array(nodeB) + [2, 0, 0]).tolist())
        #self.V.append((np.array(nodeB) + [2, 2, 0]).tolist())
        #self.V.append((np.array(nodeB) + [2, 2, 2]).tolist())
        #self.CV.append([ln, ln + 1, ln + 2, ln + 3])

    #def __add_cone(self, nodeA, nodeB, radius):
        #vect = (np.array(nodeA) - np.array(nodeB)).tolist()
        #pts = self.__circle(nodeA, vect, radius)

        #ln = len(self.V)
        #self.V.append(nodeB)
        ## first object is top of cone
        #CVlist = [ln]

        #for i, pt in enumerate(pts):
            #self.V.append(pt)
            #CVlist.append(ln + i + 1)

        #self.CV.append(CVlist)

    #def __add_circle(self, center, perp_vect, radius, polygon_element_number=10):
        #"""
        #Draw circle some circle points as tetrahedrons.
        #"""
        #pts = g3.circle(center, perp_vect, radius,
                        #polygon_element_number=polygon_element_number)
        #for pt in pts:
            #self.__add_tetr(pt)
