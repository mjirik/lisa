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
from splines import all
# import mapper
# from largrid import *

import geometry3d as g3
# import warnings
# warnings.filterwarnings('error')

class GTLar:
    """
    gtree is information about input data structure.
    endDistMultiplicator: move connected side of cylinders away from joint by multiplication of radius
    """
    def __init__(self, gtree=None,
                 endDistMultiplicator=0.5,
                 use_joints=True
                 ):
        
        logger.debug('__init__:use_joints = '+str(use_joints))
        logger.debug('__init__:endDistMultiplicator = '+str(endDistMultiplicator))
        
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
        """
        Generate joints for cylindr connections
        """
        if self.use_joints:
            logger.debug('generating joints...')
            
            for i, joint in enumerate(self.joints.values()):
                if i%10 == 0:
                    logger.debug('joint '+str(i)+'/'+str(len(self.joints.values())))
                # There is more then just one circle in this joint, so it
                # is not end of vessel
                if len(joint) > 1:
                    self.__generate_joint(joint)
                    
            logger.debug('joints generated')
                    
    def __get_cylinder_info_from_raw_joint(self, joint):
        """
        joint - joint value (list of lists), before being processed by finish(), or __generate_joint()
        
        Returns list of dictionaries with information about cylinders connected to current joint
        
            | cylinders[i]['near_points'] = list of point ids of circle on the side connected to current joint (node) 
            | cylinders[i]['far_points'] = list of point ids of circle on the other side of cylinder 
            | cylinders[i]['radius'] = radius of cylinder
            | cylinders[i]['near_node'] = position of node that is connected to current joint 
            | cylinders[i]['far_node'] = position of node on the other side of cylinder 
            | cylinders[i]['vector'] = vector of line connection between nodes
        """
        # get points of other side of cylinder, that is not connected to 
        # current joint
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
        
        # get radiuses, node positions and vectors of connected cylinders
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
                                
        return cylinders
        
    def __point_in_cylinder(self, c_nodeA, c_nodeB, radius, point, length_sq=None, radius_sq=None):
        """
        Tests if point is inside a cylinder
        http://www.flipcode.com/archives/Fast_Point-In-Cylinder_Test.shtml
        """
        if length_sq is None:
            if c_nodeA == c_nodeB:
                # wierd cylinder with 0 length
                logger.warning('__point_in_cylinder: distance between nodeA and nodeB is zero!!')
                raise Exception('distance between nodeA and nodeB is zero')
            length_sq = np.linalg.norm(np.array(c_nodeA)-np.array(c_nodeB))**2
        if radius_sq is None:
            radius_sq = radius**2
            
        dx = c_nodeB[0] - c_nodeA[0]	
        dy = c_nodeB[1] - c_nodeA[1]   
        dz = c_nodeB[2] - c_nodeA[2]

        pdx = point[0] - c_nodeA[0]	
        pdy = point[1] - c_nodeA[1]
        pdz = point[2] - c_nodeA[2]
        
        dot = pdx * dx + pdy * dy + pdz * dz
        
        if (dot < 0) or (dot > length_sq):
            # points is outside of end caps
            return False
        else:
            dsq = (pdx*pdx + pdy*pdy + pdz*pdz) - (dot*dot)/float(length_sq)
            
            if dsq > radius_sq:
                # point not inside cylinder
                return False
            else:
                return True
            

    def __generate_joint(self, joint):
        # get cylinder info
        cylinders = self.__get_cylinder_info_from_raw_joint(joint)
        
        # move connected side of cylinders away from joint by 
        # radius*self.endDistMultiplicator to create more place for joint
        for c in cylinders:
            if c['far_node'] == c['near_node']:
                # wierd cylinder with 0 length
                continue
            
            start_id = c['near_points'][0]
            end_id = c['near_points'][len(c['near_points'])-1]
            
            for p_id in range(start_id, end_id+1):
                self.V[p_id] = g3.translate(self.V[p_id], c['vector'],
                                    -c['radius']*self.endDistMultiplicator)
                # TODO - detect when g3.translate would create negative length
        # update cylinder info after moving points
        cylinders = self.__get_cylinder_info_from_raw_joint(joint)
                                    
        
        # cut out overlapping parts of cylinders (only in joint)
        new_V = list(self.V)
        for c in cylinders:
            # for every cylinder in joint...
            v = np.array(c['vector'])
            c_len = np.linalg.norm(np.array(c['near_node'])-np.array(c['far_node']))
            if c_len == 0:
                # wierd cylinder with 0 length
                continue
            
            for p_id in c['near_points']:
                # for every point in cylinder that is on the side connected to 
                # joint...
                orig_near_point = np.array(self.V[p_id])
                
                # get position of coresponding point on the far side for the 
                # point on the near side.
                # cylinder must be uncut for this to work correctly
                far_point = orig_near_point - v
                
                for cc in cylinders:
                    # for other cylinders connected to joint...
                    
                    if cc['near_points'] == c['near_points']:
                        # skip cylinder that owns tested point
                        continue 
                    elif cc['far_node'] == cc['near_node']:
                        # different cylinder, but has 0 length
                        continue
                    
                    current_point = far_point.copy()
                    current_point_last = far_point.copy()
                    
                    if self.__point_in_cylinder(cc['near_node'], cc['far_node'], cc['radius'], current_point):
                        continue # far point is inside !!! 
                    
                    while np.linalg.norm(current_point-far_point) <= c_len:
                        # slowly go from position of far_point to near_point, 
                        # and when the next step would be inside of cylinder, 
                        # set position of near_node to current position...
                        current_point_last = current_point.copy()
                        current_point = current_point + v/10.0 # move by 10% of length
                        
                        if self.__point_in_cylinder(cc['near_node'], cc['far_node'], cc['radius'], current_point):
                            if np.linalg.norm(current_point_last-far_point) < np.linalg.norm(np.array(new_V[p_id])-far_point):
                                new_V[p_id] = list(current_point_last)
                            break
                            
        self.V = list(new_V)
        
        
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
