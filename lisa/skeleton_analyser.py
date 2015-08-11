#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Modul is used for skeleton binary 3D data analysis
"""

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))

import logging
logger = logging.getLogger(__name__)

import traceback

import numpy as np
import scipy.ndimage


class SkeletonAnalyser:

    """
    | Example:
    | skan = SkeletonAnalyser(data3d_skel, volume_data, voxelsize_mm)
    | stats = skan.skeleton_analysis()
    
    | data3d_skel: 3d array with skeleton as 1s and background as 0s
    | use_filter_small_objects: removing small objects
    | filter_small_threshold: threshold for small filtering

    :arg cut_wrong_skeleton: remove short skeleton edges to terminal
    :arg aggregate_near_nodes_distance: combine near nodes to one. Parameter 
    defines distance in mm.
    """

    def __init__(self, data3d_skel, volume_data=None, voxelsize_mm=[1, 1, 1],
                 use_filter_small=False, filter_small_threshold=3,
                 cut_wrong_skeleton=True, aggregate_near_nodes_distance=0):
        # for not
        self.volume_data = volume_data
        self.voxelsize_mm = voxelsize_mm
        self.aggregate_near_nodes_distance = aggregate_near_nodes_distance

        # get array with 1 for edge, 2 is node and 3 is terminal
        logger.debug('Generating sklabel...')
        if use_filter_small:
            data3d_skel = self.filter_small_objects(data3d_skel,
                                            filter_small_threshold)
        
        # generate nodes and enges (sklabel)
        logger.debug('__skeleton_nodes, __generate_sklabel')
        skelet_nodes = self.__skeleton_nodes(data3d_skel)
        self.sklabel = self.__generate_sklabel(skelet_nodes)

        self.cut_wrong_skeleton = cut_wrong_skeleton
        self.curve_order = 2
        self.spline_smoothing = None

        logger.debug('Inited SkeletonAnalyser - voxelsize:' + str(
            voxelsize_mm) + ' volumedata:' + str(volume_data is not None))

    def skeleton_analysis(self, guiUpdateFunction=None):
        """
        | Glossary:
        | element: line structure of skeleton connected to node on both ends. (index>0)
        | node: connection point of elements. It is one or few voxelsize_mm. (index<0)
        | terminal: terminal node
        """
        def updateFunction(num, length, part):
            if int(length / 100.0) == 0 or \
                    (num % int(length / 100.0) == 0) or num == length:
                if guiUpdateFunction is not None:
                    guiUpdateFunction(num, length, part)
                logger.info('skeleton_analysis: processed ' + str(
                    num) + '/' + str(length) + ', part ' + str(part))
        
        if self.cut_wrong_skeleton:
            updateFunction(0, 1, "cuting wrong skeleton")
            self.__cut_short_skeleton_terminal_edges()
        
        stats = {}
        len_edg = np.max(self.sklabel)
        len_node = np.min(self.sklabel)
        logger.debug(
            'len_edg: ' + str(len_edg) + ' len_node: ' + str(len_node))
        
        # init radius analysis 
        logger.debug('__radius_analysis_init')
        if self.volume_data is not None:
            skdst = self.__radius_analysis_init()
        
        # get edges and nodes that are near the edge. (+bounding box)
        logger.debug(
            'skeleton_analysis: starting element_neighbors processing')
        self.elm_neigh = {}
        self.elm_box = {}
        for edg_number in (range(len_node, 0) + range(1, len_edg + 1)):
            self.elm_neigh[edg_number], self.elm_box[
                edg_number] = self.__element_neighbors(edg_number)
            # update gui progress
            updateFunction(edg_number + abs(len_node) + 1, abs(
                len_node) + len_edg + 1,
                "generating node->connected_edges lookup table")
        logger.debug(
            'skeleton_analysis: finished element_neighbors processing')
        # clear unneeded data. IMPORTANT!!
        del(self.shifted_zero) # needed by __element_neighbors
        del(self.shifted_sklabel) # needed by __element_neighbors
        
        # get main stats
        logger.debug(
            'skeleton_analysis: starting processing part: length, radius, ' +
            'curve and connections of edge')
        for edg_number in range(1, len_edg + 1):
            edgst = {}
            edgst.update(self.__connection_analysis(edg_number))
            edgst.update(self.__edge_curve(edg_number, edgst))
            edgst.update(self.__edge_length(edg_number, edgst))
            edgst.update(self.__edge_vectors(edg_number, edgst))
            # edgst = edge_analysis(sklabel, i)
            if self.volume_data is not None:
                edgst['radius_mm'] = float(self.__radius_analysis(
                    edg_number, skdst))  # slow (this takes most of time)
            stats[edgst['id']] = edgst

            # update gui progress
            updateFunction(
                edg_number, len_edg,
                "length, radius, curve, connections of edge")
        logger.debug(
            'skeleton_analysis: finished processing part: length, radius, ' +
            'curve, connections of edge')

        # @TODO dokončit
        # logger.debug(
        # 'skeleton_analysis: starting processing part:
        # angles of connected edges')
        # for edg_number in range (1,len_edg+1):
            # edgst = stats[edg_number]
            # edgst.update(self.__connected_edge_angle(edg_number, stats))

            # updateFunction(edg_number,len_edg, "angles of connected edges")
        # logger.debug('skeleton_analysis: finished processing part: angles of
        # connected edges')

        return stats

    def __remove_edge_from_stats(self, stats, edge):
        logger.debug('Cutting edge id:' + str(edge) + ' from stats')
        edg_stats = stats[edge]

        connected_edgs = edg_stats[
            'connectedEdgesA'] + edg_stats['connectedEdgesB']

        for connected in connected_edgs:
            try:
                stats[connected]['connectedEdgesA'].remove(edge)
            except:
                pass

            try:
                stats[connected]['connectedEdgesB'].remove(edge)
            except:
                pass

        del stats[edge]

        return stats

    def __cut_short_skeleton_terminal_edges(self, cut_ratio=2.0):
        """
        cut_ratio = 2.0 -> if radius of terminal edge is 2x its lenght or more,
        remove it
        """
        
        def remove_elm(elm_id, elm_neigh, elm_box, sklabel):
            sklabel[sklabel == elm_id] = 0 
            del(elm_neigh[elm_id])
            del(elm_box[elm_id])
            for elm in elm_neigh:
                elm_neigh[elm] = [x for x in elm_neigh[elm] if x != elm]
            return elm_neigh, elm_box, sklabel
        
        len_edg = np.max(self.sklabel)
        len_node = np.min(self.sklabel)
        logger.debug(
            'len_edg: ' + str(len_edg) + ' len_node: ' + str(len_node))
        
        # get edges and nodes that are near the edge. (+bounding box)
        logger.debug(
            'skeleton_analysis: starting element_neighbors processing')
        self.elm_neigh = {}
        self.elm_box = {}
        for edg_number in (range(len_node, 0) + range(1, len_edg + 1)):
            self.elm_neigh[edg_number], self.elm_box[
                edg_number] = self.__element_neighbors(edg_number)
        logger.debug(
            'skeleton_analysis: finished element_neighbors processing')
        # clear unneeded data. IMPORTANT!!
        del(self.shifted_zero) # needed by __element_neighbors
        del(self.shifted_sklabel) # needed by __element_neighbors
        # mozna fix kratkodobych potizi, ale skutecny problem byl jinde
        # try:
        #     del(self.shifted_zero) # needed by __element_neighbors
        # except:
        #     logger.warning('self.shifted_zero does not exsist')
        # try:
        #     del(self.shifted_sklabel) # needed by __element_neighbors
        # except:
        #     logger.warning('self.shifted_zero does not exsist')
        
        # remove edges+nodes that are not connected to rest of the skeleton                
        logger.debug(
            'skeleton_analysis: Cut - Removing edges that are not' +
            ' connected to rest of the skeleton (not counting its nodes)')
        cut_elm_neigh = dict(self.elm_neigh)
        cut_elm_box = dict(self.elm_box)
        for elm in self.elm_neigh:
            elm = int(elm)
            if elm>0: # if edge
                conn_nodes = [i for i in self.elm_neigh[elm] if i < 0] 
                conn_edges = []
                for n in conn_nodes:
                    try:
                        nn = self.elm_neigh[n] # get neighbours elements of node
                    except:
                        logger.debug('Node '+str(n)+' not found! May be already deleted.')
                        continue
                    
                    for e in nn: # if there are other edges connected to node add them to conn_edges
                        if e>0 and e not in conn_edges and e !=elm:
                            conn_edges.append(e)
                
                if len(conn_edges)==0: # if no other edges are connected to nodes, remove from skeleton
                    logger.debug("removing edge "+str(elm)+" with its nodes "+str(self.elm_neigh[elm]))
                    for night in self.elm_neigh[elm]:
                        remove_elm(night, cut_elm_neigh, cut_elm_box, self.sklabel)
        self.elm_neigh = cut_elm_neigh
        self.elm_box = cut_elm_box
        
        # remove elements that are not connected to the rest of skeleton
        logger.debug(
            'skeleton_analysis: Cut - Removing elements that are not connected' +
            ' to rest of the skeleton')
        cut_elm_neigh = dict(self.elm_neigh)
        cut_elm_box = dict(self.elm_box)
        for elm in self.elm_neigh:
            elm = int(elm)
            if len(self.elm_neigh[elm]) == 0:
                logger.debug("removing element "+str(elm))
                remove_elm(elm, cut_elm_neigh, cut_elm_box, self.sklabel)
        self.elm_neigh = cut_elm_neigh
        self.elm_box = cut_elm_box
        
        # get list of terminal nodes
        logger.debug('skeleton_analysis: Cut - get list of terminal nodes')
        terminal_nodes = []
        for elm in self.elm_neigh:
            if elm<0: # if node
                conn_edges = [i for i in self.elm_neigh[elm] if i > 0] 
                if len(conn_edges) == 1: # if only one edge is connected
                    terminal_nodes.append(elm)

        # init radius analysis 
        logger.debug('__radius_analysis_init')
        if self.volume_data is not None:
            skdst = self.__radius_analysis_init()

        # removes end terminal edges based on radius/length ratio
        logger.debug('skeleton_analysis: Cut - Removing bad terminal edges based on'+
            ' radius/length ratio')
        cut_elm_neigh = dict(self.elm_neigh)
        cut_elm_box = dict(self.elm_box)
        for tn in terminal_nodes:
            te = [i for i in self.elm_neigh[tn] if i > 0][0] # terminal edge
            radius = float(self.__radius_analysis(te, skdst))
            edgst = self.__connection_analysis(int(te))
            edgst.update(self.__edge_length(edg_number, edgst))
            length = edgst['lengthEstimation']
            
            #logger.debug(str(radius / float(length))+" "+str(radius)+" "+str(length))
            if (radius / float(length)) > cut_ratio:
                logger.debug("removing edge "+str(te)+" with its terminal node.")
                remove_elm(elm, cut_elm_neigh, cut_elm_box, self.sklabel)
        self.elm_neigh = cut_elm_neigh
        self.elm_box = cut_elm_box
        
        # check if some nodes are not forks but just curves
        logger.debug('skeleton_analysis: Cut - check if some nodes are not forks but just curves')
        for elm in self.elm_neigh:
            if elm<0:
                conn_edges = [i for i in self.elm_neigh[elm] if i > 0] 
                if len(conn_edges) == 2:
                    logger.warning('Node '+str(elm)+' is just a curve!!! FIX THIS!!!')
                    # TODO
        
        # regenerate new nodes and edges from cut skeleton (sklabel)
        logger.debug('regenerate new nodes and edges from cut skeleton')
        self.sklabel[self.sklabel!=0] = 1
        skelet_nodes = self.__skeleton_nodes(self.sklabel)
        self.sklabel = self.__generate_sklabel(skelet_nodes)



    def __skeleton_nodes(self, data3d_skel):
        """
        Return 3d ndarray where 0 is background, 1 is skeleton, 2 is node
        and 3 is terminal node
        """
        kernel = np.ones([3, 3, 3])

        mocnost = scipy.ndimage.filters.convolve(
            data3d_skel, kernel) * data3d_skel

        nodes = (mocnost > 3).astype(np.int8)
        terminals = ((mocnost == 2) | (mocnost == 1)).astype(np.int8)

        data3d_skel[nodes == 1] = 2
        data3d_skel[terminals == 1] = 3
        data3d_skel = self.__skeleton_nodes_aggregation(data3d_skel)

        return data3d_skel

    def __skeleton_nodes_aggregation(self, data3d_skel):
        """

        aggregate near nodes
        """

        method = 'auto'
        if self.aggregate_near_nodes_distance >= 0:
            # print 'generate structure'
            structure = generate_binary_elipsoid(
                    self.aggregate_near_nodes_distance / np.asarray(self.voxelsize_mm))
            # print 'perform dilation ', data3d_skel.shape
            # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

            # TODO select best method
            # old simple method
            # nd_dil = scipy.ndimage.binary_dilation(data3d_skel==2, structure)

            # per partes method even slower
            nd_dil = self.__skeleton_nodes_aggregation_per_each_node(data3d_skel==2, structure)
            # print "dilatation completed"
            data3d_skel[nd_dil] = 2
            # print 'set to 2'
        return data3d_skel

    def __skeleton_nodes_aggregation_per_each_node(self, data3d_skel2, structure):
        print data3d_skel2.dtype
        node_list = np.nonzero(data3d_skel2)
        nlz = zip(node_list[0], node_list[1], node_list[2])

        # print 'zip finished'

        for node_xyz in nlz:
            self.__node_dilatation(data3d_skel2, node_xyz, structure)

        return data3d_skel2


    def __node_dilatation(self, data3d_skel2, node_xyz, structure):
        """
        this function is called for each node
        """
        border = structure.shape
        
        # print 'border ', border
        # print structure.shape
        xlim = [max(0, node_xyz[0] - border[0]), min(data3d_skel2.shape[0], node_xyz[0] + border[0])]
        ylim = [max(0, node_xyz[1] - border[1]), min(data3d_skel2.shape[1], node_xyz[1] + border[1])]
        zlim = [max(0, node_xyz[2] - border[2]), min(data3d_skel2.shape[2], node_xyz[2] + border[2])]


        # dilation on small box
        nd_dil = scipy.ndimage.binary_dilation(
            data3d_skel2[xlim[0]:xlim[1],
                         ylim[0]:ylim[1],
                         zlim[0]:zlim[1]]==2, structure)

        # nd_dil = nd_dil * 2

        data3d_skel2[xlim[0]:xlim[1],
                        ylim[0]:ylim[1],
                        zlim[0]:zlim[1]]=nd_dil


    def __label_edge_by_its_terminal(self, labeled_terminals):
        import functools
        import scipy

        def max_or_zero(a):
            return min(np.max(a), 0)
        fp = np.ones([3, 3, 3], dtype=np.int)
        median_filter = functools.partial(
            scipy.ndimage.generic_filter, function=np.max, footprint=fp)
        mf = median_filter(labeled_terminals)

        for label in range(np.min(labeled_terminals), 0):
            neigh = np.min(mf[labeled_terminals == label])
            labeled_terminals[labeled_terminals == neigh] = label
        return labeled_terminals

    def filter_small_objects(self, skel, threshold=4):
        """
        Remove small objects from 
        terminals are connected to edges
        """
        skeleton_nodes = self.__skeleton_nodes(skel)
        logger.debug('skn 2 ' + str(np.sum(skeleton_nodes == 2)))
        logger.debug('skn 3 ' + str(np.sum(skeleton_nodes == 3)))
# delete nodes
        nodes = skeleton_nodes == 2
        skeleton_nodes[nodes] = 0
        # pe = ped.sed3(skeleton_nodes)
        # pe.show()
        labeled_terminals = self.__generate_sklabel(skeleton_nodes)

        logger.debug('deleted nodes')
        labeled_terminals = self.__label_edge_by_its_terminal(
            labeled_terminals)
        # print "labeled edges + terminals"
        # print np.unique(labeled_terminals)
        # pe = ped.sed3(labeled_terminals)
        # pe.show()
        for i in range(np.min(labeled_terminals), 0):
            lti = labeled_terminals == i
            if np.sum(lti) < threshold:
                # delete small
                labeled_terminals[lti] = 0
                logger.debug('mazani %s %s' % (str(i), np.sum(lti)))
        # bring nodes back
        labeled_terminals[nodes] = 1
        return (labeled_terminals != 0).astype(np.int)

    def __generate_sklabel(self, skelet_nodes):

        sklabel_edg, len_edg = scipy.ndimage.label(
            skelet_nodes == 1, structure=np.ones([3, 3, 3]))
        sklabel_nod, len_nod = scipy.ndimage.label(
            skelet_nodes > 1, structure=np.ones([3, 3, 3]))

        sklabel = sklabel_edg - sklabel_nod

        return sklabel

    def __edge_vectors(self, edg_number, edg_stats):
        """
        | Return begin and end vector of edge.
        | run after __edge_curve()
        """
# this edge
        try:
            curve_params = edg_stats['curve_params']
            vectorA = self.__get_vector_from_curve(0.25, 0, curve_params)
            vectorB = self.__get_vector_from_curve(0.75, 1, curve_params)
        except:  # Exception as ex:
            logger.warning(traceback.format_exc())
            # print(ex)
            return {}

        return {'vectorA': vectorA.tolist(), 'vectorB': vectorB.tolist()}

    def __vectors_to_angle_deg(self, v1, v2):
        """
        Return angle of two vectors in degrees
        """
# get normalised vectors
        v1u = v1 / np.linalg.norm(v1)
        v2u = v2 / np.linalg.norm(v2)
        # print 'v1u ', v1u, ' v2u ', v2u

        angle = np.arccos(np.dot(v1u, v2u))
# special cases
        if np.isnan(angle):
            if (v1u == v2u).all():
                angle == 0
            else:
                angle == np.pi

        angle_deg = np.degrees(angle)

        # print 'angl ', angle, ' angl_deg ', angle_deg
        return angle_deg

    def __vector_of_connected_edge(self,
                                   edg_number,
                                   stats,
                                   edg_end,
                                   con_edg_order):
        """
        | find common node with connected edge and its vector

        | edg_end: Which end of edge you want (0 or 1)
        | con_edg_order: Which edge of selected end of edge you want (0,1)
        """
        if edg_end == 'A':
            connectedEdges = stats[edg_number]['connectedEdgesA']
            ndid = 'nodeIdA'
        elif edg_end == 'B':
            connectedEdges = stats[edg_number]['connectedEdgesB']
            ndid = 'nodeIdB'
        else:
            logger.error('Wrong edg_end in __vector_of_connected_edge()')

        connectedEdgeStats = stats[connectedEdges[con_edg_order]]
        # import pdb; pdb.set_trace()

        if stats[edg_number][ndid] == connectedEdgeStats['nodeIdA']:
            # sousední hrana u uzlu na konci 0 má stejný node na
            # svém konci 0 jako
            # nynější hrana
            vector = connectedEdgeStats['vectorA']
        elif stats[edg_number][ndid] == connectedEdgeStats['nodeIdB']:
            vector = connectedEdgeStats['vectorB']

        return vector

    def perpendicular_to_two_vects(self, v1, v2):
        # determinant
        a = (v1[1] * v2[2]) - (v1[2] * v2[1])
        b = -((v1[0] * v2[2]) - (v1[2] * v2[0]))
        c = (v1[0] * v2[1]) - (v1[1] * v2[0])
        return [a, b, c]

    def projection_of_vect_to_xy_plane(self, vect, xy1, xy2):
        """
        Return porojection of vect to xy plane given by vectprs xy1 and xy2
        """
        norm = self.perpendicular_to_two_vects(xy1, xy2)
        vect_proj = np.array(vect) - (
            np.dot(vect, norm) / np.linalg.norm(norm) ** 2) * np.array(norm)
        return vect_proj

    def __connected_edge_angle_on_one_end(self, edg_number, stats, edg_end):
        """
        | edg_number: integer with edg_number
        | stats: dictionary with all statistics and computations
        | edg_end: letter 'A' or 'B'
        |    creates phiXa, phiXb and phiXc.

        See Schwen2012 : Analysis and algorithmic generation of hepatic vascular
        system.
        """
        out = {}

        vector_key = 'vector' + edg_end
        try:
            vector = stats[edg_number][vector_key]
        except:  # Exception as e:
            logger.warning(traceback.print_exc())

        try:
            vectorX0 = self.__vector_of_connected_edge(
                edg_number, stats, edg_end, 0)
            phiXa = self.__vectors_to_angle_deg(vectorX0, vector)

            out.update({'phiA0' + edg_end + 'a': phiXa.tolist()})
        except:  # Exception as e:
            logger.warning(traceback.print_exc())
        try:
            vectorX1 = self.__vector_of_connected_edge(
                edg_number, stats, edg_end, 1)
        except:  # Exception as e:
            logger.warning(traceback.print_exc())

        try:

            vect_proj = self.projection_of_vect_to_xy_plane(
                vector, vectorX0, vectorX1)
            phiXa = self.__vectors_to_angle_deg(vectorX0, vectorX1)
            phiXb = self.__vectors_to_angle_deg(vector, vect_proj)
            vectorX01avg = \
                np.array(vectorX0 / np.linalg.norm(vectorX0)) +\
                np.array(vectorX1 / np.linalg.norm(vectorX1))
            phiXc = self.__vectors_to_angle_deg(vectorX01avg, vect_proj)

            out.update({
                'phi' + edg_end + 'a': phiXa.tolist(),
                'phi' + edg_end + 'b': phiXb.tolist(),
                'phi' + edg_end + 'c': phiXc.tolist()
            })

        except:  # Exception as e:
            logger.warning(traceback.print_exc())

        return out

    def __connected_edge_angle(self, edg_number, stats):
        """
        count angles betwen end vectors of edges
        """

# TODO tady je nějaký binec
        out = {}
        try:
            vectorA = stats[edg_number]['vectorA']
            # vectorB = stats[edg_number]['vectorB']
            stats[edg_number]['vectorB']
        except Exception:
            traceback.print_exc()
        try:
            vectorA0 = self.__vector_of_connected_edge(
                edg_number, stats, 'A', 0)
            # angleA0a = np.arccos(np.dot(vectorA, vectorA0))
            angleA0 = self.__vectors_to_angle_deg(vectorA, vectorA0)
            print 'va ', vectorA0, 'a0a', angleA0, 'a0', angleA0
            out.update({'angleA0': angleA0.tolist()})
        except Exception:
            traceback.print_exc()
            print (
                "connected edge (number " + str(edg_number) +
                ") vectorA not found 0 ")

        try:
            # vectorA1 = self.__vector_of_connected_edge(
            self.__vector_of_connected_edge(
                edg_number, stats, 'A', 1)
        except:
            print (
                "connected edge (number " + str(edg_number) +
                ") vectorA not found 1")

        out.update(
            self.__connected_edge_angle_on_one_end(edg_number, stats, 'A'))
        out.update(
            self.__connected_edge_angle_on_one_end(edg_number, stats, 'B'))
        angleA0 = 0
        return out

#        try:
# we need find end of edge connected to our node
# import pdb; pdb.set_trace()
# if stats[edg_number]['nodeIdA'] == stats[connectedEdgesA[0]]['nodeId0']:
# sousední hrana u uzlu na konci 0 má stejný node na svém
# konci 0 jako nynější hrana
#                vectorA0 = stats[connectedEdgesA[0]]['vectorA']
#            else:
#                vectorA0 = stats[connectedEdgesA[0]]['vectorB']
#        except:
#
# second neighbors on end "0"
# if  stats[edg_number]['nodeIdA'] == stats[connectedEdgesA[1]]['nodeId0']:
# sousední hrana u uzlu na konci 0 má
# stejný node na svém konci 0 jako nynější hrana
#                vectorA1 = stats[connectedEdgesA[1]]['vectorA']
#            else:
#                vectorA1 = stats[connectedEdgesA[1]]['vectorB']
#            print "ahoj"
#        except:
#            print ("Cannot compute angles for edge " + str(edg_number))
#
#        angle0 = np.arccos(np.dot(vectorA1, vectorA0))
#
#        return {'angle0':angle0}

    def __get_vector_from_curve(self, t0, t1, curve_params):
        return (np.array(curve_model(t1, curve_params)) -
                np.array(curve_model(t0, curve_params)))

    # def node_analysis(sklabel):
        # pass

    def __element_neighbors(self, el_number):
        """
        Gives array of element neighbors numbers (edges+nodes/terminals)

        | input:
        |   self.sklabel - original labeled data
        |   el_number - element label

        | uses/creates:
        |   self.shifted_sklabel - all labels shifted to positive numbers
        |   self.shifted_zero - value of original 0

        | returns:
        |   array of neighbor values
        |        - nodes for edge, edges for node
        |   element bounding box (with border)
        """
        # check if we have shifted sklabel, if not create it.
        try:
            self.shifted_zero
            self.shifted_sklabel
        except AttributeError:
            logger.debug('Generating shifted sklabel...')
            self.shifted_zero = abs(np.min(self.sklabel)) + 1
            self.shifted_sklabel = self.sklabel + self.shifted_zero

        el_number_shifted = el_number + self.shifted_zero

        BOUNDARY_PX = 5

        if el_number < 0:
            # cant have max_label<0
            box = scipy.ndimage.find_objects(
                self.shifted_sklabel, max_label=el_number_shifted)
        else:
            box = scipy.ndimage.find_objects(
                self.sklabel, max_label=el_number)
        box = box[len(box) - 1]

        d = max(0, box[0].start - BOUNDARY_PX)
        u = min(self.sklabel.shape[0], box[0].stop + BOUNDARY_PX)
        slice_z = slice(d, u)
        d = max(0, box[1].start - BOUNDARY_PX)
        u = min(self.sklabel.shape[1], box[1].stop + BOUNDARY_PX)
        slice_y = slice(d, u)
        d = max(0, box[2].start - BOUNDARY_PX)
        u = min(self.sklabel.shape[2], box[2].stop + BOUNDARY_PX)
        slice_x = slice(d, u)
        box = (slice_z, slice_y, slice_x)

        sklabelcr = self.sklabel[box]

        # element crop
        element = (sklabelcr == el_number)

        dilat_element = scipy.ndimage.morphology.binary_dilation(
            element,
            structure=np.ones([3, 3, 3])
        )

        neighborhood = sklabelcr * dilat_element

        neighbors = np.unique(neighborhood)
        neighbors = neighbors[neighbors != 0]
        neighbors = neighbors[neighbors != el_number]

        if el_number > 0:  # elnumber is edge
            neighbors = neighbors[neighbors < 0]  # return nodes
        elif el_number < 0:  # elnumber is node
            neighbors = neighbors[neighbors > 0]  # return edge
        else:
            logger.warning('Element is zero!!')
            neighbors = []

        return neighbors, box

    def __length_from_curve_spline(self, edg_stats, N=20):
        t = np.linspace(0.0, 1.0, N)
        x, y, z = scipy.interpolate.splev(
            t,
            edg_stats['curve_params']['fitParamsSpline']
        )
        # x = points[0, :]
        # y = points[1, :]
        # z = points[2, :]
        return self.__count_length(x, y, z, N)

    def __length_from_curve_poly(self, edg_stats, N=10):

        px = np.poly1d(edg_stats['curve_params']['fitParamsX'])
        py = np.poly1d(edg_stats['curve_params']['fitParamsY'])
        pz = np.poly1d(edg_stats['curve_params']['fitParamsZ'])

        t = np.linspace(0.0, 1.0, N)

        x = px(t)
        y = py(t)
        z = pz(t)

        return self.__count_length(x, y, z, N)

    def __count_length(self, x, y, z, N):
        length = 0
        for i in range(N - 1):
            # print i, ' ', t[i]
            p1 = np.asarray([
                x[i],
                y[i],
                z[i]
            ])
            p2 = np.asarray([
                x[i + 1],
                y[i + 1],
                z[i + 1]
            ])
            length += np.linalg.norm(p2 - p1)
            # print p1

        return length

    def __ordered_points_mm(self, points_mm, nodeA_pos, nodeB_pos,
                            one_node_mode=False):

        length = 0
        startpoint = nodeA_pos
        pt_mm = [[nodeA_pos[0]], [nodeA_pos[1]], [nodeA_pos[2]]]
        while len(points_mm[0]) != 0:
            # get closest point to startpoint
            p_length = float('Inf')  # get max length
            closest_num = -1
            for p in range(0, len(points_mm[0])):
                test_point = np.array(
                    [points_mm[0][p], points_mm[1][p], points_mm[2][p]])
                p_length_new = np.linalg.norm(startpoint - test_point)
                if p_length_new < p_length:
                    p_length = p_length_new
                    closest_num = p
            closest = np.array(
                [points_mm[0][closest_num],
                 points_mm[1][closest_num],
                 points_mm[2][closest_num]])
            # add length
            pt_mm[0].append(points_mm[0][closest_num])
            pt_mm[1].append(points_mm[1][closest_num])
            pt_mm[2].append(points_mm[2][closest_num])
            length += np.linalg.norm(closest - startpoint)
            # replace startpoint with used point
            startpoint = closest
            # remove used point from points
            points_mm = [
                np.delete(points_mm[0], closest_num),
                np.delete(points_mm[1], closest_num),
                np.delete(points_mm[2], closest_num)
            ]
        # add length to nodeB
        if not one_node_mode:
            length += np.linalg.norm(nodeB_pos - startpoint)
            pt_mm[0].append(nodeB_pos[0])
            pt_mm[1].append(nodeB_pos[1])
            pt_mm[2].append(nodeB_pos[2])

        return pt_mm, length

    def __edge_length(self, edg_number, edg_stats):
        """
        Computes estimated length of edge, distance from end nodes and
        tortosity.

        | needs:
        |   edg_stats['nodeIdA']
        |   edg_stats['nodeIdB']
        |   edg_stats['nodeA_ZYX']
        |   edg_stats['nodeB_ZYX']

        | output:
        |    'lengthEstimation'  - Estimated length of edge
        |    'nodesDistance'     - Distance between connected nodes
        |    'tortuosity'        - Tortuosity
        """
        # test for needed data
        try:
            edg_stats['nodeIdA']
            edg_stats['nodeA_ZYX']
        except:
            hasNodeA = False
        else:
            hasNodeA = True

        try:
            edg_stats['nodeIdB']
            edg_stats['nodeB_ZYX']
        except:
            hasNodeB = False
        else:
            hasNodeB = True

        if (not hasNodeA) and (not hasNodeB):
            logger.warning(
                '__edge_length doesnt have needed data!!! Using unreliable' +
                'method.')
            length = float(
                np.sum(
                    self.sklabel[self.elm_box[edg_number]] == edg_number) + 2)
            medium_voxel_length = (
                self.voxelsize_mm[0] + self.voxelsize_mm[1] +
                self.voxelsize_mm[2]) / 3.0
            length = length * medium_voxel_length

            stats = {
                'lengthEstimation': float(length),
                'nodesDistance': None,
                'tortuosity': 1
            }

            return stats

        # crop used area
        box = self.elm_box[edg_number]
        sklabelcr = self.sklabel[box]

        # get absolute position of nodes
        if hasNodeA and not hasNodeB:
            logger.warning(
                '__edge_length has only one node!!! using one node mode.')
            nodeA_pos_abs = edg_stats['nodeA_ZYX']
            one_node_mode = True
        elif hasNodeB and not hasNodeA:
            logger.warning(
                '__edge_length has only one node!!! using one node mode.')
            nodeA_pos_abs = edg_stats['nodeB_ZYX']
            one_node_mode = True
        else:
            nodeA_pos_abs = edg_stats['nodeA_ZYX']
            nodeB_pos_abs = edg_stats['nodeB_ZYX']
            one_node_mode = False

        # get realtive position of nodes [Z,Y,X]
        nodeA_pos = np.array(
            [nodeA_pos_abs[0] - box[0].start,
             nodeA_pos_abs[1] - box[1].start,
             nodeA_pos_abs[2] - box[2].start])
        if not one_node_mode:
            nodeB_pos = np.array(
                [nodeB_pos_abs[0] - box[0].start,
                 nodeB_pos_abs[1] - box[1].start,
                 nodeB_pos_abs[2] - box[2].start])
        # get position in mm
        nodeA_pos = nodeA_pos * self.voxelsize_mm
        if not one_node_mode:
            nodeB_pos = nodeB_pos * self.voxelsize_mm
        else:
            nodeB_pos = None

        # get positions of edge points
        points = (sklabelcr == edg_number).nonzero()
        points_mm = [
            np.array(points[0] * self.voxelsize_mm[0]),
            np.array(points[1] * self.voxelsize_mm[1]),
            np.array(points[2] * self.voxelsize_mm[2])
        ]

        _, length_pixel = self.__ordered_points_mm(
            points_mm, nodeA_pos, nodeB_pos, one_node_mode)
        length_pixel = float(length_pixel)
        length = length_pixel
        length_poly = None
        length_spline = None
        if not one_node_mode:

            try:
                length_poly = self.__length_from_curve_poly(edg_stats)
                length_spline = self.__length_from_curve_spline(edg_stats)
                length = length_spline
            except:
                logger.info('problem with length_poly or length_spline')

        # get distance between nodes
        if one_node_mode:
            startpoint = np.array([
                points_mm[0][0],
                points_mm[1][0],
                points_mm[2][0]
            ])
            nodes_distance = np.linalg.norm(nodeA_pos - startpoint)
        else:
            nodes_distance = np.linalg.norm(nodeA_pos - nodeB_pos)

        stats = {
            'lengthEstimationPoly': length_poly,
            'lengthEstimationSpline': length_spline,
            'lengthEstimation': length,
            'lengthEstimationPixel': length_pixel,
            'nodesDistance': float(nodes_distance),
            'tortuosity': float(length / float(nodes_distance))
        }

        return stats

    def __edge_curve(self,  edg_number, edg_stats):
        """
        Return params of curve and its starts and ends locations

        | needs:
        |    edg_stats['nodeA_ZYX_mm']
        |    edg_stats['nodeB_ZYX_mm']
        """
        retval = {}
        try:

            # crop used area
            box = self.elm_box[edg_number]

            sklabelcr = self.sklabel[box]
            # get positions of edge points
            points = (sklabelcr == edg_number).nonzero()
            points_mm = [
                np.array((box[0].start + points[0]) * self.voxelsize_mm[0]),
                np.array((box[1].start + points[1]) * self.voxelsize_mm[1]),
                np.array((box[2].start + points[2]) * self.voxelsize_mm[2])
            ]
            point0_mm = np.array(edg_stats['nodeA_ZYX_mm'])
            # if 'nodeB_ZYX_mm' in edg_stats.keys():
            #     point1_mm = np.array(edg_stats['nodeB_ZYX_mm'])
            #     one_node_mode = False
            # else:
            #     point1_mm = None
            #     one_node_mode = True
            point1_mm = np.array(edg_stats['nodeB_ZYX_mm'])
            pts_mm_ord, _ = self.__ordered_points_mm(
                points_mm, point0_mm, point1_mm)

            t = np.linspace(0.0, 1.0, len(pts_mm_ord[0]))
            fitParamsX = np.polyfit(t, pts_mm_ord[0], self.curve_order)
            fitParamsY = np.polyfit(t, pts_mm_ord[1], self.curve_order)
            fitParamsZ = np.polyfit(t, pts_mm_ord[2], self.curve_order)
            # Spline
            # s - smoothing
            # w - weight
            w = np.ones(len(pts_mm_ord[0]))
            # first and last have big weight
            w[1] = len(pts_mm_ord[0])
            w[-1] = len(pts_mm_ord[0])
            tck, u = scipy.interpolate.splprep(
                pts_mm_ord, s=self.spline_smoothing)
            # tckl = np.asarray(tck).tolist()

            retval = {'curve_params':
                      {
                          'start': list(point0_mm.tolist()),
                          'vector': list((point1_mm - point0_mm).tolist()),
                          'fitParamsX': list(fitParamsX.tolist()),
                          'fitParamsY': list(fitParamsY.tolist()),
                          'fitParamsZ': list(fitParamsZ.tolist()),
                          'fitCurveStrX': str(np.poly1d(fitParamsX)),
                          'fitCurveStrY': str(np.poly1d(fitParamsY)),
                          'fitCurveStrZ': str(np.poly1d(fitParamsZ)),
                          'fitParamsSpline': tck
                      }}

        except Exception as ex:
            logger.warning("Problem in __edge_curve()")
            logger.warning(traceback.format_exc())
            print (ex)

        return retval

    # def edge_analysis(sklabel, edg_number):
        # print 'element_analysis'
# element dilate * sklabel[sklabel < 0]
        # pass
    def __radius_analysis_init(self):
        """
        Computes skeleton with distances from edge of volume.

        | sklabel: skeleton or labeled skeleton
        | volume_data: volumetric data with zeros and ones
        """
        uq = np.unique(self.volume_data)
        if (uq[0] == 0) & (uq[1] == 1):
            dst = scipy.ndimage.morphology.distance_transform_edt(
                self.volume_data,
                sampling=self.voxelsize_mm
            )

            # import ipdb; ipdb.set_trace() # BREAKPOINT
            dst = dst * (self.sklabel != 0)

            return dst

        else:
            print "__radius_analysis_init() error.  "
            return None

    def __radius_analysis(self, edg_number, skdst):
        """
        Return smaller radius of tube
        """
        # returns mean distance from skeleton to vessel border = vessel radius
        edg_skdst = skdst * (self.sklabel == edg_number)
        return np.mean(edg_skdst[edg_skdst != 0])

    def __connection_analysis(self, edg_number):
        """
        Analysis of which edge is connected
        """
        edg_neigh = self.elm_neigh[edg_number]

        if len(edg_neigh) == 1:
            logger.warning('Only one (' + str(edg_neigh) +
                           ') connected node in connection_analysis()' +
                           ' for edge number ' + str(edg_number))

            # get edges connected to end nodes
            connectedEdgesA = np.array(self.elm_neigh[edg_neigh[0]])
            # remove edg_number from connectedEdges list
            connectedEdgesA = connectedEdgesA[connectedEdgesA != edg_number]

            # get pixel and mm position of end nodes
            # node A
            box0 = self.elm_box[edg_neigh[0]]
            nd00, nd01, nd02 = (edg_neigh[0] == self.sklabel[box0]).nonzero()
            point0_mean = [np.mean(nd00), np.mean(nd01), np.mean(nd02)]
            point0 = np.array([float(point0_mean[0] + box0[0].start), float(
                point0_mean[1] + box0[1].start),
                float(point0_mean[2] + box0[2].start)])

            # node position -> mm
            point0_mm = point0 * self.voxelsize_mm

            edg_stats = {
                'id': edg_number,
                'nodeIdA': int(edg_neigh[0]),
                'connectedEdgesA': connectedEdgesA.tolist(),
                'nodeA_ZYX': point0.tolist(),
                'nodeA_ZYX_mm': point0_mm.tolist()
            }

        elif len(edg_neigh) != 2:
            logger.warning('Wrong number (' + str(edg_neigh) +
                           ') of connected nodes in connection_analysis()' +
                           ' for edge number ' + str(edg_number))
            edg_stats = {
                'id': edg_number
            }
        else:
            # get edges connected to end nodes
            connectedEdgesA = np.array(self.elm_neigh[edg_neigh[0]])
            connectedEdgesB = np.array(self.elm_neigh[edg_neigh[1]])
            # remove edg_number from connectedEdges list
            connectedEdgesA = connectedEdgesA[connectedEdgesA != edg_number]
            connectedEdgesB = connectedEdgesB[connectedEdgesB != edg_number]

            # get pixel and mm position of end nodes
            # node A
            box0 = self.elm_box[edg_neigh[0]]
            nd00, nd01, nd02 = (edg_neigh[0] == self.sklabel[box0]).nonzero()
            point0_mean = [np.mean(nd00), np.mean(nd01), np.mean(nd02)]
            point0 = np.array([float(point0_mean[0] + box0[0].start), float(
                point0_mean[1] + box0[1].start), float(point0_mean[2] +
                                                       box0[2].start)])
            # node B
            box1 = self.elm_box[edg_neigh[1]]
            nd10, nd11, nd12 = (edg_neigh[1] == self.sklabel[box1]).nonzero()
            point1_mean = [np.mean(nd10), np.mean(nd11), np.mean(nd12)]
            point1 = np.array([float(point1_mean[0] + box1[0].start), float(
                point1_mean[1] + box1[1].start), float(point1_mean[2] +
                                                       box1[2].start)])
            # node position -> mm
            point0_mm = point0 * self.voxelsize_mm
            point1_mm = point1 * self.voxelsize_mm

            edg_stats = {
                'id': edg_number,
                'nodeIdA': int(edg_neigh[0]),
                'nodeIdB': int(edg_neigh[1]),
                'connectedEdgesA': connectedEdgesA.tolist(),
                'connectedEdgesB': connectedEdgesB.tolist(),
                'nodeA_ZYX': point0.tolist(),
                'nodeB_ZYX': point1.tolist(),
                'nodeA_ZYX_mm': point0_mm.tolist(),
                'nodeB_ZYX_mm': point1_mm.tolist()
            }

        return edg_stats

def generate_binary_elipsoid(ndradius=[1, 1, 1]):
    """
    generate binary elipsoid shape
    """
    ndradius = np.asarray(ndradius).astype(np.double)
    shape = (ndradius * 2) + 1
    x, y, z = np.indices(shape)
    center1 = ndradius
    mask = (
        ((x - ndradius[0])**2 ) / ndradius[0]**2 +
        ((y - ndradius[1])**2 ) / ndradius[1]**2 +
        ((z - ndradius[2])**2 ) / ndradius[2]**2
        )
    # (y - ndradius[1])**2 < radius1**2
    # mask = mask radius1**1
    return mask < 1




def curve_model(t, params):
    p0 = params['start'][0] + t * params['vector'][0]
    p1 = params['start'][1] + t * params['vector'][1]
    p2 = params['start'][2] + t * params['vector'][2]
    return [p0, p1, p2]
