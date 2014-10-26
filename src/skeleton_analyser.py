#! /usr/bin/python
# -*- coding: utf-8 -*-

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
    Example:
    skan = SkeletonAnalyser(data3d_skel, volume_data, voxelsize_mm)
    stats = skan.skeleton_analysis()
    """

    def __init__(self, data3d_skel, volume_data=None, voxelsize_mm=[1, 1, 1]):
        self.volume_data = volume_data
        self.voxelsize_mm = voxelsize_mm

        # get array with 1 for edge, 2 is node and 3 is terminal
        logger.debug('Generating sklabel...')
        data3d_skel = self.__filter_small(data3d_skel)
        skelet_nodes = self.__skeleton_nodes(data3d_skel)
        self.sklabel = self.__generate_sklabel(skelet_nodes)
        self.cut_wrong_skeleton = False

        logger.debug('Inited SkeletonAnalyser - voxelsize:' + str(
            voxelsize_mm) + ' volumedata:' + str(volume_data is not None))

    def skeleton_analysis(self, guiUpdateFunction=None):
        """
        Glossary:
        element: line structure of skeleton connected to node on both ends
        node: connection point of elements. It is one or few voxelsize_mm
        terminal: terminal node
        """
        def updateFunction(num, length, part):
            if (num % 5 == 0) or num == length:
                logger.info('skeleton_analysis: processed ' + str(
                    num) + '/' + str(length) + ', part ' + str(part))
            if guiUpdateFunction is not None:
                guiUpdateFunction(num, length, part)

        logger.debug('__radius_analysis_init')
        if self.volume_data is not None:
            skdst = self.__radius_analysis_init()

        stats = {}
        len_edg = np.max(self.sklabel)
        len_node = np.min(self.sklabel)
        logger.debug(
            'len_edg: ' + str(len_edg) + ' len_node: ' + str(len_node))

        logger.debug(
            'skeleton_analysis: starting element_neighbors processing')
        self.elm_neigh = {}
        self.elm_box = {}
        for edg_number in (range(len_node, 0) + range(1, len_edg + 1)):
            self.elm_neigh[edg_number], self.elm_box[
                edg_number] = self.__element_neighbors(edg_number)
            updateFunction(edg_number + abs(len_node) + 1, abs(
                len_node) + len_edg + 1, 0)  # update gui progress
        logger.debug(
            'skeleton_analysis: finished element_neighbors processing')

        logger.debug('skeleton_analysis: starting first processing part')
        for edg_number in range(1, len_edg + 1):
            edgst = {}
            edgst.update(self.__connection_analysis(edg_number))
            edgst.update(self.__edge_length(edg_number, edgst))
            edgst.update(self.__edge_curve(edg_number, edgst))
            edgst.update(self.__edge_vectors(edg_number, edgst))
            # edgst = edge_analysis(sklabel, i)
            if self.volume_data is not None:
                edgst['radius_mm'] = float(self.__radius_analysis(
                    edg_number, skdst))  # slow (this takes most of time)
            stats[edgst['id']] = edgst

            updateFunction(edg_number, len_edg, 1)  # update gui progress
        logger.debug('skeleton_analysis: finished first processing part')

        if self.cut_wrong_skeleton:
            stats = self.__cut_wrong_skeleton(stats)

        # @TODO dokončit
        # logger.debug('skeleton_analysis: starting second processing part')
        # for edg_number in range (1,len_edg+1):
            # edgst = stats[edg_number]
            # edgst.update(self.__connected_edge_angle(edg_number, stats))

            # updateFunction(edg_number,len_edg,2) # update gui progress
        # logger.debug('skeleton_analysis: finished second processing part')

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

    def __cut_wrong_skeleton(self, stats, cut_ratio=2.0):
        """
        cut_ratio = 2.0 -> if radius of vessel is 2x its lenght or more,
        remove it
        """
        # remove edges that are not connected to rest of the skeleton
        logger.debug(
            'skeleton_analysis: Cut - Removing edges that are not connected' +
            ' to rest of skeleton')
        for s in dict(stats):
            connected_edgs = []
            try:
                connected_edgs += stats[s]['connectedEdgesA']
            except Exception, e:
                stats[s]['connectedEdgesA'] = []

            try:
                connected_edgs += stats[s]['connectedEdgesB']
            except Exception, e:
                stats[s]['connectedEdgesB'] = []

            if len(connected_edgs) == 0:
                logger.debug(
                    'Edge id:' + str(s) + ' is not connected to rest of' +
                    ' skeleton')
                stats = self.__remove_edge_from_stats(stats, s)
        logger.debug('skeleton_analysis: Cut - Unconnected edges removed')

        logger.debug(
            'skeleton_analysis: Cut - Getting lists of nodes,' +
            ' edges and end nodes')
        # dict of connected nodes to edge
        edg_neigh = {}
        for s in dict(stats):
            if not s in edg_neigh:
                edg_neigh[s] = []

            try:
                if not stats[s]['nodeIdA'] in edg_neigh[s]:
                    edg_neigh[s].append(stats[s]['nodeIdA'])
            except KeyError:
                logger.debug('Edge id:' + str(s) + ' is missing nodeIdA')

            try:
                if not stats[s]['nodeIdB'] in edg_neigh[s]:
                    edg_neigh[s].append(stats[s]['nodeIdB'])
            except KeyError:
                logger.debug('Edge id:' + str(s) + ' is missing nodeIdB')

        # dict of connected edges to node
        node_neigh = {}
        for e in edg_neigh:
            for node in edg_neigh[e]:
                if not node in node_neigh:
                    node_neigh[node] = []

                if not e in node_neigh[node]:
                    node_neigh[node].append(e)

        # get end nodes
        end_nodes = {}
        for i in node_neigh:
            neigh = node_neigh[i]
            if len(neigh) == 1:
                logger.debug('end node/edge: ' + str(i) + '/' + str(neigh[0]))
                end_nodes[i] = neigh[0]
        logger.debug('skeleton_analysis: Cut - Got all lists')

        # removes end edges based on radius/length ratio
        logger.debug(
            'skeleton_analysis: Cut - Removing bad segments based on' +
            ' radius/length ratio')
        for end_node in end_nodes:
            edge_end_id = end_nodes[end_node]
            edge_end_stat = stats[edge_end_id]

            if edge_end_stat['radius_mm'] /\
                    float(edge_end_stat['lengthEstimation']) > cut_ratio:
                logger.debug(
                    'bad rad/len: ' +
                    str(edge_end_stat['radius_mm']) + ' ' +
                    str(edge_end_stat['lengthEstimation']))
                stats = self.__remove_edge_from_stats(stats, edge_end_id)

        logger.debug('skeleton_analysis: Cut - Bad segments removed')

        return stats

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

        return data3d_skel

    def __filter_small(self, skel, threshold = 3):
        """
        terminals are connected to edges
        """
        skeleton_nodes = self.__skeleton_nodes(skel)
        skeleton_nodes[skeleton_nodes==3] == 1
        sklabel = self.__generate_sklabel(skeleton_nodes)
        for i in range(1, np.max(sklabel)):
            if np.sum(sklabel == i) < threshold:
# delete small
                sklabel[sklabel == i] == 0
                print "mazani "
        return (sklabel != 0).astype(np.int)


    def __generate_sklabel(self, skelet_nodes):

        sklabel_edg, len_edg = scipy.ndimage.label(
            skelet_nodes == 1, structure=np.ones([3, 3, 3]))
        sklabel_nod, len_nod = scipy.ndimage.label(
            skelet_nodes > 1, structure=np.ones([3, 3, 3]))

        sklabel = sklabel_edg - sklabel_nod

        return sklabel

    def __edge_vectors(self, edg_number, edg_stats):
        """
        Return begin and end vector of edge.
        run after __edge_curve()
        """
# this edge
        try:
            curve_params = edg_stats['curve_params']
            vectorA = self.__get_vector_from_curve(0.25, 0, curve_params)
            vectorB = self.__get_vector_from_curve(0.75, 1, curve_params)
        except Exception as ex:
            print (ex)
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
        find common node with connected edge and its vector
        edg_end: Which end of edge you want (0 or 1)
        con_edg_order: Which edge of selected end of edge you want (0,1)
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
        edg_number: integer with edg_number
        stats: dictionary with all statistics and computations
        edg_end: letter 'A' or 'B'
        creates phiXa, phiXb and phiXc.
        See Schwen2012 : Analysis and algorithmic generation of hepatic
            vascular system.
        """
        out = {}

        vector_key = 'vector' + edg_end
        try:
            vector = stats[edg_number][vector_key]
        except Exception as e:
            print (e)
            # traceback.print_exc()

        try:
            vectorX0 = self.__vector_of_connected_edge(
                edg_number, stats, edg_end, 0)
            phiXa = self.__vectors_to_angle_deg(vectorX0, vector)

            out.update({'phiA0' + edg_end + 'a': phiXa.tolist()})
        except Exception as e:
            print (e)
        try:
            vectorX1 = self.__vector_of_connected_edge(
                edg_number, stats, edg_end, 1)
        except Exception as e:
            print (e)

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

        except Exception as e:
            print (e)

        return out

    def __connected_edge_angle(self, edg_number, stats):
        """
        count angles betwen end vectors of edges
        """

# TODO tady je nějaký binec
        out = {}
        try:
            vectorA = stats[edg_number]['vectorA']
            vectorB = stats[edg_number]['vectorB']
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
            vectorA1 = self.__vector_of_connected_edge(
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
        input:
            self.sklabel - original labeled data
            el_number - element label

        uses/creates:
            self.shifted_sklabel - all labels shifted to positive numbers
            self.shifted_zero - value of original 0

        returns:
            array of neighbor values
                - nodes for edge, edges for node
            element bounding box (with border)
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

    def __edge_length(self, edg_number, edg_stats):
        """
        Computes estimated length of edge, distance from end nodes and
        tortosity.
        needs:
            edg_stats['nodeIdA']
            edg_stats['nodeIdB']
            edg_stats['nodeA_ZYX']
            edg_stats['nodeB_ZYX']
        output:
            'lengthEstimation'  - Estimated length of edge
            'nodesDistance'     - Distance between connected nodes
            'tortuosity'        - Tortuosity
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

        # get positions of edge points
        points = (sklabelcr == edg_number).nonzero()
        points_mm = [
            np.array(points[0] * self.voxelsize_mm[0]),
            np.array(points[1] * self.voxelsize_mm[1]),
            np.array(points[2] * self.voxelsize_mm[2])
        ]

        length = 0
        startpoint = nodeA_pos
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
            length += np.linalg.norm(closest - startpoint)
            # replace startpoint with used point
            startpoint = closest
            # remove used point from points
            points_mm = [np.delete(points_mm[0], closest_num), np.delete(
                points_mm[1], closest_num), np.delete(
                    points_mm[2], closest_num)]

        # add length to nodeB
        if not one_node_mode:
            length += np.linalg.norm(nodeB_pos - startpoint)

        # get distance between nodes
        if one_node_mode:
            nodes_distance = np.linalg.norm(nodeA_pos - startpoint)
        else:
            nodes_distance = np.linalg.norm(nodeA_pos - nodeB_pos)

        stats = {
            'lengthEstimation': float(length),
            'nodesDistance': float(nodes_distance),
            'tortuosity': float(length / nodes_distance)
        }

        return stats

    def __edge_curve(self,  edg_number, edg_stats):
        """
        Return params of curve and its starts and ends locations
        needs:
            edg_stats['nodeA_ZYX_mm']
            edg_stats['nodeB_ZYX_mm']
        """
        retval = {}
        try:

            point0_mm = np.array(edg_stats['nodeA_ZYX_mm'])
            point1_mm = np.array(edg_stats['nodeB_ZYX_mm'])
            retval = {'curve_params':
                      {
                          'start': point0_mm.tolist(),
                          'vector': (point1_mm - point0_mm).tolist()
                      }}
        except Exception as ex:
            logger.warning("Problem in __edge_curve()")
            print (ex)

        return retval

    # def edge_analysis(sklabel, edg_number):
        # print 'element_analysis'
# element dilate * sklabel[sklabel < 0]
        # pass
    def __radius_analysis_init(self):
        """
        Computes skeleton with distances from edge of volume.
        sklabel: skeleton or labeled skeleton
        volume_data: volumetric data with zeros and ones
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
        return smaller radius of tube
        """
        # import ipdb; ipdb.set_trace() # BREAKPOINT
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
            connectedEdgesA = self.elm_neigh[edg_neigh[0]]
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
            connectedEdgesA = self.elm_neigh[edg_neigh[0]]
            connectedEdgesB = self.elm_neigh[edg_neigh[1]]
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


def curve_model(t, params):
    p0 = params['start'][0] + t * params['vector'][0]
    p1 = params['start'][1] + t * params['vector'][1]
    p2 = params['start'][2] + t * params['vector'][2]
    return [p0, p1, p2]
