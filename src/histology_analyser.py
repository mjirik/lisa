#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
python src/histology_analyser.py -i ~/data/medical/data_orig/jatra_mikro_data/Nejlepsi_rozliseni_nevycistene -t 6800 -cr 0 -1 100 300 100 300

"""

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))

import logging
logger = logging.getLogger(__name__)

import argparse

from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
     QFont, QPixmap, QComboBox

import numpy as np
import scipy.ndimage
import misc
import datareader
#import SimpleITK as sitk
import scipy.ndimage
from PyQt4.QtGui import QApplication
import csv

import sys
import traceback


import seed_editor_qt as seqt
import skelet3d
import segmentation
import misc
import py3DSeedEditor as se

from seed_editor_qt import QTSeedEditor

GAUSSIAN_SIGMA = 1
fast_debug = False
#fast_debug = True

import histology_analyser_gui as HA_GUI

class HistologyAnalyser:
    def __init__(self, data3d, metadata, threshold, nogui=True):
        self.data3d = data3d
        self.threshold = threshold
        self.nogui = nogui

        print metadata
        if 'voxelsize_mm' not in metadata.keys():
# @TODO resolve problem with voxelsize
            metadata['voxelsize_mm'] = [0.1, 0.2, 0.3]

        self.metadata = metadata


    def remove_area(self):
        if not self.nogui:
            app = QApplication(sys.argv)
            pyed = QTSeedEditor(
                self.data3d, mode='mask'
            )
            pyed.exec_()

    def data_to_binar(self):
        data3d_thr = segmentation.vesselSegmentation(
            self.data3d,
            segmentation=np.ones(self.data3d.shape, dtype='int8'),
            #segmentation=oseg.orig_scale_segmentation,
            threshold=self.threshold, #-1,
            inputSigma=0.15,
            dilationIterations=2,
            nObj=1,
            biggestObjects=False,
            #dataFiltering = True,
            interactivity= not self.nogui,
            binaryClosingIterations=5,
            binaryOpeningIterations=1)
        return data3d_thr

    def binar_to_skeleton(self, data3d_thr):
        data3d_thr = (data3d_thr > 0).astype(np.int8)
        data3d_skel = skelet3d.skelet3d(data3d_thr)
        return data3d_skel

    def data_to_skeleton(self):
        data3d_thr = self.data_to_binar()
        data3d_skel = self.binar_to_skeleton(data3d_thr)
        return data3d_thr, data3d_skel
    
    def skeleton_to_statistics(self, data3d_thr, data3d_skel):
        skan = SkeletonAnalyser(
            data3d_skel,
            volume_data=data3d_thr,
            voxelsize_mm=self.metadata['voxelsize_mm'])

        stats = skan.skeleton_analysis()
        self.sklabel = skan.sklabel
        #data3d_nodes[data3d_nodes==3] = 2
        self.stats = {'Graph':stats}
        
    def showSegmentedData(self, data3d_thr, data3d_skel):
        skan = SkeletonAnalyser(
            data3d_skel,
            volume_data=data3d_thr,
            voxelsize_mm=self.metadata['voxelsize_mm'])
        data3d_nodes_vis = skan.sklabel.copy()
# edges
        data3d_nodes_vis[data3d_nodes_vis > 0] = 1
# nodes and terminals
        data3d_nodes_vis[data3d_nodes_vis < 0] = 2

        #pyed = seqt.QTSeedEditor(
        #    data3d,
        #    seeds=(data3d_nodes_vis).astype(np.int8),
        #    contours=data3d_thr.astype(np.int8)
        #)
        #app.exec_()
        if not self.nogui:
            pyed = se.py3DSeedEditor(
                self.data3d,
                seeds=(data3d_nodes_vis).astype(np.int8),
                contour=data3d_thr.astype(np.int8)
            )
            pyed.show()

    def run(self):
        #self.preprocessing()
        app = QApplication(sys.argv)
        if not fast_debug:
            data3d_thr = self.data_to_binar()

            #self.data3d_thri = self.muxImage(
            #        self.data3d_thr2.astype(np.uint16),
            #        metadata
            #        )
            #sitk.Show(self.data3d_thri)

            #self.data3di = self.muxImage(
            #        self.data3d.astype(np.uint16),
            #        metadata
            #        )
            #sitk.Show(self.data3di)


            #app.exec_()
            data3d_skel = self.binar_to_skeleton(data3d_thr)

            print "skelet"

        # pyed = seqt.QTSeedEditor(
        #         data3d,
        #         contours=data3d_thr.astype(np.int8),
        #         seeds=data3d_skel.astype(np.int8)
        #         )
            #app.exec_()
        else:
            struct = misc.obj_from_file(filename='tmp0.pkl', filetype='pickle')
            data3d_skel = struct['sk']
            data3d_thr = struct['thr']

        self.skeleton_to_statistics(data3d_skel)



       # import pdb; pdb.set_trace()
    def preprocessing(self):
        self.data3d = scipy.ndimage.filters.gaussian_filter(
                self.data3d,
                GAUSSIAN_SIGMA
                )
        self.data3d_thr = self.data3d > self.threshold

        self.data3d_thr2 = scipy.ndimage.morphology.binary_opening(
                self.data3d_thr
                )
        #gf = sitk.SmoothingRecursiveGaussianImageFilter()
        #gf.SetSigma(5)
        #gf = sitk.DiscreteGaussianImageFilter()
        #gf.SetVariance(1.0)
        #self.data3di2 = gf.Execute(self.data3di)#, 5)

        pass



    def muxImage(self, data3d, metadata):
        data3di = sitk.GetImageFromArray(data3d)
        data3di.SetSpacing(metadata['voxelsize_mm'])

        return data3di



    def writeStatsToYAML(self, filename='hist_stats.yaml'):
        print 'write to yaml'
        misc.obj_to_file(self.stats, filename=filename, filetype='yaml')

        #sitk.
    def writeStatsToCSV(self, filename='hist_stats.csv'):
        data = self.stats['Graph']

        with open(filename, 'wb') as csvfile:
            writer = csv.writer(
                    csvfile,
                    delimiter=';',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL
                    )

            for lineid in data:
                dataline = data[lineid]
                writer.writerow(self.__dataToCSVLine(dataline))
                #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])


    def writeSkeletonToPickle(self, filename='skel.pkl'):
        misc.obj_to_file(self.sklabel, filename=filename, filetype='pickle')


    def __dataToCSVLine(self, dataline):
        arr = []
# @TODO arr.append
        try:
            arr = [
                    dataline['id'],
                    dataline['nodeIdA'],
                    dataline['nodeIdB'],
                    dataline['radius'],
                    dataline['lengthEstimation']
                    ]
        except:
            arr = []

        return arr



    def show(self):
        app = QApplication(sys.argv)
        seqt.QTSeedEditor(self.output.astype(np.int16))
        app.exec_()


def curve_model(t, params):
    p0 = params['start'][0] + t*params['vector'][0]
    p1 = params['start'][1] + t*params['vector'][1]
    p2 = params['start'][2] + t*params['vector'][2]
    return [p0, p1, p2]


class SkeletonAnalyser:
    """
    Example:
    skan = SkeletonAnalyser(data3d_skel, volume_data, voxelsize_mm)
    stats = skan.skeleton_analysis()
    """

    def __init__(self, data3d_skel, volume_data=None, voxelsize_mm=[1,1,1]):
        self.volume_data = volume_data
        self.voxelsize_mm = voxelsize_mm
        # get array with 1 for edge, 2 is node and 3 is terminal
        skelet_nodes = self.__skeleton_nodes(data3d_skel, self.volume_data)
        self.__generate_sklabel(skelet_nodes)


    def skeleton_analysis(self):
        """
        Glossary:
        element: line structure of skeleton connected to node on both ends
        node: connection point of elements. It is one or few voxelsize_mm
        terminal: terminal node


        """
        if not fast_debug:
            if self.volume_data is not None:
                skdst = self.__radius_analysis_init()

            stats = {}
            len_edg = np.max(self.sklabel)
            #len_edg = 30

            for edg_number in range (1,len_edg):
                edgst = self.__connection_analysis(edg_number)
                edgst.update(self.__edge_length(edg_number))
                edgst.update(self.__edge_curve(edg_number, edgst, self.voxelsize_mm))
                edgst.update(self.__edge_vectors(edg_number, edgst))
                #edgst = edge_analysis(sklabel, i)
                if self.volume_data is not None:
                    edgst['radius_mm'] = float(self.__radius_analysis(edg_number,skdst))
                stats[edgst['id']] = edgst

#save data for faster debug
            struct = {'sVD':self.volume_data, 'stats':stats, 'len_edg':len_edg}
            misc.obj_to_file(struct, filename='tmp.pkl', filetype='pickle')
        else:
            struct = misc.obj_from_file(filename='tmp.pkl', filetype='pickle')
            self.volume_data = struct['sVD']
            stats = struct['stats']
            len_edg = struct['len_edg']


        #@TODO dokončit
        for edg_number in range (1,len_edg):
            edgst = stats[edg_number]
            edgst.update(self.__connected_edge_angle(edg_number, stats))




        return stats
        #import pdb; pdb.set_trace()

    def __generate_sklabel(self, skelet_nodes):

        sklabel_edg, len_edg = scipy.ndimage.label(skelet_nodes == 1, structure=np.ones([3,3,3]))
        sklabel_nod, len_nod = scipy.ndimage.label(skelet_nodes > 1, structure=np.ones([3,3,3]))

        self.sklabel = sklabel_edg - sklabel_nod


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


        return {'vectorA':vectorA.tolist(), 'vectorB': vectorB.tolist()}

    def __vectors_to_angle_deg(self, v1, v2):
        """
        Return angle of two vectors in degrees
        """
# get normalised vectors
        v1u = v1/np.linalg.norm(v1)
        v2u = v2/np.linalg.norm(v2)
        #print 'v1u ', v1u, ' v2u ', v2u

        angle = np.arccos(np.dot(v1u, v2u))
# special cases
        if np.isnan(angle):
            if (v1u == v2u).all():
                angle == 0
            else:
                angle == np.pi

        angle_deg = np.degrees(angle)

        #print 'angl ', angle, ' angl_deg ', angle_deg
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
        elif edg_end == 'B' :
            connectedEdges = stats[edg_number]['connectedEdgesB']
            ndid = 'nodeIdB'
        else:
            logger.error ('Wrong edg_end in __vector_of_connected_edge()')


        connectedEdgeStats = stats[connectedEdges[con_edg_order]]
        #import pdb; pdb.set_trace()

        if stats[edg_number][ndid] == connectedEdgeStats['nodeIdA']:
# sousední hrana u uzlu na konci 0 má stejný node na svém konci 0 jako nynější hrana
            vector = connectedEdgeStats['vectorA']
        elif stats[edg_number][ndid] == connectedEdgeStats['nodeIdB']:
            vector = connectedEdgeStats['vectorB']


        return vector

    def perpendicular_to_two_vects(self, v1, v2):
#determinant
        a = (v1[1]*v2[2]) - (v1[2]*v2[1])
        b = -((v1[0]*v2[2]) - (v1[2]*v2[0]))
        c = (v1[0]*v2[1]) - (v1[1]*v2[0])
        return [a,b,c]


    def projection_of_vect_to_xy_plane(self, vect, xy1, xy2):
        """
        Return porojection of vect to xy plane given by vectprs xy1 and xy2
        """
        norm = self.perpendicular_to_two_vects(xy1, xy2)
        vect_proj = np.array(vect) - (np.dot(vect,norm)/np.linalg.norm(norm)**2) * np.array(norm)
        return vect_proj

    def __connected_edge_angle_on_one_end(self, edg_number, stats, edg_end):
        """
        edg_number: integer with edg_number
        stats: dictionary with all statistics and computations
        edg_end: letter 'A' or 'B'
        creates phiXa, phiXb and phiXc.
        See Schwen2012 : Analysis and algorithmic generation of hepatic vascular system.
        """
        out = {}

        vector_key = 'vector' + edg_end
        try:
            vector = stats[edg_number][vector_key]
        except Exception as e:
            print (e)
            #traceback.print_exc()

        try:
            vectorX0 = self.__vector_of_connected_edge(edg_number, stats, edg_end, 0)
            phiXa = self.__vectors_to_angle_deg(vectorX0, vector)

            out.update({'phiA0' + edg_end + 'a':phiXa.tolist()})
        except Exception as e:
            print (e)
        try:
            vectorX1 = self.__vector_of_connected_edge(edg_number, stats, edg_end, 1)
        except Exception as e:
            print (e)

        try:

            vect_proj = self.projection_of_vect_to_xy_plane(vector, vectorX0, vectorX1)
            phiXa = self.__vectors_to_angle_deg(vectorX0, vectorX1)
            phiXb = self.__vectors_to_angle_deg(vector, vect_proj)
            vectorX01avg = \
                    np.array(vectorX0/np.linalg.norm(vectorX0)) +\
                    np.array(vectorX1/np.linalg.norm(vectorX1))
            phiXc = self.__vectors_to_angle_deg(vectorX01avg, vect_proj)

            out.update({
                'phi' + edg_end + 'a':phiXa.tolist(),
                'phi' + edg_end + 'b':phiXb.tolist(),
                'phi' + edg_end + 'c':phiXc.tolist()
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
        except Exception as e:
            traceback.print_exc()
        try:
            vectorA0 = self.__vector_of_connected_edge(edg_number, stats, 'A', 0)
            #angleA0a = np.arccos(np.dot(vectorA, vectorA0))
            angleA0 = self.__vectors_to_angle_deg(vectorA, vectorA0)
            print 'va ', vectorA0, 'a0a', angleA0, 'a0', angleA0
            out.update({'angleA0':angleA0.tolist()})
        except Exception as e:
            traceback.print_exc()
            print ("connected edge (number " + str(edg_number) + ") vectorA not found 0 " )

        try:
            vectorA1 = self.__vector_of_connected_edge(edg_number, stats, 'A', 1)
        except Exception as e:
            print ("connected edge (number " + str(edg_number) + ") vectorA not found 1")

        out.update(self.__connected_edge_angle_on_one_end(edg_number, stats, 'A'))
        out.update(self.__connected_edge_angle_on_one_end(edg_number, stats, 'B'))
        angleA0 = 0
        return out

#        try:
## we need find end of edge connected to our node
#            #import pdb; pdb.set_trace()
#            if stats[edg_number]['nodeIdA'] == stats[connectedEdgesA[0]]['nodeId0']:
## sousední hrana u uzlu na konci 0 má stejný node na svém konci 0 jako nynější hrana
#                vectorA0 = stats[connectedEdgesA[0]]['vectorA']
#            else:
#                vectorA0 = stats[connectedEdgesA[0]]['vectorB']
#        except:
#
#            # second neighbors on end "0"
#            if  stats[edg_number]['nodeIdA'] == stats[connectedEdgesA[1]]['nodeId0']:
## sousední hrana u uzlu na konci 0 má stejný node na svém konci 0 jako nynější hrana
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
        return (np.array(curve_model(t1, curve_params)) - \
                np.array(curve_model(t0,curve_params)))



    def __skeleton_nodes(self, data3d_skel, data3d_thr):
        """
        Return 3d ndarray where 0 is background, 1 is skeleton, 2 is node
        and 3 is terminal node

        """
        # @TODO  remove data3d_thr


# -----------------  get nodes --------------------------
        kernel = np.ones([3,3,3])
        #kernel[0,0,0]=0
        #kernel[0,0,2]=0
        #kernel[0,2,0]=0
        #kernel[0,2,2]=0
        #kernel[2,0,0]=0
        #kernel[2,0,2]=0
        #kernel[2,2,0]=0
        #kernel[2,2,2]=0

        #kernel = np.zeros([3,3,3])
        #kernel[1,1,:] = 1
        #kernel[1,:,1] = 1
        #kernel[:,1,1] = 1

        #data3d_skel = np.zeros([40,40,40])
        #data3d_skel[10,10,10] = 1
        #data3d_skel[10,11,10] = 1
        #data3d_skel[10,12,10] = 1
        #data3d_skel = data3d_skel.astype(np.int8)


        mocnost = scipy.ndimage.filters.convolve(data3d_skel, kernel) * data3d_skel
        #import pdb; pdb.set_trace()

        nodes = (mocnost > 3).astype(np.int8)
        terminals = ((mocnost == 2) | (mocnost == 1)).astype(np.int8)

        nt = nodes - terminals

        #pyed = seqt.QTSeedEditor(
        #        mocnost,
        #        contours=data3d_thr.astype(np.int8),
        #        seeds=nt
        #        )

        #import pdb; pdb.set_trace()

        data3d_skel[nodes==1] = 2
        data3d_skel[terminals==1] = 3

        return data3d_skel

    def node_analysis(sklabel):
        pass

    def __element_neighbors(self, el_number):
        """
        Gives array of element neghbors numbers
        """

        BOUNDARY_PX = 5
        element = (self.sklabel == el_number)
        enz = element.nonzero()



# limits for square neighborhood
        lo0 = np.max([0, np.min(enz[0]) - BOUNDARY_PX])
        lo1 = np.max([0, np.min(enz[1]) - BOUNDARY_PX])
        lo2 = np.max([0, np.min(enz[2]) - BOUNDARY_PX])
        hi0 = np.min([self.sklabel.shape[0], np.max(enz[0]) + BOUNDARY_PX])
        hi1 = np.min([self.sklabel.shape[1], np.max(enz[1]) + BOUNDARY_PX])
        hi2 = np.min([self.sklabel.shape[2], np.max(enz[2]) + BOUNDARY_PX])

# sklabel crop
        sklabelcr = self.sklabel[
                lo0:hi0,
                lo1:hi1,
                lo2:hi2
                ]

        # element crop
        element = (sklabelcr == el_number)

        dilat_element = scipy.ndimage.morphology.binary_dilation(
                element,
                structure=np.ones([3,3,3])
                )

        neighborhood = sklabelcr * dilat_element

# if el_number is edge, return nodes
        neighbors = np.unique(neighborhood)
        neighbors = neighbors [neighbors != 0]
        neighbors = neighbors [neighbors != el_number]
        #neighbors = [np.unique(neighborhood) != el_number]
        #if el_number > 0:
        #    neighbors = np.unique(neighborhood)[np.unique(neighborhood)<0]
        #else:
        #    neighbors = np.unique(neighborhood)[np.unique(neighborhood)>0]
        return neighbors


    def __edge_length(self, edg_number):
        #self.voxelsize_mm
        return {'lengthEstimation':float(np.sum(self.sklabel == edg_number) + 2)}

    def __edge_curve(self,  edg_number, edg_stats, voxelsize_mm):
        """
        Return params of curve and its starts and ends locations
        """
        retval = {}
        try:
            nd00, nd01, nd02 = (edg_stats['nodeIdA'] == self.sklabel).nonzero()
            nd10, nd11, nd12 = (edg_stats['nodeIdB'] == self.sklabel).nonzero()
            point0 = np.array([np.mean(nd00), np.mean(nd01), np.mean(nd02)])
            point1 = np.array([np.mean(nd10), np.mean(nd11), np.mean(nd12)])
            point0_mm = point0 * voxelsize_mm
            point1_mm = point1 * voxelsize_mm
            retval = {'curve_params':
                      {'start':point0_mm.tolist(),
                       'vector':(point1_mm-point0_mm).tolist()},
                      'nodeA_XYZ_mm': point0_mm.tolist(),
                      'nodeB_XYZ_mm': point1_mm.tolist()
                      }
        except Exception as ex:
            logger.warning("Problem in __edge_curve()")
            print (ex)

        return retval


    def edge_analysis(sklabel, edg_number):
        print 'element_analysis'





# element dilate * sklabel[sklabel < 0]

        pass

    def __radius_analysis_init(self):
        """
        Computes skeleton with distances from edge of volume.
        sklabel: skeleton or labeled skeleton
        volume_data: volumetric data with zeros and ones
        """
        uq = np.unique(self.volume_data)
        if (uq[0]==0) & (uq[1]==1):
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
        #import ipdb; ipdb.set_trace() # BREAKPOINT
        edg_skdst = skdst * (self.sklabel == edg_number)
        return np.mean(edg_skdst[edg_skdst != 0])



    def __connection_analysis(self, edg_number):
        """
        Analysis of which edge is connected
        """
        edg_neigh = self.__element_neighbors(edg_number)
        if len(edg_neigh) != 2:
            print ('Wrong number (' + str(edg_neigh) +
                    ') of connected edges in connection_analysis() for\
                    edge number ' + str(edg_number))
            edg_stats = {
                    'id':edg_number
                    }
        else:
            connectedEdgesA = self.__element_neighbors(edg_neigh[0])
            connectedEdgesB = self.__element_neighbors(edg_neigh[1])
# remove edg_number from connectedEdges list
            connectedEdgesA = connectedEdgesA[connectedEdgesA != edg_number]
            connectedEdgesB = connectedEdgesB[connectedEdgesB != edg_number]
            print 'edg_neigh ', edg_neigh, ' ,0: ', connectedEdgesA, '  ,0 '

            #import pdb; pdb.set_trace()

            edg_stats = {
                    'id':edg_number,
                    'nodeIdA':int(edg_neigh[0]),
                    'nodeIdB':int(edg_neigh[1]),
                    'connectedEdgesA':connectedEdgesA.tolist(),
                    'connectedEdgesB':connectedEdgesB.tolist()
                    }

        return edg_stats

def generate_sample_data(m=1, noise_level=0.02, gauss_sigma=0.15):
    """
    Generate sample vessel system.
    J. Kunes
    
    Input:
        m - output will be (100*m)^3 numpy array
        noise_level - noise power, disable noise with -1
        gauss_sigma - gauss filter sigma, disable filter with -1
        
    Output:
        (100*m)^3 numpy array
            voxel size = [1,1,1]
    """
    import thresholding_functions
    
    data3d = np.zeros((100*m,100*m,100*m), dtype=np.int)

    # size 8
    data3d_new = np.ones((100*m,100*m,100*m), dtype=np.bool)
    data3d_new[0:30*m,20*m,20*m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 8*m] = 0
    data3d[data3d_new == 0] = 1
    # size 7
    data3d_new = np.ones((100*m,100*m,100*m), dtype=np.bool)
    data3d_new[31*m:70*m,20*m,20*m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 7*m] = 0
    data3d[data3d_new == 0] = 1
    # size 6
    data3d_new = np.ones((100*m,100*m,100*m), dtype=np.bool)
    data3d_new[70*m,20*m:50*m,20*m] = 0
    data3d_new[31*m,20*m,20*m:70*m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 6*m] = 0
    data3d[data3d_new == 0] = 1
    # size 5
    data3d_new = np.ones((100*m,100*m,100*m), dtype=np.bool)
    data3d_new[70*m:95*m,20*m,20*m] = 0
    data3d_new[31*m:60*m,20*m,70*m] = 0
    data3d_new[70*m:90*m,50*m,20*m] = 0
    data3d_new[70*m,50*m,20*m:50*m] = 0
    data3d_new[31*m,20*m:45*m,20*m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 5*m] = 0
    data3d[data3d_new == 0] = 1
    # size 4
    data3d_new = np.ones((100*m,100*m,100*m), dtype=np.bool)
    data3d_new[31*m,20*m:50*m,70*m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 4*m] = 0
    data3d[data3d_new == 0] = 1
    # size 3
    data3d_new = np.ones((100*m,100*m,100*m), dtype=np.bool)
    data3d_new[31*m:50*m,50*m,70*m] = 0
    data3d_new[31*m:50*m,45*m,20*m] = 0
    data3d_new[70*m,50*m:70*m,50*m] = 0
    data3d_new[70*m:80*m,50*m,50*m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 3*m] = 0
    data3d[data3d_new == 0] = 1
    
    data3d = data3d*3030
    data3d += 5920
    
    if gauss_sigma>=0:
        sigma = np.round(gauss_sigma, 2)
        sigmaNew = thresholding_functions.calculateSigma([1,1,1], sigma)
        data3d = thresholding_functions.gaussFilter(data3d, sigmaNew)
    
    if noise_level>=0:
        noise = np.random.normal(1,noise_level,(100*m,100*m,100*m))
        data3d = data3d*noise
    
    return data3d

def parser_init():
    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser'
    )
    parser.add_argument('-i', '--inputfile',
        default=None,
        help='Input file, .tif file')
#    parser.add_argument('-o', '--outputfile',
#        default='histout.pkl',
#        help='output file')
    parser.add_argument('-t', '--threshold', type=int,
        default=-1, #6600,
        help='data threshold, default -1 (gui/automatic selection)')
    parser.add_argument(
        '-is', '--input_is_skeleton', action='store_true',
        help='Input file is .pkl file with skeleton')
    parser.add_argument('-cr', '--crop', type=int, metavar='N', nargs='+',
        #default=[0,-1,0,-1,0,-1],
        default=None,
        help='Segmentation labels, default 1')
    parser.add_argument(
        '--crgui', action='store_true',
        help='GUI crop')
    parser.add_argument(
        '--nogui', action='store_true',
        help='Disable GUI')
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()
    
    return args

# Processing data without gui
def processData(inputfile=None,threshold=None,skeleton=False,crop=None):
    ### when input is just skeleton
    if skeleton:
        logger.info("input is skeleton")
        struct = misc.obj_from_file(filename='tmp0.pkl', filetype='pickle')
        data3d_skel = struct['skel']
        data3d_thr = struct['thr']
        data3d = struct['data3d']
        metadata = struct['metadata']
        ha = HistologyAnalyser(data3d, metadata, threshold, nogui=True)
        logger.info("end of is skeleton")
    else: 
        ### Reading/Generating data
        if inputfile is None: ## Using generated sample data
            logger.info('Generating sample data...')
            metadata = {'voxelsize_mm': [1, 1, 1]}
            data3d = generate_sample_data(2)
        else: ## Normal runtime
            dr = datareader.DataReader()
            data3d, metadata = dr.Get3DData(inputfile)
        
        ### Crop data
        if crop is not None:
            logger.debug('Croping data: %s', str(crop))
            data3d = data3d[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
        
        ### Init HistologyAnalyser object
        logger.debug('Init HistologyAnalyser object')
        ha = HistologyAnalyser(data3d, metadata, threshold, nogui=True)
        
        ### No GUI == No Remove Area
        
        ### Segmentation
        logger.debug('Segmentation')
        data3d_thr, data3d_skel = ha.data_to_skeleton()
        
    ### Computing statistics
    logger.info("######### statistics")
    ha.skeleton_to_statistics(data3d_thr, data3d_skel)
    
    ### Saving files
    logger.info("##### write to file")
    ha.writeStatsToCSV()
    ha.writeStatsToYAML()
    ha.writeSkeletonToPickle('skel.pkl')
    #struct = {'skel': data3d_skel, 'thr': data3d_thr, 'data3d': data3d, 'metadata':metadata}
    #misc.obj_to_file(struct, filename='tmp0.pkl', filetype='pickle')
    
    ### End
    logger.info('Finished')

def main():
    args = parser_init()

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    #ch = logging.StreamHandler() #https://docs.python.org/2/howto/logging.html#configuring-logging
    #logger.addHandler(ch)

    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    if args.nogui:
        logger.info('Running without GUI')
        logger.info('Input file -> %s', args.inputfile)
        logger.info('Data crop -> %s', str(args.crop))
        logger.info('Threshold -> %s', args.threshold)
        processData(inputfile=args.inputfile,threshold=args.threshold,skeleton=args.input_is_skeleton,crop=args.crop)
    else:
        app = QApplication(sys.argv)
        gui = HA_GUI.HistologyAnalyserWindow(inputfile=args.inputfile,threshold=args.threshold,skeleton=args.input_is_skeleton,crop=args.crop,crgui=args.crgui)
        sys.exit(app.exec_())
        
if __name__ == "__main__":
    main()
