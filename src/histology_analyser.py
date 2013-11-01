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


import numpy as np
import scipy.ndimage
import misc
import datareader
import SimpleITK as sitk
import scipy.ndimage
from PyQt4.QtGui import QApplication
import csv

import sys, traceback


import seed_editor_qt as seqt
import skelet3d
import segmentation
import misc


GAUSSIAN_SIGMA = 1


class HistologyAnalyser:
    def __init__ (self, data3d, metadata, threshold):
        self.data3d = data3d
        self.metadata = metadata
        self.threshold = threshold



    def run(self):
        #self.preprocessing()

        data3d_thr = segmentation.vesselSegmentation(
            self.data3d,
            segmentation = np.ones(self.data3d.shape, dtype='int8'),
            #segmentation = oseg.orig_scale_segmentation,
            threshold = -1,
            inputSigma = 0.15,
            dilationIterations = 2,
            nObj = 1,
            biggestObjects = False,
#        dataFiltering = True,
            interactivity = True,
            binaryClosingIterations = 5,
            binaryOpeningIterations = 1)
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

        app = QApplication(sys.argv)

        #app.exec_()

        print "skelet"
        data3d_thr = (data3d_thr > 0).astype(np.int8)
        data3d_skel = skelet3d.skelet3d(data3d_thr)

       # pyed = seqt.QTSeedEditor(
       #         data3d, 
       #         contours=data3d_thr.astype(np.int8),
       #         seeds=data3d_skel.astype(np.int8)
       #         )
        #app.exec_()
        skan = SkeletonAnalyser(data3d_skel, volume_data=data3d_thr)
        data3d_nodes_vis = skan.sklabel.copy()
# edges
        data3d_nodes_vis[data3d_nodes_vis > 0 ] = 1
# nodes and terminals
        data3d_nodes_vis[data3d_nodes_vis < 0 ] = 2

        pyed = seqt.QTSeedEditor(
                data3d,
                seeds=(data3d_nodes_vis).astype(np.int8),
                contours=data3d_thr.astype(np.int8)
                )
        app.exec_()
        stats = skan.skeleton_analysis()
        #data3d_nodes[data3d_nodes==3] = 2
        self.stats = stats



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
        misc.obj_to_file(self.stats, filetype='yaml')

        #sitk.
    def writeStatsToCSV(self, filename='hist_stats.csv'):
        data = self.stats

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
    def __dataToCSVLine(self, dataline):
        arr = []
# @TODO arr.append
        try:
            arr = [
                    dataline['id'], 
                    dataline['nodeId0'],
                    dataline['nodeId1'],
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
        if self.volume_data is not None:
            skdst = self.__radius_analysis_init()

        stats = {}
        len_edg = np.max(self.sklabel)
        len_edg = 30
        
        for edg_number in range (1,len_edg):
            edgst = self.__connection_analysis(edg_number)
            edgst.update(self.__edge_length(edg_number))
            edgst.update(self.__edge_curve(edg_number, edgst))
            edgst.update(self.__edge_vectors(edg_number, edgst))
            #edgst = edge_analysis(sklabel, i)
            if self.volume_data is not None:
                edgst['radius'] = float(self.__radius_analysis(skdst, edg_number))
            stats[edgst['id']] = edgst

        # second run for connected analysis
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
            vector0 = self.__get_vector_from_curve(0.25, 0, curve_params)
            vector1 = self.__get_vector_from_curve(0.75, 1, curve_params)
        except Exception as ex:
            print (ex)
            return {}


        return {'vector0':vector0.tolist(), 'vector1': vector1.tolist()}

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
        if edg_end == 0:
            connectedEdges = stats[edg_number]['connectedEdges0']
            ndid = 'nodeId0'
        else:
            connectedEdges = stats[edg_number]['connectedEdges1']
            ndid = 'nodeId1'

        connectedEdgeStats = connectedEdges[con_edg_order]

        if stats[edg_number][ndid] == connectedEdgeStats['nodeId0']:
# sousední hrana u uzlu na konci 0 má stejný node na svém konci 0 jako nynější hrana
            vector = connectedEdgeStats['vector0']
        elif stats[edg_number][ndid] == connectedEdgeStats['nodeId1']:
            vector = connectedEdgeStats['vector1']


        return vector

    def __connected_edge_angle(self, edg_number, stats):
        """
        count angles betwen end vectors of edges
        """

        vector0 = stats[edg_number]['vector0']
        vector1 = stats[edg_number]['vector1']
        try:
            vector00 = self.__vector_of_connected_edge(edg_number, stats, 0, 0)
        except Exception as e:
            print ("connected edge (number ", edg_number, ")vector not found")
            print (e)

        angle0 = 0
        #angle0 = np.arccos(np.dot(vector0, vector00))
        return {'angle0':angle0}

#        try:
## we need find end of edge connected to our node
#            #import pdb; pdb.set_trace()
#            if stats[edg_number]['nodeId0'] == stats[connectedEdges0[0]]['nodeId0']:
## sousední hrana u uzlu na konci 0 má stejný node na svém konci 0 jako nynější hrana
#                vector00 = stats[connectedEdges0[0]]['vector0']
#            else:
#                vector00 = stats[connectedEdges0[0]]['vector1']
#        except:
#
#            # second neighbors on end "0"
#            if  stats[edg_number]['nodeId0'] == stats[connectedEdges0[1]]['nodeId0']:
## sousední hrana u uzlu na konci 0 má stejný node na svém konci 0 jako nynější hrana
#                vector01 = stats[connectedEdges0[1]]['vector0']
#            else:
#                vector01 = stats[connectedEdges0[1]]['vector1']
#            print "ahoj"
#        except:
#            print ("Cannot compute angles for edge " + str(edg_number))
#
#        angle0 = np.arccos(np.dot(vector01, vector00))
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

    def __edge_curve(self,  edg_number, edg_stats):
        retval = {}
        try:
            nd00, nd01, nd02 = (edg_stats['nodeId0'] == self.sklabel).nonzero()
            nd10, nd11, nd12 = (edg_stats['nodeId1'] == self.sklabel).nonzero()
            point0 = np.array([np.mean(nd00), np.mean(nd01), np.mean(nd02)])
            point1 = np.array([np.mean(nd10), np.mean(nd11), np.mean(nd12)])
            retval = {'curve_params':
                    {'start':point0.tolist(), 
                        'vector':(point1-point0).tolist()}}
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
            dst = dst * (self.sklabel != 0)
            
            return dst

        else:
            print "__radius_analysis_init() error.  "
            return None


    def __radius_analysis(self, edg_number, skdst):
        """
        return smaller radius of tube
        """
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
            connectedEdges0 = self.__element_neighbors(edg_neigh[0])
            connectedEdges1 = self.__element_neighbors(edg_neigh[1])
# remove edg_number from connectedEdges list
            connectedEdges0 = connectedEdges0[connectedEdges0 != edg_number]
            connectedEdges1 = connectedEdges1[connectedEdges1 != edg_number]
            print 'edg_neigh ', edg_neigh, ' ,0: ', connectedEdges0, '  ,0 '

            #import pdb; pdb.set_trace()

            edg_stats = {
                    'id':edg_number,
                    'nodeId0':int(edg_neigh[0]),
                    'nodeId1':int(edg_neigh[1]),
                    'connectedEdges0':connectedEdges0.tolist(),
                    'connectedEdges1':connectedEdges1.tolist()
                    }

        return edg_stats



if __name__ == "__main__":
    import misc
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description='\
            3D visualization of segmentation\n\
            \npython show_segmentation.py\n\
            \npython show_segmentation.py -i resection.pkl -l 2 3 4 -d 4')
    parser.add_argument('-i', '--inputfile',
            default='organ.pkl',
            help='input file')
    parser.add_argument('-t', '--threshold', type=int,
            default=6600,
            help='data threshold, default 1')
    parser.add_argument('-cr', '--crop', type=int, metavar='N', nargs='+',
            default=[0,-1,0,-1,0,-1],
            help='segmentation labels, default 1')
    args = parser.parse_args()

    #data = misc.obj_from_file(args.inputfile, filetype = 'pickle')


    dr = datareader.DataReader()
    data3d, metadata = dr.Get3DData(args.inputfile)
# crop data
    cr = args.crop
    data3d = data3d[cr[0]:cr[1], cr[2]:cr[3], cr[4]:cr[5]]



    ha = HistologyAnalyser(data3d, metadata, args.threshold)
    ha.run()
    ha.writeStatsToCSV()
    import pdb; pdb.set_trace()
    ha.writeStatsToYAML()
    #ha.show()
    

