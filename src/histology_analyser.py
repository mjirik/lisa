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

import seed_editor_qt as seqt
import skelet3d
import segmentation


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
        data3d_skel = skelet3d.skelet3d(data3d_thr)

       # pyed = seqt.QTSeedEditor(
       #         data3d, 
       #         contours=data3d_thr.astype(np.int8),
       #         seeds=data3d_skel.astype(np.int8)
       #         )
        #app.exec_()
        data3d_nodes = skeleton_nodes(data3d_skel, data3d_thr)
        stats = skeleton_analysis(data3d_nodes)
        #data3d_nodes[data3d_nodes==3] = 2
        self.stats = stats

       # pyed = seqt.QTSeedEditor(
       #         data3d,
       #         seeds=(data3d_nodes).astype(np.int8),
       #         contours=data3d_thr.astype(np.int8)
       #         )


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

            for dataline in data:
                writer.writerow(self.__dataToCSVLine(dataline))
                #spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
    def __dataToCSVLine(self, dataline):
        try:
            arr = [
                    dataline['id'], 
                    dataline['nodeId0'],
                    dataline['nodeId1']
                    ]
        except:
            arr = []

        return arr



    def show(self):
        app = QApplication(sys.argv)
        seqt.QTSeedEditor(self.output.astype(np.int16))
        app.exec_()


def skeleton_nodes(data3d_skel, data3d_thr):
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

def element_neighbors(sklabel, el_number):
    """
    Gives array of element neghbors numbers
    """

    BOUNDARY_PX = 5
    element = (sklabel == el_number)
    enz = element.nonzero()



# limits for square neighborhood
    lo0 = np.max([0, np.min(enz[0]) - BOUNDARY_PX])
    lo1 = np.max([0, np.min(enz[1]) - BOUNDARY_PX])
    lo2 = np.max([0, np.min(enz[2]) - BOUNDARY_PX])
    hi0 = np.min([sklabel.shape[0], np.max(enz[0]) + BOUNDARY_PX])
    hi1 = np.min([sklabel.shape[1], np.max(enz[1]) + BOUNDARY_PX])
    hi2 = np.min([sklabel.shape[2], np.max(enz[2]) + BOUNDARY_PX])

# sklabel crop
    sklabelcr = sklabel[
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


def edge_analysis(sklabel, edg_number):
    print 'element_analysis'


    


# element dilate * sklabel[sklabel < 0]

    pass

def connection_analysis(sklabel, edg_number):
    """
    Analysis of which edge is connected
    """
    edg_neigh = element_neighbors(sklabel,edg_number)
    if len(edg_neigh) != 2:
        print ('Wrong number (' + str(edg_neigh) +
                ') of connected edges in connection_analysis() for\
                        edge number ' + str(edg_number))
        edg_stats = {
                'id':edg_number
                }
    else:
        connectedEdges0 = element_neighbors(sklabel, edg_neigh[0])
        connectedEdges1 = element_neighbors(sklabel, edg_neigh[1])
# remove edg_number from connectedEdges list
        connectedEdges0 = connectedEdges0[connectedEdges0 != edg_number]
        connectedEdges1 = connectedEdges1[connectedEdges1 != edg_number]
        print 'edg_neigh ', edg_neigh, ' ,0: ', connectedEdges0, '  ,0 '

        #import pdb; pdb.set_trace()

        edg_stats = {
                'id':edg_number,
                'nodeId0':edg_neigh[0],
                'nodeId1':edg_neigh[1],
                'connectedEdges0':connectedEdges0,
                'connectedEdges1':connectedEdges1
                }

    return edg_stats

def skeleton_analysis(skelet_nodes, volume_data = None):
    """
    Glossary:
    element: line structure of skeleton connected to node on both ends
    node: connection point of elements. It is one or few voxelsize_mm
    terminal: terminal node
    

    """
    sklabel_edg, len_edg = scipy.ndimage.label(skelet_nodes == 1, structure=np.ones([3,3,3]))
    sklabel_nod, len_nod = scipy.ndimage.label(skelet_nodes > 1, structure=np.ones([3,3,3]))

    sklabel = sklabel_edg - sklabel_nod

    stats = []
    len_edg = 100
    
    for i in range (1,len_edg):
        edgst = connection_analysis(sklabel, i)
        #edgst = edge_analysis(sklabel, i)
        stats.append(edgst)

    return stats
    #import pdb; pdb.set_trace()



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
    #ha.show()
    

