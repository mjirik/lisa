# ! /usr/bin/python
# -*- coding: utf-8 -*-

"""
python src/histology_analyser.py -i ~/data/medical/data_orig/ja\
tra_mikro_data/Nejlepsi_rozliseni_nevycistene -t 6800 -cr 0 -1 100 300 100 300
"""

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))

import traceback

import logging
logger = logging.getLogger(__name__)

import argparse

from PyQt4.QtGui import QApplication

import numpy as np
import scipy.ndimage
import misc
from io3d import datareader
import csv

import skelet3d
import segmentation
import sed3 as se

import histology_analyser_gui as HA_GUI
from histology_report import HistologyReport
from skeleton_analyser import SkeletonAnalyser
import surface_measurement


class HistologyAnalyser:

    def __init__(self, data3d, metadata, threshold=-1, binaryClosing=-1,
                 binaryOpening=-1, nogui=True, qapp=None,
                 aggregate_near_nodes_distance=0, ):
        self.qapp=qapp
        self.data3d = data3d
        if 'voxelsize_mm' not in metadata.keys():
            metadata['voxelsize_mm'] = [1, 1, 1]
        self.metadata = metadata

        self.data3d_thr = None
        self.data3d_skel = None

        self.threshold = threshold
        
        if binaryClosing == -1:
            self.binaryClosing = 1
        else:
            self.binaryClosing = binaryClosing
        if binaryOpening == -1:
            self.binaryOpening = 1
        else:
            self.binaryOpening = binaryOpening

        self.nogui = nogui

        self.data3d_masked = None
        
        self.aggregate_near_nodes_distance = aggregate_near_nodes_distance
        self.graph_label = 'microstructure'
        self.anotation = ''

    def set_anotation(self, text):
        self.anotation = text

    def get_voxelsize(self):
        return self.metadata['voxelsize_mm']

    def get_data3d(self):
        return self.data3d

    def get_data3d_segmented(self):
        return self.data3d_thr

    def get_data3d_skeleton(self):
        return self.data3d_skel

    def create_border_for_skeletonization(self, data3d, size=50):
        logger.debug('Generating border for skeletonization')
        data3d = data3d.copy()
        for i in range(size):
            logger.debug('iteration num '+str(i))

            new_shape = (data3d.shape[0]+2,
                         data3d.shape[1]+2,
                         data3d.shape[2]+2)
            logger.debug('new shape: '+str(new_shape))
            work_array = np.ones(new_shape, dtype = type(data3d[0][0][0]))

            # copy sides
            work_array[1:-1, 0, 1:-1] = data3d[:, 0, :]
            work_array[1:-1, -1, 1:-1] = data3d[:, -1, :]

            work_array[0, 1:-1, 1:-1] = data3d[0, :, :]
            work_array[-1, 1:-1, 1:-1] = data3d[-1, :, :]

            work_array[1:-1, 1:-1, 0] = data3d[:, :, 0]
            work_array[1:-1, 1:-1, -1] = data3d[:, :, -1]

            # copy corners sides
            work_array[1:-1, 0, 0] = data3d[:, 0, 0]
            work_array[1:-1, -1, 0] = data3d[:, -1, 0]
            work_array[1:-1, 0, -1] = data3d[:, 0, -1]
            work_array[1:-1, -1, -1] = data3d[:, -1, -1]

            work_array[0, 1:-1, 0] = data3d[0, :, 0]
            work_array[0, 1:-1, -1] = data3d[0, :, -1]
            work_array[-1, 1:-1, 0] = data3d[-1, :, 0]
            work_array[-1, 1:-1, -1] = data3d[-1, :, -1]

            work_array[0, 0, 1:-1] = data3d[0, 0, :]
            work_array[0, -1, 1:-1] = data3d[0, -1, :]
            work_array[-1, 0, 1:-1] = data3d[-1, 0, :]
            work_array[-1, -1, 1:-1] = data3d[-1, -1, :]

            # copy corners
            work_array[0, 0, 0] = data3d[0, 0, 0]
            work_array[0, 0, -1] = data3d[0, 0, -1]

            work_array[0, -1, 0] = data3d[0, -1, 0]
            work_array[0, -1, -1] = data3d[0, -1, -1]

            work_array[-1, 0, 0] = data3d[-1, 0, 0]
            work_array[-1, 0, -1] = data3d[-1, 0, -1]

            work_array[-1, -1, 0] = data3d[-1, -1, 0]
            work_array[-1, -1, -1] = data3d[-1, -1, -1]

            # erode
            erode_iterations = int(round(i/8))
            if erode_iterations>=1:
                work_array = scipy.ndimage.morphology.binary_erosion(work_array, border_value = 1, iterations = erode_iterations)

            # check if everything is eroded -> exit
            work_array[1:-1, 1:-1, 1:-1] = np.ones(data3d.shape, dtype = type(data3d[0][0][0]))
            eroded_sum = np.sum(np.sum(np.sum(work_array)))
            orig_sum = data3d.shape[0]*data3d.shape[1]*data3d.shape[2]

            if eroded_sum==orig_sum:
                eroded = True
            else:
                eroded = False

            # copy original data to center
            work_array[1:-1, 1:-1, 1:-1] = data3d

            data3d = work_array.copy()

            # if everything eroded -> exit
            if eroded:
                logger.debug('Everything eroded -> exiting early')
                break

        return data3d

    def data_to_binar(self):
        # ## Median filter
        filteredData = scipy.ndimage.filters.median_filter(self.data3d, size=2)

        # ## Segmentation
        data3d_thr = segmentation.vesselSegmentation(
            filteredData,  # self.data3d,
            segmentation=np.ones(filteredData.shape, dtype='int8'),
            threshold=self.threshold,
            inputSigma=0,  # 0.15,
            dilationIterations=2,
            nObj=1,
            biggestObjects=False,
            useSeedsOfCompactObjects=False,
            interactivity=not self.nogui,
            binaryClosingIterations=self.binaryClosing, # noqa 5,
            binaryOpeningIterations=self.binaryOpening, # 1 # noqa
            qapp=self.qapp
            )
        del(filteredData)

        # ## Zalepeni der
        scipy.ndimage.morphology.binary_fill_holes(data3d_thr,
                                                   output=data3d_thr)

        self.data3d_thr = data3d_thr

    def binar_to_skeleton(self):
        # create border with generated stuff for skeletonization
        #expanded_data = self.create_border_for_skeletonization(self.data3d_thr, size=50)
        expanded_data = self.data3d_thr

        expanded_skel = skelet3d.skelet3d(
            (expanded_data > 0).astype(np.int8)
        )

        # cut data shape back to original size
        border = (expanded_data.shape[0]-self.data3d_thr.shape[0])/2
        if border!=0:
            self.data3d_skel = expanded_skel[border:-border, border:-border, border:-border].copy()
        else:
            self.data3d_skel = expanded_skel

    def data_to_skeleton(self):
        self.data_to_binar()
        self.binar_to_skeleton()

    def data_to_statistics(self, guiUpdateFunction=None):
        self.stats = {}

        if (self.data3d_skel is None) or (self.data3d_thr is None):
            logger.debug('Skeleton was not generated!!! Generating now...')
            self.data_to_skeleton()

        # # add general info
        logger.debug('Computing general statistics...')
        if guiUpdateFunction is not None: guiUpdateFunction(0, 3, "Volumetric methods (Vv, Sv)")
        vs = self.metadata['voxelsize_mm']
        voxel_volume_mm3 = vs[0]*vs[1]*vs[2]
        volume_px = self.data3d.shape[0] \
            * self.data3d.shape[1] * self.data3d.shape[2]
        volume_mm3 = volume_px*voxel_volume_mm3

        # pouzij oblast zajmu
        if self.data3d_masked is None:
            self.data3d_masked = np.ones(self.data3d.shape, np.int8)
        self.data3d_masked[self.data3d_masked > 1] = 1
        used_volume_px = np.sum(np.sum(np.sum(self.data3d_masked)))
        used_volume_mm3 = used_volume_px*voxel_volume_mm3
        if guiUpdateFunction is not None: guiUpdateFunction(1, 3, "Volumetric methods (Vv, Sv)")

        vessel_volume_fraction = float(np.sum(np.sum(np.sum(
            self.data3d_thr)))) / float(used_volume_px)
        if guiUpdateFunction is not None: guiUpdateFunction(2, 3, "Volumetric methods (Vv, Sv)")
        
        sv = surface_measurement.surface_density(
            self.data3d_thr, self.metadata['voxelsize_mm'], self.data3d_masked)
        if guiUpdateFunction is not None: guiUpdateFunction(3, 3, "Volumetric methods (Vv, Sv)")

        info = {
            'voxel_size_mm': list(vs),
            'voxelsize_mm': list(vs),
            'voxel_volume_mm3': float(voxel_volume_mm3),
            'shape_px': list(self.data3d.shape),
            'volume_px': float(volume_px),
            'volume_mm3': float(volume_mm3),
            'used_volume_px': float(used_volume_px),
            'used_volume_mm3': float(used_volume_mm3),
            'vessel_volume_fraction': float(vessel_volume_fraction),
            'surface_density': float(sv),
            'processed_by': 'Histology Analyser',
            'anotation': self.anotation
        }
        self.stats.update({'general': info})

        # # process skeleton to statistics
        logger.debug('Computing skeleton to statistics...')
        skan = SkeletonAnalyser(
            self.data3d_skel,
            volume_data=self.data3d_thr,
            voxelsize_mm=self.metadata['voxelsize_mm'],
            aggregate_near_nodes_distance = self.aggregate_near_nodes_distance
            )
        stats = skan.skeleton_analysis(guiUpdateFunction=guiUpdateFunction)
        # needed only by self.writeSkeletonToPickle()
        self.sklabel = skan.sklabel
        self.stats.update({'graph': {self.graph_label: stats}})
        
        # compute Nv statistic
        self.stats['general']['Nv'] = float(self.get_Nv())
        
    def __insert_node(self, nodes, nodeIdLabel, key):
        """
        used in getNv()
        nodeIdALabel: "A" or "B"
        """
        edge = self.stats['graph'][self.graph_label][key]
        try:
            nodeIdLab = edge["nodeId" + nodeIdLabel]
            if nodeIdLab in nodes: 
                nodes[nodeIdLab]['connected_edges'] += [key] 
            else:
                nodes[nodeIdLab] = {
                        'connected_edges':[key],
                        'centroid': self.stats['graph'][self.graph_label][key]['node' + nodeIdLabel + '_ZYX_mm']
                        }
        except Exception, e:
            logger.warning('get_Nv(): no nodeIdA')
        return nodes

    def get_Nv(self):

        logger.debug('Computing Nv...')
        
        # get lists of edges that are connected to nodes
        nodes = {}
        for key in self.stats['graph'][self.graph_label]:
            nodes = self.__insert_node(nodes, 'A', key)
            nodes = self.__insert_node(nodes, 'B', key)
            # edge = self.stats['graph'][key]
            # # TODO use insert_node function
            # try:
            #     nodeIdA = edge['nodeIdA']
            #     if nodeIdA in nodes: 
            #         nodes[nodeIdA] += [key] 
            #     else: nodes[nodeIdA] = [key]
            # except Exception, e:
            #     logger.warning('get_Nv(): no nodeIdA')
            # try:
            #     nodeIdB = edge['nodeIdB']
            #     if nodeIdB in nodes: nodes[nodeIdB] += [key] 
            #     else: nodes[nodeIdB] = [key]
            # except Exception, e:
            #     logger.warning('get_Nv(): no nodeIdB')
        logger.debug('Read ' + str(len(nodes)) + ' nodes')
        
        # Get Pn (number of nodes with valence n)
        max_l = 0
        for n_key in nodes:
            max_l = max(max_l, len(nodes[n_key]['connected_edges']))
        Pn = np.zeros(max_l+1)
        for n_key in nodes:
            Pn[len(nodes[n_key]['connected_edges'])] += 1
        logger.debug(str(Pn))
        
        # Get N(cap)
        Ncap = 0
        for n in range(len(Pn)):
            if n > 1: # dont count terminal nodes 
                Ncap += ( (n-2)/2.0 ) * Pn[n]
        
        # Get V(ref)
        Vref = self.stats['general']['used_volume_mm3']
        
        # Compute Nv
        Nv = ( Ncap / float(Vref) ) + 1
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        logger.debug('Nv is ' + str(Nv))
        
        return Nv

    def showSegmentedData(self):
        skan = SkeletonAnalyser(
            self.data3d_skel,
            volume_data=self.data3d_thr,
            voxelsize_mm=self.metadata['voxelsize_mm'],
            aggregate_near_nodes_distance = self.aggregate_near_nodes_distance
            )
        data3d_nodes_vis = skan.sklabel.copy()
        del(skan)

        # edges
        data3d_nodes_vis[data3d_nodes_vis > 0] = 1
        # nodes and terminals
        data3d_nodes_vis[data3d_nodes_vis < 0] = 2

        if not self.nogui:
            # @TODO change sed3 to sed3qt
            pyed = se.sed3(
                self.data3d,
                seeds=(data3d_nodes_vis).astype(np.int8),
                contour=self.data3d_thr.astype(np.int8)
            )
            pyed.show()


    def writeStatsToYAML(self, filename='hist_stats.yaml'):
        logger.debug('writeStatsToYAML')
        self.prepareVesselTree()
        misc.obj_to_file(self.stats, filename=filename, filetype='yaml')

    def prepareVesselTree(self):
        gr = self.stats['graph'][self.graph_label]
        for i in gr:
            logger.info('Processing edge num '+str(i))
            try:
                gr[i]['lengthEstimationPoly'] = float(gr[i]['lengthEstimationPoly'])
            except:
                logger.warning('lengthEstimationPoly not a number')
                gr[i]['lengthEstimationPoly'] = None
                
            try:
                gr[i]['lengthEstimationSpline'] = float(gr[i]['lengthEstimationSpline'])
            except:
                logger.warning('lengthEstimationSpline not a number')
                gr[i]['lengthEstimationSpline'] = None
            
            try:
                self.stats['lengthEstimation'] = float(gr[i]['lengthEstimation'])
            except:
                logger.warning('lengthEstimation not a number')
                gr[i]['lengthEstimation'] = None
            
            try:
                gr[i]['lengthEstimationPixel'] = float(gr[i]['lengthEstimationPixel'])
            except:
                logger.warning('lengthEstimationPixel not a number')
                gr[i]['lengthEstimationPixel'] = None
        self.stats['graph'][self.graph_label] = gr

    def exportVT2esofspy(self, filename='trace.txt'):
        import vesseltree_export as vte
        logger.debug('esofspy export')
        self.prepareVesselTree()

        vte.vt2esofspy(self.stats, filename)


        

    def writeStatsToCSV(self, filename='hist_stats.csv'):
        info = self.stats['general']
        data = self.stats['graph'][self.graph_label]

        with open(filename, 'wb') as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter=';',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            # save csv info
            info_labels = ['shape_px', 'surface_density', 'used_volume_mm3', 'used_volume_px', 'vessel_volume_fraction', 'volume_mm3', 'volume_px', 'voxel_size_mm', 'voxel_volume_mm3']
            entry_labels = ['connectedEdgesA', 'connectedEdgesB', 'id', 'lengthEstimation', 'lengthEstimationPixel', 'lengthEstimationPoly', 'lengthEstimationSpline', 'nodeA_ZYX', 'nodeA_ZYX_mm', 'nodeB_ZYX', 'nodeB_ZYX_mm', 'nodeIdA', 'nodeIdB', 'nodesDistance', 'radius_mm', 'tortuosity', 'vectorA', 'vectorB']

            try:
                writer.writerow(info_labels)
                writer.writerow(entry_labels)
            except Exception, e:
                logger.error('Error when saving line (csv info) to csv: '+str(e))

            # save info
            try:
                writer.writerow(['__info__'])
                writer.writerow(info['shape_px'])
                writer.writerow([info['surface_density']])
                writer.writerow([info['used_volume_mm3']])
                writer.writerow([info['used_volume_px']])
                writer.writerow([info['vessel_volume_fraction']])
                writer.writerow([info['volume_mm3']])
                writer.writerow([info['volume_px']])
                writer.writerow(info['voxel_size_mm'])
                writer.writerow([info['voxel_volume_mm3']])
            except Exception, e:
                logger.error('Error when saving line (info) to csv: '+str(e))
                logger.error(traceback.format_exc())

            # save data
            for lineid in data:
                dataline = data[lineid]
                try:
                    writer.writerow(['__entry__'])
                    writer.writerow(dataline['connectedEdgesA'])
                    writer.writerow(dataline['connectedEdgesB'])
                    #writer.writerow(dataline['curve_params']['start'])
                    #writer.writerow(dataline['curve_params']['vector'])
                    writer.writerow([dataline['id']])
                    writer.writerow([dataline['lengthEstimation']])
                    writer.writerow([dataline['lengthEstimationPixel']])
                    writer.writerow([dataline['lengthEstimationPoly']])
                    writer.writerow([dataline['lengthEstimationSpline']])
                    writer.writerow(dataline['nodeA_ZYX'])
                    writer.writerow(dataline['nodeA_ZYX_mm'])
                    writer.writerow(dataline['nodeB_ZYX'])
                    writer.writerow(dataline['nodeB_ZYX_mm'])
                    writer.writerow([dataline['nodeIdA']])
                    writer.writerow([dataline['nodeIdB']])
                    writer.writerow([dataline['nodesDistance']])
                    writer.writerow([dataline['radius_mm']])
                    writer.writerow([dataline['tortuosity']])
                    writer.writerow(dataline['vectorA'])
                    writer.writerow(dataline['vectorB'])

                except Exception, e:
                    logger.error('Error when saving line '+str(lineid)+' (data) to csv: '+str(e))
                    logger.error(traceback.format_exc())


    def writeSkeletonToPickle(self, filename='skel.pkl'):
        misc.obj_to_file(self.sklabel, filename=filename, filetype='pickle')


def generate_sample_data(m=1, noise_level=0.005, gauss_sigma=0.1):
    """
    Generate sample vessel system.

    J. Kunes

    | Input:
    |     m - output will be (100*m)^3 numpy array
    |     noise_level - noise power, disable noise with -1
    |     gauss_sigma - gauss filter sigma, disable filter with -1

    | Output:
    |     (100*m)^3 numpy array
    |         voxel size = [1,1,1]
    """
    import thresholding_functions

    data3d = np.zeros((100 * m, 100 * m, 100 * m), dtype=np.int)

    # size 8
    data3d_new = np.ones((100 * m, 100 * m, 100 * m), dtype=np.bool)
    data3d_new[0:30 * m, 20 * m, 20 * m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 8 * m] = 0
    data3d[data3d_new == 0] = 1
    # size 7
    data3d_new = np.ones((100 * m, 100 * m, 100 * m), dtype=np.bool)
    data3d_new[31 * m:70 * m, 20 * m, 20 * m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 7 * m] = 0
    data3d[data3d_new == 0] = 1
    # size 6
    data3d_new = np.ones((100 * m, 100 * m, 100 * m), dtype=np.bool)
    data3d_new[70 * m, 20 * m:50 * m, 20 * m] = 0
    data3d_new[31 * m, 20 * m, 20 * m:70 * m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 6 * m] = 0
    data3d[data3d_new == 0] = 1
    # size 5
    data3d_new = np.ones((100 * m, 100 * m, 100 * m), dtype=np.bool)
    data3d_new[70 * m:95 * m, 20 * m, 20 * m] = 0
    data3d_new[31 * m:60 * m, 20 * m, 70 * m] = 0
    data3d_new[70 * m:90 * m, 50 * m, 20 * m] = 0
    data3d_new[70 * m, 50 * m, 20 * m:50 * m] = 0
    data3d_new[31 * m, 20 * m: 45 * m, 20 * m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 5*m] = 0
    data3d[data3d_new == 0] = 1
    # size 4
    data3d_new = np.ones((100*m, 100*m, 100*m), dtype=np.bool)
    data3d_new[31*m, 20*m:50*m, 70*m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 4*m] = 0
    data3d[data3d_new == 0] = 1
    # size 3
    data3d_new = np.ones((100*m, 100*m, 100*m), dtype=np.bool)
    data3d_new[31*m:50*m, 50*m, 70*m] = 0
    data3d_new[31*m:50*m, 45*m, 20*m] = 0
    data3d_new[70*m, 50*m:70*m, 50*m] = 0
    data3d_new[70*m:80*m, 50*m, 50*m] = 0
    data3d_new[scipy.ndimage.distance_transform_edt(data3d_new) <= 3*m] = 0
    data3d[data3d_new == 0] = 1

    data3d = data3d*3030   # 3030+5920 = vessel value
    data3d += 5920         # 5920 = background value

    if gauss_sigma > 0:
        sigma = np.round(gauss_sigma, 2)
        sigmaNew = thresholding_functions.calculateSigma([1, 1, 1], sigma)
        data3d = thresholding_functions.gaussFilter(data3d, sigmaNew)

    if noise_level > 0:
        noise = np.random.normal(1, noise_level, (100*m, 100*m, 100*m))
        data3d = data3d*noise

    return data3d


def parser_init():  # pragma: no cover
    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser'
    )

    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        help='Input file/directory. Generates sample data, if not set.')

    parser.add_argument(
        '-vs', '--voxelsize',
        default=None,
        type=float, metavar='N', nargs='+',
        help='Size of one voxel. Format: "Z Y X"')

    parser.add_argument(
        '--binaryclosing', type=int,
        default=-1, # in HystologyAnalyser.__init__()
        help='Sets number of binary closing operations. Only in --nogui mode')
        
    parser.add_argument(
        '--binaryopening', type=int,
        default=-1, # in HystologyAnalyser.__init__()
        help='Sets number of binary opening operations. Only in --nogui mode')
        
    parser.add_argument(
        '-t', '--threshold', type=int,
        default=-1,
        help='Segmentation threshold. Default -1 (GUI/Automatic selection)')

    parser.add_argument(
        '-m', '--maskfile',
        default=None,
        help='Input file with mask data. Must have same dimensions as data after crop.')
    
    parser.add_argument(
        '--savemask',
        action='store_true',
        help='Saves used mask to mask.pkl file. Only in GUI mode')

    parser.add_argument(
        '--aggregatenearnodes', type=np.double,
        default=0, 
        help='aggregate_near_nodes_distance value')
     
    parser.add_argument(
        '-cr', '--crop',
        default=None,
        type=int, metavar='N', nargs='+',
        help='Crops input data. In GUI mode, crops before GUI crop. Default is\
None. Format: "z1 z2 y1 y2 x1 x2". -1 = None (start or end of axis).')

    parser.add_argument(
        '--nogui',
        action='store_true',
        help='Disable GUI, requires threshold argument')

    parser.add_argument(
        '-vte', '--vt2esofspy',
        default="tracer.txt",
        help='name of output vessel tree export to esofspy file')


    parser.add_argument(
        '-vtf', '--vtfile',
        default=None,
        help='name of output vessel tree file')


    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Debug mode')



    args = parser.parse_args()

    # replaces -1 in crop with None
    if args.crop is not None:
        args.crop = [None if x == -1 else x for x in args.crop]

    return args


def processData(args):
    """
    Processing data without gui
    """
    inputfile=args.inputfile
    threshold=args.threshold
    mask_file=args.maskfile
    crop=args.crop
    voxelsize=args.voxelsize
    binaryClosing=args.binaryopening
    binaryOpening=args.binaryclosing
    vtfile=args.vtfile
    vt2esofspy=args.vt2esofspy
    
    # Reading/Generating data
    if inputfile is None:  # # Using generated sample data
        logger.info('Generating sample data...')
        metadata = {'voxelsize_mm': [1, 1, 1]}
        data3d = generate_sample_data(1, 0, 0)
    else:  # Normal runtime
        dr = datareader.DataReader()
        data3d, metadata = dr.Get3DData(inputfile)

    # Custom voxel size
    if voxelsize is not None:
        metadata['voxelsize_mm'] = voxelsize

    # Crop data
    if crop is not None:
        logger.debug('Croping data: %s', str(crop))
        data3d = data3d[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]].copy()

    # Init HistologyAnalyser object
    logger.debug('Init HistologyAnalyser object')
    ha = HistologyAnalyser(data3d, metadata, threshold, 
             binaryClosing=binaryClosing, binaryOpening=binaryOpening, 
             nogui=True, aggregate_near_nodes_distance=args.aggregatenearnodes)

    # Remove Area = Load mask from file
    if mask_file is not None:
        logger.debug('Loading mask from file...')
        mask = misc.obj_from_file(filename=mask_file, filetype='pickle')
        if ha.data3d.shape == mask.shape:
            ha.data3d_masked = mask
            ha.data3d[mask == 0] = np.min(ha.data3d)
        else:
            raise ValueError('Mask file has wrong dimensions '+str(mask.shape))
    
    # Segmentation
    logger.debug('Segmentation')
    ha.data_to_skeleton()

    # Computing statistics
    logger.info("# ## ## ## ## statistics")
    ha.data_to_statistics()

    # Saving files
    logger.info("# ## ## write stats to file")
    ha.writeStatsToCSV()
    ha.writeStatsToYAML(vtfile)
    if vt2esofspy is not None:
        ha.exportVT2esofspy(vt2esofspy)

    # ## Histology report
    logger.info("# ## ## Histology report")
    hr = HistologyReport()
    hr.data = ha.stats
    hr.generateStats()
    hr.writeReportToCSV()
    hr.writeReportToYAML()

    
    # Add results Record
    if crop is not None:
        label = str(crop[0])+"-"+str(crop[1])
    else:
        label = "0-end"
    if inputfile is None:
        hr.addResultsRecord(label=label)
    else:
        hr.addResultsRecord(label=label, datapath=inputfile)

    # ## End
    logger.info('Finished')


def main():  # pragma: no cover
    args = parser_init()

    logging.basicConfig(stream=sys.stdout)
    logger = logging.getLogger()
    
    logger.setLevel(logging.WARNING)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        
    try:
        os.remove("ha.log")
    except OSError:
        pass
    fh = logging.FileHandler("ha.log")
    fh.setFormatter(logging.Formatter(fmt='%(levelname)s:%(name)s:%(message)s'))
    logger.addHandler(fh)

    logger.info('Input file -> %s', args.inputfile)
    logger.info('Data crop -> %s', str(args.crop))
    logger.info('Threshold -> %s', args.threshold)

    if args.nogui:
        logger.info('Running without GUI')
        processData(args)
    else:
        app = QApplication(sys.argv)
        HA_GUI.HistologyAnalyserWindow(
            inputfile=args.inputfile,
            voxelsize=args.voxelsize,
            crop=args.crop,
            args=args, 
            qapp=app
        )
        sys.exit(app.exec_())

if __name__ == "__main__":  # pragma: no cover
    main()
