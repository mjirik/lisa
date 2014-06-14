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
import thresholding_functions

from seed_editor_qt import QTSeedEditor

GAUSSIAN_SIGMA = 1
fast_debug = False
#fast_debug = True

import histology_analyser_gui as HA_GUI
from skeleton_analyser import SkeletonAnalyser

class HistologyAnalyser:
    def __init__(self, data3d, metadata, threshold=-1, binaryClosing=1, binaryOpening=1, nogui=True):
        self.data3d = data3d
        self.nogui = nogui
        self.threshold = threshold
        self.binaryClosing = binaryClosing
        self.binaryOpening = binaryOpening

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
        ### Median filter
        filteredData = scipy.ndimage.filters.median_filter(self.data3d, size=2) 
        
        ### Segmentation
        data3d_thr = segmentation.vesselSegmentation(
            filteredData, #self.data3d,
            segmentation=np.ones(self.data3d.shape, dtype='int8'),
            threshold=self.threshold,
            inputSigma=0, #0.15,
            dilationIterations=2,
            nObj=1,
            biggestObjects= False,
            interactivity= not self.nogui,
            binaryClosingIterations=self.binaryClosing, #5,  # TODO !!! - vytvari na stranach oblasti ktere se pak nenasegmentuji
            binaryOpeningIterations=self.binaryOpening #1
            )
        
        ### Zalepeni der
        data3d_thr = scipy.ndimage.morphology.binary_fill_holes(data3d_thr)
        
        return data3d_thr

    def binar_to_skeleton(self, data3d_thr):
        data3d_thr = (data3d_thr > 0).astype(np.int8)
        data3d_skel = skelet3d.skelet3d(data3d_thr)
        return data3d_skel

    def data_to_skeleton(self):
        data3d_thr = self.data_to_binar()
        data3d_skel = self.binar_to_skeleton(data3d_thr)
        return data3d_thr, data3d_skel
    
    def skeleton_to_statistics(self, data3d_thr, data3d_skel, guiUpdateFunction=None):
        skan = SkeletonAnalyser(
            data3d_skel,
            volume_data=data3d_thr,
            voxelsize_mm=self.metadata['voxelsize_mm'])

        stats = skan.skeleton_analysis(guiUpdateFunction=guiUpdateFunction)
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
        logger.debug('writeStatsToYAML')
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

def generate_sample_data(m=1, noise_level=0.005, gauss_sigma=0.1):
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
    
    data3d = data3d*3030   # 3030+5920 = vessel value
    data3d += 5920         # 5920 = background value
    
    if gauss_sigma>0:
        sigma = np.round(gauss_sigma, 2)
        sigmaNew = thresholding_functions.calculateSigma([1,1,1], sigma)
        data3d = thresholding_functions.gaussFilter(data3d, sigmaNew)
    
    if noise_level>0:
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
        default=-1, 
        help='data threshold, default -1 (gui/automatic selection)')
    parser.add_argument(
        '-is', '--input_is_skeleton', action='store_true',
        help='Input file is .pkl file with skeleton')
    parser.add_argument('-cr', '--crop', type=int, metavar='N', nargs='+',
        default=None,
        help='Segmentation labels, default 1')
    parser.add_argument( # TODO - not needed??
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
            data3d = generate_sample_data(1)
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
        processData(inputfile=args.inputfile,
                    threshold=args.threshold,
                    skeleton=args.input_is_skeleton,
                    crop=args.crop)
    else:
        app = QApplication(sys.argv)
        gui = HA_GUI.HistologyAnalyserWindow(inputfile=args.inputfile,
                                            skeleton=args.input_is_skeleton,
                                            crop=args.crop,
                                            crgui=args.crgui)
        sys.exit(app.exec_())
        
if __name__ == "__main__":
    main()
