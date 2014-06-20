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

import logging
logger = logging.getLogger(__name__)

import argparse

from PyQt4.QtGui import QApplication

import numpy as np
import scipy.ndimage
import misc
import datareader
import csv

# import seed_editor_qt as seqt
import skelet3d
import segmentation
import py3DSeedEditor as se

import histology_analyser_gui as HA_GUI
from skeleton_analyser import SkeletonAnalyser


class HistologyAnalyser:

    def __init__(self, data3d, metadata, threshold=-1, binaryClosing=1,
                 binaryOpening=1, nogui=True):
        self.data3d = data3d
        if 'voxelsize_mm' not in metadata.keys():
            metadata['voxelsize_mm'] = [1, 1, 1]
        self.metadata = metadata

        self.data3d_thr = None
        self.data3d_skel = None

        self.threshold = threshold
        self.binaryClosing = binaryClosing
        self.binaryOpening = binaryOpening

        self.nogui = nogui

    def get_voxelsize(self):
        return self.metadata['voxelsize_mm']

    def get_data3d(self):
        return self.data3d

    def get_data3d_segmented(self):
        return self.data3d_thr

    def get_data3d_skeleton(self):
        return self.data3d_skel

    def data_to_binar(self):
        # ## Median filter
        filteredData = scipy.ndimage.filters.median_filter(self.data3d, size=2)

        # ## Segmentation
        data3d_thr = segmentation.vesselSegmentation(
            filteredData,  # self.data3d,
            segmentation=np.ones(self.data3d.shape, dtype='int8'),
            threshold=self.threshold,
            inputSigma=0,  # 0.15,
            dilationIterations=2,
            nObj=1,
            biggestObjects=False,
            interactivity=not self.nogui,
            binaryClosingIterations=self.binaryClosing, # noqa 5,  # TODO !!! - vytvari na stranach oblasti ktere se pak nenasegmentuji
            binaryOpeningIterations=self.binaryOpening # 1 # noqa
            )
        del(filteredData)

        # ## Zalepeni der
        scipy.ndimage.morphology.binary_fill_holes(data3d_thr,
                                                   output=data3d_thr)

        self.data3d_thr = data3d_thr

    def binar_to_skeleton(self):
        self.data3d_skel = skelet3d.skelet3d(
            (self.data3d_thr > 0).astype(np.int8)
        )

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
        vs = self.metadata['voxelsize_mm']
        voxel_volume_mm3 = vs[0]*vs[1]*vs[2]
        volume_px = self.data3d.shape[0] \
            * self.data3d.shape[1] * self.data3d.shape[2]
        volume_mm3 = volume_px*voxel_volume_mm3
        vessel_volume_fraction = float(np.sum(np.sum(np.sum(
            self.data3d_thr)))) / float(volume_px)
        info = {
            'voxel_size_mm': list(vs),
            'voxel_volume_mm3': float(voxel_volume_mm3),
            'shape_px': list(self.data3d.shape),
            'volume_px': float(volume_px),
            'volume_mm3': float(volume_mm3),
            'vessel_volume_fraction': float(vessel_volume_fraction)
        }
        self.stats.update({'General': info})

        # # process skeleton to statistics
        logger.debug('Computing skeleton to statistics...')
        skan = SkeletonAnalyser(
            self.data3d_skel,
            volume_data=self.data3d_thr,
            voxelsize_mm=self.metadata['voxelsize_mm']
            )
        stats = skan.skeleton_analysis(guiUpdateFunction=guiUpdateFunction)
        # needed only by self.writeSkeletonToPickle()
        self.sklabel = skan.sklabel
        self.stats.update({'Graph': stats})

    def showSegmentedData(self):
        skan = SkeletonAnalyser(
            self.data3d_skel,
            volume_data=self.data3d_thr,
            voxelsize_mm=self.metadata['voxelsize_mm'])
        data3d_nodes_vis = skan.sklabel.copy()
        del(skan)

        # edges
        data3d_nodes_vis[data3d_nodes_vis > 0] = 1
        # nodes and terminals
        data3d_nodes_vis[data3d_nodes_vis < 0] = 2

        if not self.nogui:
            pyed = se.py3DSeedEditor(
                self.data3d,
                seeds=(data3d_nodes_vis).astype(np.int8),
                contour=self.data3d_thr.astype(np.int8)
            )
            pyed.show()

    def writeStatsToYAML(self, filename='hist_stats.yaml'):
        logger.debug('writeStatsToYAML')
        misc.obj_to_file(self.stats, filename=filename, filetype='yaml')

    # TODO - fix this to save everything
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

    def writeSkeletonToPickle(self, filename='skel.pkl'):
        misc.obj_to_file(self.sklabel, filename=filename, filetype='pickle')

    def __dataToCSVLine(self, dataline):  # TODO - finish this
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

# TODO - include this in generate_sample_data()
# def muxImage(self, data3d, metadata):
    # import SimpleITK as sitk
    # data3di = sitk.GetImageFromArray(data3d)
    # data3di.SetSpacing(metadata['voxelsize_mm'])

    # return data3di


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


def parser_init():
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
        '-t', '--threshold', type=int,
        default=-1,
        help='Segmentation threshold. Default -1 (GUI/Automatic selection)')

    parser.add_argument(
        '-is', '--input_is_skeleton', action='store_true',
        help='Input file is .pkl file with skeleton')

    parser.add_argument(
        '-cr', '--crop',
        default=None,
        type=int, metavar='N', nargs='+',
        help='Crops input data. In GUI mode, crops before GUI crop. Default is\
None. Format: "z1 z2 y1 y2 x1 x2". -1 = None (start or end of axis).')

    parser.add_argument(
        '--nogui',
        action='store_true',
        help='Disable GUI')

    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Debug mode')

    args = parser.parse_args()

    # replaces -1 in crop with None
    if args.crop is not None:
        args.crop = [None if x == -1 else x for x in args.crop]

    return args


def processData(inputfile=None, threshold=None, skeleton=False,
                crop=None, voxelsize=None):
    # Processing data without gui
    if skeleton:  # when input is just skeleton
        # TODO - test if works or just delete
        logger.info("input is skeleton")
        struct = misc.obj_from_file(filename='tmp0.pkl', filetype='pickle')
        data3d = struct['data3d']
        metadata = struct['metadata']
        ha = HistologyAnalyser(data3d, metadata, threshold, nogui=True)
        ha.data3d_skel = struct['skel']
        ha.data3d_thr = struct['thr']
        logger.info("end of is skeleton")
    else:
        # ## Reading/Generating data
        if inputfile is None:  # # Using generated sample data
            logger.info('Generating sample data...')
            metadata = {'voxelsize_mm': [1, 1, 1]}
            data3d = generate_sample_data(1, 0, 0)
        else:  # # Normal runtime
            dr = datareader.DataReader()
            data3d, metadata = dr.Get3DData(inputfile)

        # ## Custom voxel size
        if voxelsize is not None:
            metadata['voxelsize_mm'] = voxelsize

        # ## Crop data
        if crop is not None:
            logger.debug('Croping data: %s', str(crop))
            data3d = data3d[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]

        # ## Init HistologyAnalyser object
        logger.debug('Init HistologyAnalyser object')
        ha = HistologyAnalyser(data3d, metadata, threshold, nogui=True)

        # ## No GUI == No Remove Area

        # ## Segmentation
        logger.debug('Segmentation')
        ha.data_to_skeleton()

    # ## Computing statistics
    logger.info("# ## ## ## ## statistics")
    ha.data_to_statistics()

    # ## Saving files
    logger.info("# ## ## write to file")
    ha.writeStatsToCSV()
    ha.writeStatsToYAML()
    ha.writeSkeletonToPickle('skel.pkl')
    # struct = {'skel': data3d_skel, 'thr': data3d_thr, 'data3d': data3d,
    # 'metadata':metadata}
    # misc.obj_to_file(struct, filename='tmp0.pkl', filetype='pickle')

    # ## End
    logger.info('Finished')


def main():
    args = parser_init()

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    # ch = logging.StreamHandler()
    # https://docs.python.org/2/howto/logging.html# configuring-logging
    # logger.addHandler(ch)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info('Input file -> %s', args.inputfile)
    logger.info('Data crop -> %s', str(args.crop))
    logger.info('Threshold -> %s', args.threshold)

    if args.nogui:
        logger.info('Running without GUI')
        processData(inputfile=args.inputfile,
                    threshold=args.threshold,
                    skeleton=args.input_is_skeleton,
                    crop=args.crop,
                    voxelsize=args.voxelsize
                    )
    else:
        app = QApplication(sys.argv)
        # gui =
        HA_GUI.HistologyAnalyserWindow(
            inputfile=args.inputfile,
            skeleton=args.input_is_skeleton,
            crop=args.crop,
            voxelsize=args.voxelsize
        )
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()
