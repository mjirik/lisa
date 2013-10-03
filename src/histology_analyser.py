#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))
import logging
logger = logging.getLogger(__name__)

import argparse


import numpy as np
import misc
import datareader
import SimpleITK as sitk
import seed_editor_qt as seqt

class HistologyAnalyser:
    def __init__ (self, data3d, metadata, threshold):
        self.data3di = self.muxImage(
                data3d.astype(np.uint16),
                metadata
                )
        self.metadata = metadata
        self.threshold = threshold


    def thresh(self, thr):
        return (self.data3di > thr)

    def run(self):
        import pdb; pdb.set_trace()
        self.preprocessing()

        datathr = self.thresh(self.threshold)


        sitk.Show(datathr * 100)
        import pdb; pdb.set_trace()
        dataskel = sitk.BinaryThinning(datathr)
        sitk.Show(dataskel * 100)
        import pdb; pdb.set_trace()

        self.output = sitk.GetArrayFromImage(dataskel)
        import pdb; pdb.set_trace()

    def preprocessing(self):
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

    def show(self):
        app = QApplication(sys.argv)
        seqt.QTSeedEditor(self.output.astype(np.int16))
        app.exec_()





if __name__ == "__main__":
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
    ha.show()
    

