#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(path_to_script, "../extern/pycat/"))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)


#import apdb
#  apdb.set_trace();
#import scipy.io
import numpy as np
import scipy

# ----------------- my scripts --------
import py3DSeedEditor
import dcmreaddata1 as dcmr
import pycat
import argparse
#import py3DSeedEditor

import segmentation
import qmisc






class SupportStructureSegmentation():
    def __init__(self,  
            data3d = None, 
            voxelsize_mm = None, 
            autocrop = True, 
            autocrop_margin_mm = [10,10,10], 
            modality = 'CT',
            slab = {'none':0, 'bone':8,'lungs':9,'heart':10}
            ):
        """
        Segmentaton of support structures for liver segmentatio based on 
        location prior.
        """
        


        self.data3d = data3d
        self.voxelsize_mm = voxelsize_mm
        self.autocrop = autocrop
        self.autocrop_margin_mm = np.array(autocrop_margin_mm)
        self.autocrop_margin = self.autocrop_margin_mm/self.voxelsize_mm
        self.crinfo = [[0,-1],[0,-1],[0,-1]]
        self.segmentation = None
        self.slab = slab

        

        #import pdb; pdb.set_trace()


#    def 
    def import_data(self, data):
        self.data = data
        self.data3d = data['data3d']
        self.voxelsize_mm = data['voxelsize_mm']
    
    def import_dir(self, datadir):
        self.data3d, self.metadata = dcmr.dcm_read_from_dir(datadir)
        self.voxelsize_mm = np.array(self.metadata['voxelsizemm'])


    def bone_segmentation(self):
        self.segmentation = np.array(self.data3d > 1300).astype(np.int8)*self.slab['bone']
        pass

    def heart_segmentation(self):
        pass

    def lungs_segmentation(self):
        self.segmentation = np.zeros(self.data3d.shape)
        pass



    def _crop(self, data, crinfo):
        """
        Crop data with crinfo
        """
        data = data[crinfo[0][0]:crinfo[0][1], crinfo[1][0]:crinfo[1][1], crinfo[2][0]:crinfo[2][1]]
        return data


    def _crinfo_from_specific_data (self, data, margin):
# hledáme automatický ořez, nonzero dá indexy
        nzi = np.nonzero(data)

        x1 = np.min(nzi[0]) - margin[0]
        x2 = np.max(nzi[0]) + margin[0] + 1
        y1 = np.min(nzi[1]) - margin[0]
        y2 = np.max(nzi[1]) + margin[0] + 1
        z1 = np.min(nzi[2]) - margin[0]
        z2 = np.max(nzi[2]) + margin[0] + 1 

# ošetření mezí polí
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if z1 < 0:
            z1 = 0

        if x2 > data.shape[0]:
            x2 = data.shape[0]-1
        if y2 > data.shape[1]:
            y2 = data.shape[1]-1
        if z2 > data.shape[2]:
            z2 = data.shape[2]-1

# ořez
        crinfo = [[x1, x2],[y1,y2],[z1,z2]]
        #dataout = self._crop(data,crinfo)
        #dataout = data[x1:x2, y1:y2, z1:z2]
        return crinfo


    def im_crop(self, im,  roi_start, roi_stop):
        im_out = im[ \
                roi_start[0]:roi_stop[0],\
                roi_start[1]:roi_stop[1],\
                roi_start[2]:roi_stop[2],\
                ]
        return  im_out
    
    def export(self):
        slab={}
        slab['none'] = 0
        slab['liver'] = 1
        slab['lesions'] = 6

        data = {}
        data['version'] = (1,0,0)
        data['data3d'] = self.data3d
        data['crinfo'] = self.crinfo
        data['segmentation'] = self.segmentation
        data['slab'] = slab
        data['voxelsize_mm'] = self.voxelsize_mm
        #import pdb; pdb.set_trace()
        return data




    def visualization(self):
        """
        Run viewer with output data3d and segmentation
        """

        from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        import numpy as np
#, QMainWindow
        app = QApplication(sys.argv)
        #pyed = QTSeedEditor(self.data3d, contours=(self.segmentation>0))
        pyed = QTSeedEditor(self.segmentation)
        pyed.exec_()


        #import pdb; pdb.set_trace()

        
        #pyed = QTSeedEditor(deletemask, mode='draw')
        #pyed.exec_()

        app.exit()
        





def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description=
            'Segmentation of bones, lungs and heart.')
    parser.add_argument('-dd','--dcmdir',
            default=None,
            help='path to data dir')
    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-exd', '--exampledata', action='store_true', 
            help='run unittest')
    parser.add_argument('-so', '--show_output', action='store_true', 
            help='Show output data in viewer')
    args = parser.parse_args()



    if args.debug:
        logger.setLevel(logging.DEBUG)


    if args.exampledata:

        args.dcmdir = '../sample_data/matlab/examples/sample_data/DICOM/digest_article/'
        
#    if dcmdir == None:

    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
    data3d, metadata = dcmr.dcm_read_from_dir(args.dcmdir)


    sseg = SupportStructureSegmentation(data3d = data3d, 
            voxelsize_mm = metadata['voxelsizemm'], 
            )

    sseg.bone_segmentation()
    

    #print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    #igc = pycat.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()


    #igc.make_gc()
    #igc.show_segmentation()

    # volume 
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

    
    #pyed = py3DSeedEditor.py3DSeedEditor(oseg.data3d, contour = oseg.segmentation)
    #pyed.show()

    if args.show_output:
        sseg.show_output()

    savestring = raw_input ('Save output data? (y/n): ')
    #sn = int(snstring)
    if savestring in ['Y','y']:
        import misc

        data = sseg.export()

        misc.obj_to_file(data, "organ.pickle", filetype = 'pickle')
    #output = segmentation.vesselSegmentation(oseg.data3d, oseg.orig_segmentation)
    

if __name__ == "__main__":
    main()
