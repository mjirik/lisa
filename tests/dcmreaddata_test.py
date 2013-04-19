#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/pycat/extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest


from PyQt4.QtGui import QFileDialog, QApplication, QMainWindow

#import numpy as np


#import dcmreaddata1 as dcmr
import dcmreaddata as dcmr

class DicomReaderTest(unittest.TestCase):
    interactivetTest = False
#    def setUp(self):
#        #self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_06mm_jenjatraplus/')
#        self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
#        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
#        reader = dcmr.DicomReader(self.dcmdir)
#        self.data3d = reader.get_3Ddata()
#        self.metadata = reader.get_metaData()

    def test_read_volumetry_overlay_with_dicom_module(self):
        import dicom
        dcmfile = os.path.join(path_to_script, '../sample_data/volumetrie/volumetry_slice.DCM')
        data = dicom.read_file(dcmfile)
        data3d = data.pixel_array

        
        overlay = data[0x6000,0x3000]
        import pdb; pdb.set_trace()
        pol = []
        for i in range(1,len(overlay)):
            pass
            #pol[i] = ord(

# 168,168 odpovida v mm cca 130,130, voxelsizemm je 0.7734
        x = 168
        y = 168

        # index pixelu
        k = (512*x) + y
        value1 = data.pixel_array.flat[k]
        byte1 = ord(data.PixelData[k*2+1])
        byte2 = ord(data.PixelData[k*2+0])
        value2 = byte2*256+byte1


        # (168*512)+168;ord(pxdata[(k*2)-1]),ord(pxdata[k*2]), ord(pxdata[(k*2)+1]), data.pixel_array.flat[k]
        



    def test_dcmread(self):

        dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
        #dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
        reader = dcmr.DicomReader(dcmdir)
        data3d = reader.get_3Ddata()
        metadata = reader.get_metaData()
#slice size is 512x512
        self.assertEqual(data3d.shape[0],512)
# voxelsize depth = 5 mm
        self.assertEqual(metadata['voxelsizemm'][2],5)

    def test_dcmread_series_number(self):

        dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
        #dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
# spravne cislo serie je 7
        reader =  dcmr.DicomReader(dcmdir,series_number = 7)
        data3d = reader.get_3Ddata()
        metadata = reader.get_metaData()
        self.assertEqual(data3d.shape[0],512)
        self.assertEqual(metadata['voxelsizemm'][2],5)

    @unittest.skipIf(not interactivetTest, 'interactiveTest')
    def test_dcmread_select_series(self):

        #dirpath = dcmr.get_dcmdir_qt()
        dirpath = '/home/mjirik/data/medical/data_orig/46328096/'
        #dirpath = dcmr.get_dcmdir_qt()
        #app = QMainWindow()
        reader = dcmr.DicomReader(dirpath, series_number = 55555)#, #qt_app =app)
        #app.exit()
        self.data3d = reader.get_3Ddata()
        self.metadata = reader.get_metaData()

    #@unittest.skipIf(not interactivetTest, 'interactiveTest')
    @unittest.skip('skip')
    def test_dcmread_get_dcmdir_qt(self):

        dirpath = dcmr.get_dcmdir_qt()
        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
        reader = dcmr.DicomReader(dirpath)
        self.data3d = reader.get_3Ddata()
        self.metadata = reader.get_metaData()

        #sss.visualization()
        #import pdb; pdb.set_trace()


#    def test_synthetic_data_lesions_automatic_localization(self):
#        """
#        Function uses lesions  automatic localization in synthetic data.
#        """
#        #dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
## data
#        slab = {'none':0, 'liver':1, 'porta':2, 'lesions':6}
#        voxelsize_mm = np.array([1.0,1.0,1.2])
#
#        segm = np.zeros([256,256,80], dtype=np.int16)
#
#        # liver
#        segm[70:190,40:220,30:60] = slab['liver']
## port
#        segm[120:130,70:220,40:45] = slab['porta']
#        segm[80:130,100:110,40:45] = slab['porta']
#        segm[120:170,130:135,40:44] = slab['porta']
#
#        # vytvoření kopie segmentace - před určením lézí
#        segm_pre = copy.copy(segm)
#
#        segm[150:180,70:105,42:55] = slab['lesions']
#
#
#        data3d = np.zeros(segm.shape)
#        data3d[segm== slab['none']] = 680
#        data3d[segm== slab['liver']] = 1180
#        data3d[segm== slab['porta']] = 1230
#        data3d[segm== slab['lesions']] = 1110
#        #noise = (np.random.rand(segm.shape[0], segm.shape[1], segm.shape[2])*30).astype(np.int16)
#        noise = (np.random.normal(0,30,segm.shape))#.astype(np.int16)
#        data3d = (data3d + noise  ).astype(np.int16)
#
#
#        data={'data3d':data3d, 
#                'slab':slab,
#                'voxelsize_mm':voxelsize_mm,
#                'segmentation':segm_pre
#                }
#
## @TODO je tam bug, prohlížeč neumí korektně pracovat s doubly 
##        app = QApplication(sys.argv)
##        #pyed = QTSeedEditor(noise )
##        pyed = QTSeedEditor(data3d)
##        pyed.exec_()
##        #img3d = np.zeros([256,256,80], dtype=np.int16)
#        
#       # pyed = py3DSeedEditor.py3DSeedEditor(data3d)
#       # pyed.show()
#
#        tumory = lesions.Lesions()
#
#        tumory.import_data(data)
#        tumory.automatic_localization()
#        #tumory.visualization()
#         
#
#
## ověření výsledku
#        #pyed = py3DSeedEditor.py3DSeedEditor(outputTmp, contour=segm==slab['porta'])
#        #pyed.show()
#
#        errim = np.abs(
#                (tumory.segmentation == slab['lesions']).astype(np.int) - 
#                (segm == slab['lesions']).astype(np.int))
#
## ověření výsledku
#        #pyed = py3DSeedEditor.py3DSeedEditor(errim, contour=segm==slab['porta'])
#        #pyed.show()
##evaluation
#        sum_of_wrong_voxels = np.sum(errim)
#        sum_of_voxels = np.prod(segm.shape)
#
#        #print "wrong ", sum_of_wrong_voxels
#        #print "voxels", sum_of_voxels
#
#        errorrate = sum_of_wrong_voxels/sum_of_voxels
#
#
#        self.assertLess(errorrate,0.1)
#        self.assertLess(errorrate,0.1)
#
#
#

if __name__ == "__main__":
    unittest.main()
