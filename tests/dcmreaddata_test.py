#! /usr/bin/python
# -*- coding: utf-8 -*-


import logging
logger = logging.getLogger(__name__)

# import funkcí z jiného adresáře
import sys
import os.path
import copy

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest


from PyQt4.QtGui import QFileDialog, QApplication, QMainWindow

import numpy as np

try:
    import pydicom
except ImportError as e:
    import dicom as pydicom
    logger.warning("Used dicom instead of pydicom")
pydicom.config.debug(False)

#
import io3d
import io3d.dcmreaddata as dcmr
import lisa.dataset

class DicomReaderTest(unittest.TestCase):
    interactivetTest = False
#    def setUp(self):
#        #self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_06mm_jenjatraplus/')
#        self.dcmdir = os.path.join(path_to_script, '../sample_data/jatra_5mm')
#        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
#        reader = dcmr.DicomReader(self.dcmdir)
#        self.data3d = reader.get_3Ddata()
#        self.metadata = reader.get_metaData()

    def test_DicomReader_overlay(self):
        import os.path as op
        # sample_data_path = "~/data/medical/orig/sample_data/"
        # sample_data_path = op.expanduser(sample_data_path)
        # sample_data_path = io3d.datasets.join_path("medical/orig/sample_data/", get_root=True)
        sample_data_path = io3d.datasets.join_path("medical", "orig", get_root=True)
        #import matplotlib.pyplot as plt

        # dcmdir = lisa.dataset.join_sdp('volumetrie/')
        dcmdir = os.path.join(sample_data_path, 'volumetrie')
        # dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
        reader = dcmr.DicomReader(dcmdir)
        overlay = reader.get_overlay()
        #import pdb; pdb.set_trace()
        #plt.imshow(overlay[1][:,:,0])
        #plt.show()

        self.assertEqual(overlay[1][0, 200, 200], 1)
        self.assertEqual(overlay[1][0, 100, 100], 0)

    def test_read_volumetry_overlay_with_dicom_module(self):
        """
        pydicom module is used for load dicom data. Dicom overlay
        is saved on (60xx,3000) bit after bit. Data are decoded and
        each bit is stored as array element.
        """
        # import dicom
        # import sed3
        #import matplotlib.pyplot as plt
        dcmfile = lisa.dataset.join_sdp('volumetrie/volumetry_slice.DCM')
        data = pydicom.read_file(dcmfile)



        # overlay index
        i_overlay = 1
        n_bits = 8


        # On (60xx,3000) are stored ovelays.
        # First is (6000,3000), second (6002,3000), third (6004,3000),
        # and so on.
        dicom_tag1 = 0x6000 + 2*i_overlay

        overlay_raw = data[dicom_tag1, 0x3000].value

        # On (60xx,0010) and (60xx,0011) is stored overlay size
        rows = data[dicom_tag1, 0x0010].value # rows = 512
        cols = data[dicom_tag1, 0x0011].value # cols = 512

        decoded_linear = np.zeros(len(overlay_raw)*n_bits)

        # Decoding data. Each bit is stored as array element
        for i in range(1,len(overlay_raw)):
            for k in range (0, n_bits):
                one_byte = overlay_raw[i]
                if sys.version_info.major == 2:
                    byte_as_int = ord(one_byte)
                else:
                    byte_as_int = one_byte
                decoded_linear[i * n_bits + k] = (byte_as_int >> k) & 0b1

        #overlay = np.array(pol)

        overlay = np.reshape(decoded_linear,[rows,cols])

        #plt.imshow(overlay)
        #plt.show()

        self. assertEqual(overlay[200,200],1)
        self. assertEqual(overlay[100,100],0)
        #pyed = sed3.sed3(overlay)
        #pyed.show()
        #import pdb; pdb.set_trace()







    def test_dcmread(self):

        dcmdir = lisa.dataset.join_sdp('jatra_5mm')
        #dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
        reader = dcmr.DicomReader(dcmdir)
        data3d = reader.get_3Ddata()
        metadata = reader.get_metaData()
#slice size is 512x512
        self.assertEqual(data3d.shape[2],512)
# voxelsize depth = 5 mm
        self.assertEqual(metadata['voxelsize_mm'][0],5)

    def test_dicomread_read(self):
        dcmdir = lisa.dataset.join_sdp('jatra_5mm')
        #dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
        data3d, metadata = io3d.datareader.read(dcmdir, dataplus_format=False)
#slice size is 512x512
        self.assertEqual(data3d.shape[2],512)
# voxelsize depth = 5 mm
        self.assertEqual(metadata['voxelsize_mm'][0],5)

    def test_dcmread_series_number(self):

        dcmdir = lisa.dataset.join_sdp('jatra_5mm')
        #dcmdir = '/home/mjirik/data/medical/data_orig/jatra-kma/jatra_5mm/'
        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(self.dcmdir)
# spravne cislo serie je 7
        reader =  dcmr.DicomReader(dcmdir,series_number = 7)
        data3d = reader.get_3Ddata()
        metadata = reader.get_metaData()
        self.assertEqual(data3d.shape[2],512)
        self.assertEqual(metadata['voxelsize_mm'][0],5)

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
#       # pyed = sed3.sed3(data3d)
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
#        #pyed = sed3.sed3(outputTmp, contour=segm==slab['porta'])
#        #pyed.show()
#
#        errim = np.abs(
#                (tumory.segmentation == slab['lesions']).astype(np.int) -
#                (segm == slab['lesions']).astype(np.int))
#
## ověření výsledku
#        #pyed = sed3.sed3(errim, contour=segm==slab['porta'])
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
