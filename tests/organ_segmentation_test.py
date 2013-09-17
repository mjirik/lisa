#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src/"))
sys.path.append(os.path.join(path_to_script, "../extern/py3DSeedEditor/"))
sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest

import numpy as np


import organ_segmentation
import dcmreaddata as dcmr


#  nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams


class OrganSegmentationTest(unittest.TestCase):
    interactiveTest = False
    verbose = False

    def generate_data(self):

        img3d = (np.random.rand(30,30,30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation [10:25,4:24,2:16] = 1
        img3d = img3d + segmentation*20
        seeds[12:18,9:16, 3:6] = 1
        seeds[19:22,21:27, 19:21] = 2

        voxelsize_mm = [5,5,5]
        metadata = {'voxelsize_mm': voxelsize_mm}
        return img3d, metadata, seeds, segmentation

    @unittest.skipIf(not interactiveTest, "interactive test")
    def test_viewer_seeds(self):

        from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        import numpy as np
        img3d = (np.random.rand(30,30,30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        seeds[12:18,9:16, 3:6] = 1
        seeds[19:22,21:27, 3:6] = 2
#, QMainWindow
        app = QApplication(sys.argv)
        pyed = QTSeedEditor(img3d, seeds=seeds)
        pyed.exec_()


        deletemask = pyed.getSeeds()
        #import pdb; pdb.set_trace()

        
        #pyed = QTSeedEditor(deletemask, mode='draw')
        #pyed.exec_()

        app.exit()
    #@unittest.skip("demonstrating skipping")
    @unittest.skipIf(not interactiveTest, "interactive test")
    def test_whole_organ_segmentation_interactive(self):
        """
        Interactive test uses dicom data for segmentation
        """
        dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
        oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4)
        
# manual seeds setting
        print ("with left mouse button select some pixels of the brain")
        print ("with right mouse button select some pixels of other tissues and background")

        oseg.interactivity()

        volume = oseg.get_segmented_volume_size_mm3()
        print volume

        self.assertGreater(volume, 50000)
        self.assertLess(volume, 1200000)


#        roi_mm = [[3,3,3],[150,150,50]]
#        oseg.ni_set_roi()
#        coordinates_mm = [[110,50,30], [10,10,10]]
#        label = [1,2]
#        radius = [5,5]
#        oseg.ni_set_seeds(coordinates_mm, label, radius)
#
#        oseg.make_segmentation()


# @TODO doladit boundary penalties
    @unittest.skipIf(not interactiveTest, "interactive test")
    def test_organ_segmentation_with_boundary_penalties(self):
        """
        Interactivity is stored to file
        """
        import misc
        dcmdir = os.path.join(path_to_script,'./../sample_data/jatra_5mm')
        
        #gcparams = {'pairwiseAlpha':10, 'use_boundary_penalties':True}
        segparams = {'pairwise_alpha_per':3, 'use_boundary_penalties':True,'boundary_penalties_sigma':200}
        oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4, segparams=segparams)
        oseg.add_seeds_mm([120],[120],[70], label=1, radius=30)
        oseg.add_seeds_mm([170,220,250],[250,280,200],[70], label=2, radius=30)

        "boundary penalties"
        oseg.interactivity()
        #oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()

        misc.obj_to_file(oseg.get_iparams(),'iparams.pkl',filetype='pickle')

        self.assertGreater(volume,1000000)
    #@unittest.skipIf(not interactiveTest, "interactive test")
    def test_create_iparams(self):
        """
        Interactivity is stored to file
        """
        if self.verbose:
            print "test_create_iparams"
        import misc
        dcmdir = os.path.join(path_to_script,'./../sample_data/jatra_5mm')
        
        segparams = {'pairwiseAlpha':20, 'use_boundary_penalties':False,'boundary_penalties_sigma':200}
        #oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4)
        oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4, segparams=segparams)
        #oseg.add_seeds_mm([120,160],[150,120],[70], label=1, radius=20)
        oseg.add_seeds_mm([120,160],[150,80],[85], label=1, radius=20)
        oseg.add_seeds_mm([170,220,250,100],[250,300,200,350],[85], label=2, radius=20)
        oseg.add_seeds_mm([170],[240],[70], label=2, radius=20)
        
        #print "test_ipars"
        #oseg.interactivity()
        oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()
        #print 'vol %.3g ' %(volume)

        misc.obj_to_file(oseg.get_iparams(),'iparams.pkl',filetype='pickle')

        self.assertGreater(volume,1000000)


    @unittest.skipIf(not interactiveTest, "interactive test")
    def test_stored_interactivity(self):
        pass

    def test_roi(self):
        """
        Test setting of ROI. It is in pixels, not in mm
        """

        img3d = (np.random.rand(30,30,30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation [10:25,4:24,2:16] = 1
        img3d = img3d + segmentation*20
        seeds[12:18,9:16, 3:6] = 1
        seeds[19:22,21:27, 19:21] = 2


        roi = [[7,27],[2,29],[0,26]]
        seeds = seeds[7:27, 2:29, 0:26]
        voxelsize_mm = [5,5,5]
        metadata = {'voxelsize_mm': voxelsize_mm}

        oseg = organ_segmentation.OrganSegmentation(None,
                data3d=img3d,
                metadata=metadata,
                seeds=seeds,
                roi=roi,
                working_voxelsize_mm=5)

        #oseg.interactivity(min_val=0, max_val=30)
        oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()
        self.assertGreater(volume,500000)

    def test_box_segmentation(self):
        """
        Function uses organ_segmentation  for synthetic box object 
        segmentation.
        """
        #dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
# data
        img3d = np.random.rand(64,64,32) * 5
        img3d[12:32,5:25,4:24] = img3d [12:32,5:25,4:24] + 15

#seeds
        seeds = np.zeros([64,64,32], np.int8)
        seeds [13:31,22:25,9:12] = 1
        seeds [6:9,3:32,9:12] = 2
#[mm]  10 x 10 x 10        #voxelsize_mm = [1,4,3]
        voxelsize_mm = [5,5,5]
        metadata = {'voxelsize_mm': voxelsize_mm}

        oseg = organ_segmentation.OrganSegmentation(None,\
                data3d=img3d, metadata = metadata, \
                seeds = seeds, \
                working_voxelsize_mm = 10)
        

        # oseg.seeds = seeds
        #oseg.make_gc()
# manual seeds setting
        #print ("with left mouse button select some pixels of the brain")
        #print ("with right mouse button select some pixels of other tissues and background")

        oseg.ninteractivity()
        #oseg.interactivity()

        volume = oseg.get_segmented_volume_size_mm3()
        
        #import pdb; pdb.set_trace()

        #mel by to být litr. tedy milion mm3
        self.assertGreater(volume,900000)
        self.assertLess(volume,1100000)

    def test_volume_resize(self):
        #from scipy.sparse.import lil_matrix

        pass


    #@unittest.skipIf(True,"interactive test")
    @unittest.skipIf(not interactiveTest, "interactive test")
    def test_vincentka_06_slice_thickness_interactive(self):
        """
        Interactive test. SliceThickness is not voxel depth. If it is, this 
        test will fail.
        """
        #dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
        dcmdir = os.path.expanduser('~/data/medical/data_orig/vincentka/13021610/10200000/')
        dcmdir = os.path.expanduser('~/data/medical/data_orig/vincentka/13021610/12460000/')
        oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4)
        
# manual seeds setting
        print ("with left mouse button select some pixels of the bottle content")
        print ("with right mouse button select some pixels of background")

        oseg.interactivity()

        volume = oseg.get_segmented_volume_size_mm3()
        #print volume

        self.assertGreater(volume,550000)
        self.assertLess(volume,850000)

    def setUp(self):
        """ Nastavení společných proměnných pro testy  """
        self.assertTrue(True)


    # @TODO dodělat přidávání uzlů pomocí mm
    #@unittest.skipIf(not interactiveTest, "interactive test")
    def test_add_seeds_mm(self):
        """
        Function uses organ_segmentation object for segmentation
        """
        dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
        oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4)

        oseg.add_seeds_mm([120],[120],[60], 1, 25)
        oseg.add_seeds_mm([25],[100],[60], 2, 25)

        # pro kontrolu lze odkomentovat
        #oseg.interactivity()

        oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()
        
        #import pdb; pdb.set_trace()

        #mel by to být litr. tedy milion mm3
        self.assertGreater(volume,800000)
        self.assertLess(volume,1200000)

        #roi_mm = [[3,3,3],[150,150,50]]
        #oseg.ni_set_roi()
        #coordinates_mm = [[110,50,30], [10,10,10]]
        #label = [1,2]
        #radius = [5,5]
        #oseg.ni_set_seeds(coordinates_mm, label, radius)

        #oseg.make_segmentation()

        #oseg.noninteractivity()
        pass

    @unittest.skip("demonstrating skipping")
    def test_dicomread_and_graphcut(self):
        """
        Test dicomread module and graphcut module
        """
        dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/')
        data3d, metadata = dcmr.dcm_read_from_dir(dcmdir)

        #print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

        igc = pycut.ImageGraphCut(data3d, zoom = 0.5)
        seeds = igc.seeds
        seeds[0,:,0] = 1
        seeds[60:66,60:66,5:6] = 2
        igc.noninteractivity(seeds)


        igc.make_gc()
        segmentation = igc.segmentation
        self.assertTrue(segmentation[14, 4, 1] == 0)
        self.assertTrue(segmentation[127, 120, 10] == 1)
        self.assertTrue(np.sum(segmentation==1) > 100)
        self.assertTrue(np.sum(segmentation==0) > 100)
        #igc.show_segmentation()

if __name__ == "__main__":
    unittest.main()
