# ! /usr/bin/python
# -*- coding: utf-8 -*-


# import funkcí z jiného adresáře
import sys
import os.path

# path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest

import numpy as np
from nose.plugins.attrib import attr


from lisa import organ_segmentation
import pysegbase.dcmreaddata as dcmr
import lisa.dataset


# nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams # noqa


class OrganSegmentationTest(unittest.TestCase):
    interactiveTest = False
    verbose = False

    def generate_data(self):

        img3d = (np.random.rand(30, 30, 30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation[10:25, 4:24, 2:16] = 1
        img3d = img3d + segmentation*20
        seeds[12:18, 9:16, 3:6] = 1
        seeds[19:22, 21:27, 19:21] = 2

        voxelsize_mm = [5, 5, 5]
        metadata = {'voxelsize_mm': voxelsize_mm}
        return img3d, metadata, seeds, segmentation

    # @unittest.skipIf(not interactiveTest, "interactive test")
    @attr("interactive")
    def test_viewer_seeds(self):

        try:
            from pysegbase.seed_editor_qt import QTSeedEditor
        except:
            print("Deprecated of pyseg_base as submodule")
            from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        import numpy as np
        img3d = (np.random.rand(30, 30, 30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        seeds[3:6, 12:18, 9:16] = 1
        seeds[3:6, 19:22, 21:27] = 2
        # , QMainWindow
        app = QApplication(sys.argv)
        pyed = QTSeedEditor(img3d, seeds=seeds)
        pyed.exec_()

        # deletemask = pyed.getSeeds()
        # import pdb; pdb.set_trace()

        # pyed = QTSeedEditor(deletemask, mode='draw')
        # pyed.exec_()

        app.exit()
    # @unittest.skip("demonstrating skipping")

    @attr("interactive")
    def test_whole_organ_segmentation_interactive(self):
        """
        Interactive test uses dicom data for segmentation
        """
        dcmdir = os.path.join(
            lisa.dataset.sample_data_path(),
            'matlab/examples/sample_data/DICOM/digest_article/'
        )
            # path_to_script,
            # './../sample_data/matlab/examples/sample_data/DICOM/digest_article/') # noqa
        oseg = organ_segmentation.OrganSegmentation(
            dcmdir, working_voxelsize_mm=4, manualroi=False)

# manual seeds setting
        print ("with left mouse button select some pixels of the brain")
        print ("with right mouse button select some pixels of other tissues\
and background")

        oseg.interactivity()

        volume = oseg.get_segmented_volume_size_mm3()
        print volume

        self.assertGreater(volume, 50000)
        self.assertLess(volume, 1200000)


#        roi_mm = [[3, 3, 3], [150, 150, 50]]
#        oseg.ni_set_roi()
#        coordinates_mm = [[110, 50, 30], [10, 10, 10]]
#        label = [1, 2]
#        radius = [5, 5]
#        oseg.ni_set_seeds(coordinates_mm, label, radius)
#
#        oseg.make_segmentation()


# @TODO doladit boundary penalties
    # @unittest.skipIf(not interactiveTest, "interactive test")
    @unittest.skip("interactivity params are obsolete")
    def test_organ_segmentation_with_boundary_penalties(self):
        """
        Interactivity is stored to file
        """
        dcmdir = os.path.join(
            lisa.dataset.sample_data_path(),
            'jatra_5mm')

        print "Interactive test: with left mouse button select liver, \
            with right mouse button select other tissues"
        # gcparams = {'pairwiseAlpha':10, 'use_boundary_penalties':True}
        segparams = {'pairwise_alpha_per': 3,
                     'use_boundary_penalties': True,
                     'boundary_penalties_sigma': 200}
        oseg = organ_segmentation.OrganSegmentation(
            dcmdir, working_voxelsize_mm=4, segparams=segparams)
        oseg.add_seeds_mm([120], [120], [400], label=1, radius=30)
        oseg.add_seeds_mm([170, 220, 250], [250, 280, 200], [400], label=2,
                          radius=30)

        "boundary penalties"
        oseg.interactivity()
        # oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()

        # misc.obj_to_file(oseg.get_iparams(),'iparams.pkl', filetype='pickle')

        self.assertGreater(volume, 1000000)
    # @unittest.skipIf(not interactiveTest, "interactive test")

    @unittest.skip("interactivity params are obsolete")
    def test_create_iparams(self):
        """
        Interactivity is stored to file
        """
        if self.verbose:
            print "test_create_iparams"
        import misc
        dcmdir = os.path.join(lisa.dataset.sample_data_path(), 'jatra_5mm')
            # path_to_script, './../sample_data/jatra_5mm')

        segparams = {'pairwiseAlpha': 20,
                     'use_boundary_penalties': False,
                     'boundary_penalties_sigma': 200}
        # oseg = organ_segmentation.OrganSegmentation(dcmdir, working_voxelsize_mm = 4) # noqa
        oseg = organ_segmentation.OrganSegmentation(
            dcmdir, working_voxelsize_mm=4,
            segparams=segparams, manualroi=False)
        # oseg.add_seeds_mm([120, 160], [150, 120], [70], label=1, radius=20)
        oseg.add_seeds_mm([120, 160], [150, 80], [85], label=1, radius=20)
        oseg.add_seeds_mm([170, 220, 250, 100], [250, 300, 200, 350], [85],
                          label=2, radius=20)
        oseg.add_seeds_mm([170], [240], [70], label=2, radius=20)

        # print "test_ipars"
        # oseg.interactivity()
        oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()
        # print 'vol %.3g ' %(volume)

        misc.obj_to_file(oseg.get_iparams(), 'iparams.pkl', filetype='pickle')

        self.assertGreater(volume, 1000000)

    # @unittest.skipIf(not interactiveTest, "interactive test")
    @attr("interactive")
    def test_stored_interactivity(self):
        pass

# TODO finish this test
    def test_synth_liver(self):
        params = {}
        self.synthetic_liver_template(params)

    def synthetic_liver(self):
        """
        Create synthetic data. There is some liver and porta -like object.
        """
        # data
        slab = {'none': 0, 'liver': 1, 'porta': 2}
        voxelsize_mm = np.array([1.0, 1.0, 1.2])

        segm = np.zeros([80, 256, 250], dtype=np.int16)

        # liver
        segm[30:60, 70:180, 40:190] = slab['liver']
        # porta
        segm[40:45, 120:130, 70:190] = slab['porta']
        segm[40:45, 80:130, 100:110] = slab['porta']
        segm[40:44, 120:170, 130:135] = slab['porta']

        data3d = np.zeros(segm.shape)
        data3d[segm == slab['liver']] = 146
        data3d[segm == slab['porta']] = 206
        noise = (np.random.normal(0, 10, segm.shape))  # .astype(np.int16)
        data3d = (data3d + noise).astype(np.int16)
        return data3d, segm, voxelsize_mm, slab

    def synthetic_liver_template(self, params):
        """
        Function uses organ_segmentation  for synthetic box object
        segmentation.
        """
        # dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/') # noqa
# data

        data3d, segm, voxelsize_mm, slab = self.synthetic_liver()

# seeds
        seeds = np.zeros(data3d.shape, np.int8)
        seeds[40:55, 90:120, 70:110] = 1
        seeds[30:45, 190:200, 40:90] = 2
# [mm]  10 x 10 x 10        # voxelsize_mm = [1, 4, 3]
        metadata = {'voxelsize_mm': voxelsize_mm}

        oseg = organ_segmentation.OrganSegmentation(
            None,
            data3d=data3d,
            metadata=metadata,
            seeds=seeds,
            working_voxelsize_mm=5,
            manualroi=False,
            autocrop=False,

            **params
        )

        oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()
        oseg.portalVeinSegmentation(interactivity=False, threshold=180)
        oseg.saveVesselTree('porta')

        # print '> 0 '
        # print np.sum(oseg.segmentation > 0)
        # print np.sum(segm > 0)
        # print np.sum(oseg.segmentation > 0) * np.prod(voxelsize_mm)
        # print np.sum(segm > 0) * np.prod(voxelsize_mm)
        # print 'computed ', volume
        # print voxelsize_mm
        # print oseg.voxelsize_mm

        # import pdb; pdb.set_trace()
        # import sed3
        # ed = sed3.sed3(data3d, seeds=seeds,
        #                contour=(oseg.segmentation))
        # ed.show()
        # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        # mel by to být litr. tedy milion mm3
        # je to zvlastni. pro nekter verze knihoven je to 630, pro jine 580
        self.assertGreater(volume, 570000)
        self.assertLess(volume, 640000)

    def test_roi(self):
        """
        Test setting of ROI. It is in pixels, not in mm
        """

        img3d = (np.random.rand(30, 30, 30)*10).astype(np.int16)
        seeds = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation = (np.zeros(img3d.shape)).astype(np.int8)
        segmentation[10:25, 4:24, 2:16] = 1
        img3d = img3d + segmentation*20
        seeds[12:18, 9:16, 3:6] = 1
        seeds[19:22, 21:27, 19:21] = 2

        roi = [[7, 27], [2, 29], [0, 26]]
        # seeds = seeds[7:27, 2:29, 0:26]
        voxelsize_mm = [5, 5, 5]
        metadata = {'voxelsize_mm': voxelsize_mm}

        oseg = organ_segmentation.OrganSegmentation(
            None,
            data3d=img3d,
            metadata=metadata,
            seeds=seeds,
            roi=roi,
            working_voxelsize_mm=5,
            manualroi=False)

        # from PyQt4.QtGui import QApplication
        # app = QApplication(sys.argv)
        # oseg.interactivity(min_val=0, max_val=30)
        oseg.ninteractivity()
        datap = oseg.export()

        volume = oseg.get_segmented_volume_size_mm3()
        self.assertGreater(volume, 500000)
        self.assertIn('data3d', datap.keys())
        self.assertIn('voxelsize_mm', datap.keys())

    def test_box_segmentation(self):
        params = {'segmentation_smoothing': False}
        self.box_segmentation_template(params)

    def test_box_segmentation_with_smoothing(self):
        """
        Function uses organ_segmentation  for synthetic box object
        segmentation.
        """
        params = {'segmentation_smoothing': True}
        self.box_segmentation_template(params)
        # dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/') # noqa

    def box_segmentation_template(self, params):
        """
        Function uses organ_segmentation  for synthetic box object
        segmentation.
        """
        # dcmdir = os.path.join(path_to_script,'./../sample_data/matlab/examples/sample_data/DICOM/digest_article/') # noqa
# data
        img3d = np.random.rand(32, 64, 64) * 3
        img3d[4:24, 12:32, 5:25] = img3d[4:24, 12:32, 5:25] + 25

# seeds
        seeds = np.zeros([32, 64, 64], np.int8)
        seeds[9:12, 13:29, 18:25] = 1
        seeds[9:12, 4:9, 3:32] = 2
# [mm]  10 x 10 x 10        # voxelsize_mm = [1, 4, 3]
        voxelsize_mm = [5, 5, 5]
        metadata = {'voxelsize_mm': voxelsize_mm}

        oseg = organ_segmentation.OrganSegmentation(
            None,
            data3d=img3d,
            metadata=metadata,
            seeds=seeds,
            working_voxelsize_mm=10,
            manualroi=False,
            **params
        )

        # oseg.seeds = seeds
        # oseg.make_gc()
# manual seeds setting

        # from PyQt4.QtGui import QApplication
        # app = QApplication(sys.argv)
        # oseg.interactivity()
        oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()

        # import pdb; pdb.set_trace()

        # mel by to být litr. tedy milion mm3
        self.assertGreater(volume, 900000)
        self.assertLess(volume, 1100000)

    def test_volume_resize(self):
        # from scipy.sparse.import lil_matrix

        pass

    # @unittest.skipIf(True, "interactive test")
    # @unittest.skipIf(not interactiveTest, "interactive test")
    @attr("interactive")
    def test_vincentka_06_slice_thickness_interactive(self):
        """
        Interactive test. SliceThickness is not voxel depth. If it is, this
        test will fail.
        """
        # dcmdir = os.path.join(path_to_script, './../sample_data/matlab/examples/sample_data/DICOM/digest_article/') #noqa
        dcmdir = os.path.expanduser(
            '~/data/medical/data_orig/vincentka/13021610/10200000/')
        dcmdir = os.path.expanduser(
            '~/data/medical/data_orig/vincentka/13021610/12460000/')
        oseg = organ_segmentation.OrganSegmentation(dcmdir,
                                                    working_voxelsize_mm=4,
                                                    manualroi=False)

# manual seeds setting
        print(
            "with left mouse button select some pixels of the bottle content")
        print("with right mouse button select some pixels of background")

        oseg.interactivity()

        volume = oseg.get_segmented_volume_size_mm3()
        # print volume

        self.assertGreater(volume, 550000)
        self.assertLess(volume, 850000)

    def setUp(self):
        """ Nastavení společných proměnných pro testy  """
        self.assertTrue(True)

    # @TODO dodělat přidávání uzlů pomocí mm
    # @unittest.skipIf(not interactiveTest, "interactive test")
    def test_add_seeds_mm(self):
        """
        Function uses organ_segmentation object for segmentation
        """
        dcmdir = os.path.join(
            lisa.dataset.sample_data_path(),
            'matlab/examples/sample_data/DICOM/digest_article/'
            # path_to_script,
            # './../sample_data/matlab/examples/sample_data/DICOM/digest_article/'
        )
        oseg = organ_segmentation.OrganSegmentation(dcmdir,
                                                    working_voxelsize_mm=4,
                                                    manualroi=False)

        oseg.add_seeds_mm([120], [120], [80], 1, 25)
        oseg.add_seeds_mm([25], [100], [80], 2, 25)

        # pro kontrolu lze odkomentovat
        # oseg.interactivity()

        oseg.ninteractivity()

        volume = oseg.get_segmented_volume_size_mm3()

        # import pdb; pdb.set_trace()

        # mel by to být litr. tedy milion mm3
        self.assertGreater(volume, 800000)
        self.assertLess(volume, 1200000)

        # roi_mm = [[3, 3, 3], [150, 150, 50]]
        # oseg.ni_set_roi()
        # coordinates_mm = [[110, 50, 30], [10, 10, 10]]
        # label = [1, 2]
        # radius = [5, 5]
        # oseg.ni_set_seeds(coordinates_mm, label, radius)

        # oseg.make_segmentation()

        # oseg.noninteractivity()
        pass

    @unittest.skip("demonstrating skipping")
    def test_dicomread_and_graphcut(self):
        """
        Test dicomread module and graphcut module
        """
        try:
            from pysegbase import pycut
        except:
            print("Deprecated of pyseg_base as submodule")
            import pycut

        dcmdir = os.path.join(path_to_script, './../sample_data/matlab/examples/sample_data/DICOM/digest_article/') #noqa
        data3d, metadata = dcmr.dcm_read_from_dir(dcmdir)

        # print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) ) #noqa

        igc = pycut.ImageGraphCut(data3d, zoom=0.5)
        seeds = igc.seeds
        seeds[0, :, 0] = 1
        seeds[60:66, 60:66, 5:6] = 2
        igc.noninteractivity(seeds)

        igc.make_gc()
        segmentation = igc.segmentation
        self.assertTrue(segmentation[14, 4, 1] == 0)
        self.assertTrue(segmentation[127, 120, 10] == 1)
        self.assertTrue(np.sum(segmentation == 1) > 100)
        self.assertTrue(np.sum(segmentation == 0) > 100)
        # igc.show_segmentation()

if __name__ == "__main__":
    unittest.main()
