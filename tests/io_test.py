# ! /usr/bin/python
# -*- coding: utf-8 -*-


# import funkcí z jiného adresáře
import sys
import os.path
import os.path as op


# path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest

import numpy as np
# from nose.plugins.attrib import attr


from lisa import organ_segmentation
import imcut.dcmreaddata as dcmr
import lisa.dataset
import io3d


# nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams # noqa


class InOutTest(unittest.TestCase):
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

    def test_export_dicom(self):
        """
        Interactivity is stored to file
        """
        import io3d
        import io3d.datareader

        if self.verbose:
            print("test_create_iparams")
        data_path = os.path.join(lisa.dataset.sample_data_path(), 'liver-orig001.mhd')
        seg_path = os.path.join(lisa.dataset.sample_data_path(), 'liver-seg001.mhd')

        # load data
        oseg = organ_segmentation.OrganSegmentation()
        oseg.load_data(datapath=data_path)
        oseg.import_segmentation_from_file(seg_path)

        # export to dicom
        output_datapath = "data3d.dcm"
        output_segpath = "segmentation.dcm"
        oseg.save_input_dcm(output_datapath)
        oseg.save_outputs_dcm(output_segpath)

        # check data
        data3d_orig, metadata = io3d.datareader.read(data_path, dataplus_format=False)
        data3d_stored, metadata = io3d.datareader.read(output_datapath, dataplus_format=False)

        err_data = np.sum(np.abs(data3d_orig - data3d_stored))
        self.assertLess(err_data, 500, "data error")

        # check segmentation
        seg_orig, metadata = io3d.datareader.read(data_path, dataplus_format=False)
        seg_stored, metadata = io3d.datareader.read(output_datapath, dataplus_format=False)

        err_data = np.sum(np.abs(seg_orig - seg_stored))
        self.assertLess(err_data, 500, "segmentation error")

        os.remove(output_datapath)
        os.remove(output_segpath)

    def test_lisa_read_mhd_save_pklz(self):

        infn = io3d.datasets.join_path("liver-orig001.mhd")

        oseg = lisa.organ_segmentation.OrganSegmentation(infn)
        oseg.save_outputs("test_mhd.pklz")
        # self.oseg_w = OrganSegmentationWindow(oseg) # noqa


if __name__ == "__main__":
    unittest.main()
