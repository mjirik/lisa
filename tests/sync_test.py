# ! /usr/bin/python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

# import funkcí z jiného adresáře
import sys
import os.path
import shutil

# path_to_script = os.path.dirname(os.path.abspath(__file__))
import unittest

import numpy as np
from nose.plugins.attrib import attr


from lisa import organ_segmentation
import lisa.dataset
import lisa.lisa_data


# nosetests tests/organ_segmentation_test.py:OrganSegmentationTest.test_create_iparams # noqa


class OrganSegmentationTest(unittest.TestCase):

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

    # @unittest.skip("in progress")
    def test_sync_paul(self):
        """
        sync with paul account
        """

        path_to_paul = lisa.lisa_data.path('sync','paul')

        if os.path.exists(path_to_paul):
            remove_read_only_flag_on_win32(path_to_paul)
            # handleRemoveReadonly je asi zbytečné
            shutil.rmtree(path_to_paul, onerror=handleRemoveReadonly, ignore_errors=False)
        oseg = organ_segmentation.OrganSegmentation(None)
        oseg.sync_lisa_data('paul','P4ul')


        file_path = lisa.lisa_data.path('sync','paul', 'from_server', 'test.txt')
        logger.debug('file_path %s', file_path)
        self.assertTrue(os.path.exists(file_path))

import errno, os, stat, shutil

def remove_read_only_flag_on_win32(path):
    """
    remove read only flag on windows files recusivelly
    :param path:
    :return:
    """

    platform = sys.platform
    if platform == "win32":
        import win32api, win32con
        for root, dirs, files in os.walk(path):
            for momo in dirs:
                pth = os.path.join(root, momo)
                win32api.SetFileAttributes(pth, win32con.FILE_ATTRIBUTE_NORMAL)
                pass
                # win32api.SetFileAttributes(pth, win32con.FILE_ATTRIBUTE_NORMAL)
            for momo in files:
                pth = os.path.join(root, momo)
                win32api.SetFileAttributes(pth, win32con.FILE_ATTRIBUTE_NORMAL)

def handleRemoveReadonly(func, path, exc):
  excvalue = exc[1]
  if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
      os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO| stat.S_IWRITE) # 0777
      os.unlink(path)

      func(path)
  else:
      import traceback
      traceback.print_exc()
      raise Exception("handleRemoveReadonly exception" + str(func))

if __name__ == "__main__":
    unittest.main()
