#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path
import logging

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../experiments/"))
# sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
# sys.path.append(os.path.join(path_to_script, "../src/"))
import unittest
from nose.plugins.attrib import attr

import experiments.tiled_liver_statistics as tls


class TextureFeaturesExperimentTest(unittest.TestCase):

    # @unittest.skip("comment after implementation")

    @attr('slow')
    def test_run_experiments(self):
        """
        """
        import lisa.texture_features as tfeat
        from sklearn import svm
        from sklearn.naive_bayes import GaussianNB
        import classification
        self.dcmdir = os.path.join(
            path_to_script, '../sample_data/jatra_06mm_jenjatraplus/')
        yaml_file = os.path.join(
            path_to_script, '../experiments/20130919_liver_statistics.yaml')

        # write_csv(fvall)
        gf = tfeat.GaborFeatures()
        glcmf = tfeat.GlcmFeatures()
        haralick = tfeat.HaralickFeatures()

        list_of_feature_fcn = [
            [tls.feat_hist, []],
            # [gf.feats_gabor, []],
            # [glcmf.feats_glcm, []],
            # [haralick.feats_haralick, [True]]
        ]
        list_of_classifiers = [
            # [GaussianNB, []],
            # [svm.SVC, []],
            [classification.GMMClassifier,
                {'n_components': 2, 'covariance_type': 'full'}],
        ]
        featrs_plus_classifs = tls.make_product_list(list_of_feature_fcn,
                                                     list_of_classifiers)

        tile_shape = [50, 50, 50]

        tls.experiment(yaml_file, yaml_file,
                       featrs_plus_classifs, tile_shape=tile_shape,
                       use_voxelsize_norm=False,
                       working_voxelsize_mm=[1, 1, 1],
                       visualization=False)


        # slab = {'none':0, 'bone':8,'lungs':9,'heart':10}
        # import pdb; pdb.set_trace()
# SupportStructureSegmentation
        # sss = support_structure_segmentation.SupportStructureSegmentation(
                # data3d = self.data3d,
                # voxelsize_mm = self.metadata['voxelsize_mm'],
                # modality = 'CT',
                # slab = slab
                #)
        # sss.lungs_segmentation()
        # sss.segmentation[260:270,160:170,1:10] = 2
        # sss.visualization()
        # total number of voxels segmented as bones in spine
        # probebox1 = sss.segmentation [260:270,160:170,1:10]== slab['lungs']
        # self.assertGreater(np.sum(probebox1),20)
        # total number of voexel segmented as none in upper left corner
        # probebox1 = sss.segmentation[10:20,10:20,5:15] == slab['none']
        # self.assertGreater(np.sum(probebox1),900)
        # import pdb; pdb.set_trace()
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger().setLevel(logging.DEBUG)
    unittest.main()
