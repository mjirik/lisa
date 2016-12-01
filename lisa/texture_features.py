# ! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Texture features
"""

import numpy as np
import skimage
import skimage.filters
import skimage.feature
from scipy import ndimage as nd


def feat_hist2(data3d_orig):
    bins = range(-512, 512, 100)
    hist1, bin_edges1 = np.histogram(data3d_orig, bins=bins)
    return hist1


class FeaturesCombinedFeatures():
    """
    This object alows combine two or three features into one.
    """
    def __init__(
        self,
        feature_function1,
        feature_function2,
        feature_function3=None,
    ):
        self.ff1 = feature_function1
        self.ff2 = feature_function2
        self.ff3 = feature_function3
        self.description = self.ff1.__name__ + "+" + self.ff2.__name__
        if feature_function3 is not None:
            self.description +=  "+" + self.ff3.__name__

    def features(self, data3d):
        fv1 = self.ff1(data3d)
        fv2 = self.ff2(data3d)
        feats = np.concatenate((fv1, fv2))
        if self.ff3 is not None:
            fv3 = self.ff3(data3d)
            feats = np.concatenate((feats, fv3))

        return np.array(feats).reshape(-1)


# Gabor filters --------
class GaborFeatures():
    def __init__(self):
        # prepare filter bank kernels
        self.kernels = []
        for theta in range(4):
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for frequency in (0.05, 0.25):
                    kernel = np.real(
                        skimage.filters.gabor_kernel(
                            frequency, theta=theta,
                            sigma_x=sigma, sigma_y=sigma))
                    self.kernels.append(kernel)

    def feats_gabor(self, data3d):
        """
        Compute features based on Gabor filters.
        """
        fv = self.__compute_gabor_feats(
            data3d[:, :, 0], self.kernels).reshape(-1)
        return fv

    def __compute_gabor_feats(self, image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = nd.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats


class GlcmFeatures():
    def feats_glcm(self, data3d):
        # feats = np.zeros((len(kernels), 2), dtype=np.double)
        # @TODO data are cast to uint8
        feats = []
        im = data3d[0, :, :]
        w_center = 100
        w_width = 250
        im_uint8 = (1.0 /
                    (1 + np.exp((w_center - im) / w_center))
                    * w_width).astype(np.uint8)
        glcm = skimage.feature.greycomatrix(
            im_uint8,
            distances=[5],
            angles=[0, np.pi / 2],
            levels=256,
            symmetric=True,
            normed=True)
        feats.append(skimage.feature.greycoprops(glcm, 'dissimilarity'))
        feats.append(skimage.feature.greycoprops(glcm, 'correlation'))
        feats.append(skimage.feature.greycoprops(glcm, 'contrast'))
        feats.append(skimage.feature.greycoprops(glcm, 'ASM'))
        feats.append(skimage.feature.greycoprops(glcm, 'energy'))
        return np.array(feats).reshape(-1)


class HaralickFeatures():
    def feats_haralick(self, data3d, direction_independent=True):
        import mahotas
        import mahotas.features

        mhf = mahotas.features.haralick(data3d+1024)

        if direction_independent:
            mhf = np.average(mhf, 0)

        return mhf.reshape(-1)
