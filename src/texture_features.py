# ! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Texture features
"""

import numpy as np
import skimage
import skimage.filter
import skimage.feature
from scipy import ndimage as nd


def feat_hist2(data3d_orig):
    bins = range(-512, 512, 100)
    hist1, bin_edges1 = np.histogram(data3d_orig, bins=bins)
    return hist1


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
                        skimage.filter.gabor_kernel(
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
        im_uint8 = (1.0 / (1 + np.exp((100 - im)/100)) * 250).astype(np.uint8)
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
