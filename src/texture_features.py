# ! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Texture features
"""

import numpy as np
import skimage
import skimage.filter
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
