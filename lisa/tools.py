__author__ = 'Ryba'
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as skexp
from skimage.segmentation import mark_boundaries
import os
import glob
import dicom
# import cv2
# from skimage import measure
import skimage.measure as skimea
import skimage.morphology as skimor
import skimage.filter as skifil
import scipy.stats as scista

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def get_seeds(im, minT=0.95, maxT=1.05, minInt=0, maxInt=255, debug=False):
    vals = im[np.where(np.logical_and(im>=minInt, im<=maxInt))]
    hist, bins = skexp.histogram(vals)
    max_peakIdx = hist.argmax()

    minT *= bins[max_peakIdx]
    maxT *= bins[max_peakIdx]
    histTIdxs = (bins >= minT) * (bins <= maxT)
    histTIdxs = np.nonzero(histTIdxs)[0]
    class1TMin = minT
    class1TMax = maxT

    seed_mask = np.where( (im >= class1TMin) * (im <= class1TMax), 1, 0)

    if debug:
        plt.figure()
        plt.plot(bins, hist)
        plt.hold(True)

        plt.plot(bins[max_peakIdx], hist[max_peakIdx], 'ro')
        plt.plot(bins[histTIdxs], hist[histTIdxs], 'r')
        plt.plot(bins[histTIdxs[0]], hist[histTIdxs[0]], 'rx')
        plt.plot(bins[histTIdxs[-1]], hist[histTIdxs[-1]], 'rx')
        plt.title('Image histogram and its class1 = maximal peak (red dot) +/- minT/maxT % of its density (red lines).')
        plt.show()

    #minT *= hist[max_peakIdx]
    #maxT *= hist[max_peakIdx]
    #histTIdxs = (hist >= minT) * (hist <= maxT)
    #histTIdxs = np.nonzero(histTIdxs)[0]
    #histTIdxs = histTIdxs.astype(np.int)minT *= hist[max_peakIdx]
    #class1TMin = bins[histTIdxs[0]]
    #class1TMax = bins[histTIdxs[-1]

    #if debug:
    #    plt.figure()
    #    plt.plot(bins, hist)
    #    plt.hold(True)
    #
    #    plt.plot(bins[max_peakIdx], hist[max_peakIdx], 'ro')
    #    plt.plot(bins[histTIdxs], hist[histTIdxs], 'r')
    #    plt.plot(bins[histTIdxs[0]], hist[histTIdxs[0]], 'rx')
    #    plt.plot(bins[histTIdxs[-1]], hist[histTIdxs[-1]], 'rx')
    #    plt.title('Image histogram and its class1 = maximal peak (red dot) +/- minT/maxT % of its density (red lines).')
    #    plt.show()

    return seed_mask, class1TMin, class1TMax


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def seeds2superpixels(seed_mask, superpixels, debug=False, im=None):
    seeds = np.argwhere(seed_mask)
    superseeds = np.zeros_like(seed_mask)

    for s in seeds:
        label = superpixels[s[0], s[1]]
        superseeds = np.where(superpixels==label, 1, superseeds)

    if debug:
        plt.figure(), plt.gray()
        plt.subplot(121), plt.imshow(im), plt.hold(True), plt.plot(seeds[:,1], seeds[:,0], 'ro'), plt.axis('image')
        plt.subplot(122), plt.imshow(im), plt.hold(True), plt.plot(seeds[:,1], seeds[:,0], 'ro'),
        plt.imshow(mark_boundaries(im, superseeds, color=(1,0,0))), plt.axis('image')
        plt.show()

    return superseeds


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def intensity_range2superpixels(im, superpixels, intMinT=0.95, intMaxT=1.05, debug=False, intMin=0, intMax=255):#, fromInt=0, toInt=255):

    superseeds = np.zeros_like(superpixels)

    #if not intMin and not intMax:
    #    hist, bins = skexp.histogram(im)
    #
    #    #zeroing values that are lower/higher than fromInt/toInt
    #    toLow = np.where(bins < fromInt)
    #    hist[toLow] = 0
    #    toHigh = np.where(bins > toInt)
    #    hist[toHigh] = 0
    #
    #    max_peakIdx = hist.argmax()
    #    intMin = intMinT * bins[max_peakIdx]
    #    intMax = intMaxT * bins[max_peakIdx]

    sp_means = np.zeros(superpixels.max()+1)
    for sp in range(superpixels.max()+1):
        values = im[np.where(superpixels==sp)]
        mean = np.mean(values)
        sp_means[sp] = mean

    idxs = np.argwhere(np.logical_and(sp_means>=intMin, sp_means<=intMax))
    for i in idxs:
        superseeds = np.where(superpixels==i[0], 1, superseeds)

    if debug:
        plt.figure(), plt.gray()
        plt.imshow(im), plt.hold(True), plt.imshow(mark_boundaries(im, superseeds, color=(1,0,0)))
        plt.axis('image')
        plt.show()

    return superseeds


def show_slice(data, segmentation=None, lesions=None, show='True'):
    plt.figure()
    plt.gray()
    plt.imshow(data)

    if segmentation is not None:
        plt.hold(True)
        contours = skimea.find_contours(segmentation, 1)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'b', linewidth=2)

    if lesions is not None:
        plt.hold(True)
        contours = skimea.find_contours(lesions, 1)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)

    plt.axis('image')

    if show:
        plt.show()


def change_slice_index(data):
    nSlices = data.shape[2]
    data_reshaped = np.zeros(np.hstack((data.shape[2],data.shape[0],data.shape[1])))
    for i in range(nSlices):
        data_reshaped[i,:,:] = data[:,:,i]
    return data_reshaped


def read_data(dcmdir, indices=None, wildcard='*.dcm', type=np.int16):

    dcmlist = []
    for infile in glob.glob(os.path.join(dcmdir, wildcard)):
        dcmlist.append(infile)

    if indices == None:
        indices = range(len(dcmlist))

    data3d = []
    for i in range(len(indices)):
        ind = indices[i]
        onefile = dcmlist[ind]
        if wildcard == '*.dcm':
            data = dicom.read_file(onefile)
            data2d = data.pixel_array
            try:
                data2d = (np.float(data.RescaleSlope) * data2d) + np.float(data.RescaleIntercept)
            except:
                print('problem with RescaleSlope and RescaleIntercept')
        else:
            data2d = cv2.imread(onefile, 0)

        if len(data3d) == 0:
            shp2 = data2d.shape
            data3d = np.zeros([shp2[0], shp2[1], len(indices)], dtype=type)

        data3d[:,:,i] = data2d

    #need to reshape data to have slice index (ndim==3)
    if data3d.ndim == 2:
        data3d.resize(np.hstack((data3d.shape,1)))

    return data3d


def windowing(data, level=50, width=300, sub1024=False, sliceId=2):
    #srovnani na standardni skalu = odecteni 1024HU
    if sub1024:
        data -= 1024

    #zjisteni minimalni a maximalni density
    minHU = level - width
    maxHU = level + width

    if data.ndim == 3:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                #rescalovani intenzity tak, aby skala <minHU, maxHU> odpovidala intervalu <0,255>
                data[:,:,idx] = skexp.rescale_intensity(data[:,:,idx], in_range=(minHU, maxHU), out_range=(0, 255))
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                #rescalovani intenzity tak, aby skala <minHU, maxHU> odpovidala intervalu <0,255>
                data[idx,:,:] = skexp.rescale_intensity(data[idx,:,:], in_range=(minHU, maxHU), out_range=(0, 255))
    else:
        data = skexp.rescale_intensity(data, in_range=(minHU, maxHU), out_range=(0, 255))

    return data.astype(np.uint8)


def smoothing(data, d=10, sigmaColor=10, sigmaSpace=10, sliceId=2):
    if data.ndim == 3:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                data[:,:,idx] = cv2.bilateralFilter( data[:,:,idx], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace )
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                data[idx,:,:] = cv2.bilateralFilter( data[idx,:,:], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace )
    else:
        data = cv2.bilateralFilter( data, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace )
    return data

def smoothing_bilateral(data, sigma_space=15, sigma_color=0.05, pseudo_3D='True', sliceId=2):
    if data.ndim == 3 and pseudo_3D:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                temp = skifil.denoise_bilateral(data[:, :, idx], sigma_range=sigma_color, sigma_spatial=sigma_space)
                data[idx, :, :] = (255 * temp).astype(np.uint8)
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                temp = skifil.denoise_bilateral(data[idx, :, :], sigma_range=sigma_color, sigma_spatial=sigma_space)
                data[idx, :, :] = (255 * temp).astype(np.uint8)
    else:
        data = skifil.denoise_bilateral(data, sigma_range=sigma_color, sigma_spatial=sigma_space)
        data = (255 * data).astype(np.uint8)
    return data

def smoothing_tv(data, weight, pseudo_3D='True', multichannel=False, sliceId=2):
    if data.ndim == 3 and pseudo_3D:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                temp = skifil.denoise_tv_chambolle(data[:, :, idx], weight=weight, multichannel=multichannel)
                data[:, :, idx] = (255 * temp).astype(np.uint8)
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                temp = skifil.denoise_tv_chambolle(data[idx, :, :], weight=weight, multichannel=multichannel)
                data[idx, :, :] = (255 * temp).astype(np.uint8)
    else:
        data = skifil.denoise_tv_chambolle(data, weight=weight, multichannel=False)
        data = (255 * data).astype(np.uint8)
    return data

def canny(data, sigma=1, sliceId=2):
    edges = np.zeros(data.shape, dtype=np.bool)
    if sliceId == 2:
        for idx in range(data.shape[2]):
            edges[:, :, idx] = skifil.canny(data[:, :, idx], sigma=sigma)
    elif sliceId == 0:
        for idx in range(data.shape[0]):
            edges[idx, :, :] = skifil.canny(data[idx, :, :], sigma=sigma)
    return edges

def scharr(data, sliceId=2):
    edges = np.zeros(data.shape)
    if sliceId == 2:
        for idx in range(data.shape[2]):
            edges[:, :, idx] = skifil.scharr(data[:, :, idx])
    elif sliceId == 0:
        for idx in range(data.shape[0]):
            edges[idx, :, :] = skifil.scharr(data[idx, :, :])
    return edges

def sobel(data, sliceId=2):
    edges = np.zeros(data.shape)
    if sliceId == 2:
        for idx in range(data.shape[2]):
            edges[:, :, idx] = skifil.sobel(data[:, :, idx])
    elif sliceId == 0:
        for idx in range(data.shape[0]):
            edges[idx, :, :] = skifil.sobel(data[idx, :, :])
    return edges

def roberts(data, sliceId=2):
    edges = np.zeros(data.shape)
    if sliceId == 2:
        for idx in range(data.shape[2]):
            edges[:, :, idx] = skifil.roberts(data[:, :, idx])
    elif sliceId == 0:
        for idx in range(data.shape[0]):
            edges[idx, :, :] = skifil.roberts(data[idx, :, :])
    return edges

def analyse_histogram(data, roi=None, debug=False, dens_min=20, dens_max=255, minT=0.95, maxT=1.05):
    if roi == None:
        #roi = np.ones(data.shape, dtype=np.bool)
        roi = np.logical_and(data >= dens_min, data <= dens_max)

    smooth = smoothing_tv(data, weight=0.1, sliceId=0)
    voxels = data[np.nonzero(roi)]
    hist, bins = skexp.histogram(voxels)
    max_peakIdx = hist.argmax()

    minT = minT * hist[max_peakIdx]
    maxT = maxT * hist[max_peakIdx]
    histTIdxs = (hist >= minT) * (hist <= maxT)
    histTIdxs = np.nonzero(histTIdxs)[0]
    histTIdxs = histTIdxs.astype(np.int)

    class1TMin = bins[histTIdxs[0]]
    class1TMax = bins[histTIdxs[-1]]

    # liver = data * (roi > 0)
    liver = smooth * (roi > 0)
    class1 = np.where( (liver >= class1TMin) * (liver <= class1TMax), 1, 0)

    if debug:
        plt.figure()
        plt.plot(bins, hist)
        plt.hold(True)

        plt.plot(bins[max_peakIdx], hist[max_peakIdx], 'ro')
        plt.plot(bins[histTIdxs], hist[histTIdxs], 'r')
        plt.plot(bins[histTIdxs[0]], hist[histTIdxs[0]], 'rx')
        plt.plot(bins[histTIdxs[-1]], hist[histTIdxs[-1]], 'rx')
        plt.title('Histogram of liver density and its class1 = maximal peak (red dot) +-5% of its density (red line).')
        plt.show()

    return class1


def intensity_probability(data, std=20, roi=None, dens_min=5, dens_max=255):
    if roi == None:
        # roi = np.logical_and(data >= dens_min, data <= dens_max)
        roi = np.ones(data.shape, dtype=np.bool)
    voxels = data[np.nonzero(roi)]
    hist, bins = skexp.histogram(voxels)

    # zeroing histogram outside interval <dens_min, dens_max>
    # update: it's not necessary if there's a roi provided
    hist[:dens_min] = 0
    hist[dens_max:] = 0

    max_id = hist.argmax()
    mu = round(bins[max_id])

    prb = scista.norm(loc=mu, scale=std)
    probs_L = prb.pdf(voxels)

    print('liver pdf: mu = %i, std = %i'%(mu, std))

    # plt.figure()
    # plt.plot(bins, hist)
    # plt.hold(True)
    # plt.plot(mu, hist[max_id], 'ro')
    # plt.show()

    probs = np.zeros(data.shape)

    coords = np.argwhere(roi)
    n_elems = coords.shape[0]
    for i in range(n_elems):
        if data.ndim == 3:
            probs[coords[i,0], coords[i,1], coords[i,2]] = probs_L[i]
        else:
            probs[coords[i,0], coords[i,1]] = probs_L[i]

    return probs, mu


def get_zunics_compatness(obj):
    m000 = obj.sum()
    m200 = get_central_moment(obj, 2, 0, 0)
    m020 = get_central_moment(obj, 0, 2, 0)
    m002 = get_central_moment(obj, 0, 0, 2)
    term1 = (3**(5./3)) / (5 * (4*np.pi)**(2./3))
    term2 = m000**(5./3) / (m200 + m020 + m002)
    K = term1 * term2
    return K


def get_central_moment(obj, p, q, r):
    elems = np.argwhere(obj)
    m000 = obj.sum()
    m100 = (elems[:,0]).sum()
    m010 = (elems[:,1]).sum()
    m001 = (elems[:,2]).sum()
    xc = m100 / m000
    yc = m010 / m000
    zc = m001 / m000

    mom = 0
    for el in elems:
        mom += (el[0] - xc)**p + (el[1] - yc)**q + (el[2] - zc)**r

    return mom


def opening3D(data, selem=skimor.disk(3)):
    for i in range(data.shape[0]):
        data[i,:,:] = skimor.binary_opening(data[i,:,:], selem)
    return data


def closing3D(data, selem=skimor.disk(3)):
    for i in range(data.shape[0]):
        data[i,:,:] = skimor.binary_closing(data[i,:,:], selem)
    return data


def resize3D(data, scale, sliceId=2):
    if sliceId == 2:
        n_slices = data.shape[2]
        new_shape = cv2.resize(data[:,:,0], None, fx=scale, fy=scale).shape
        new_data = np.zeros(np.hstack((new_shape,n_slices)))
        for i in range(n_slices):
            new_data[:,:,i] = cv2.resize(data[:,:,i], None, fx=scale, fy=scale)
    elif sliceId == 0:
        n_slices = data.shape[0]
        new_shape = cv2.resize(data[0,:,:], None, fx=scale, fy=scale).shape
        new_data = np.zeros(np.hstack((n_slices, np.array(new_shape))))
        for i in range(n_slices):
            new_data[i,:,:] = cv2.resize(data[i,:,:], None, fx=scale, fy=scale)
    return new_data


def get_overlay(mask, alpha=0.3, color='r'):
    layer = None
    if color == 'r':
        layer = np.dstack((255*mask, np.zeros_like(mask), np.zeros_like(mask), alpha * mask))
    elif color == 'g':
        layer = alpha * np.dstack((np.zeros_like(mask), mask, np.zeros_like(mask)))
    elif color == 'b':
        layer = alpha * np.dstack((np.zeros_like(mask), np.zeros_like(mask), mask))
    elif color == 'c':
        layer = alpha * np.dstack((np.zeros_like(mask), mask, mask))
    elif color == 'm':
        layer = alpha * np.dstack((mask, np.zeros_like(mask), mask))
    elif color == 'y':
        layer = alpha * np.dstack((mask, mask, np.zeros_like(mask)))
    else:
        print 'Unknown color, using red as default.'
        layer = alpha * np.dstack((mask, np.zeros_like(mask), np.zeros_like(mask)))
    return layer


def slim_seeds(seeds, sliceId=2):
    slims = np.zeros_like(seeds)
    if sliceId == 0:
        for i in range(seeds.shape[0]):
            layer = seeds[i,:,:]
            labels = skimor.label(layer, neighbors=4, background=0) + 1
            n_labels = labels.max()
            for o in range(1,n_labels+1):
                centroid = np.round(skimea.regionprops(labels == o)[0].centroid)
                slims[i, centroid[0], centroid[1]] = 1
    return slims
