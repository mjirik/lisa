#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import argparse
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
# sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector

import logging
logger = logging.getLogger(__name__)


#import apdb
#  apdb.set_trace();\
#import scipy.io
import numpy as np
import scipy
import scipy.ndimage
from imtools import misc, qmisc


# ----------------- my scripts --------
class BodyNavigation:

    def __init__(self, data3d, voxelsize_mm):
        self.voxelsize_mm = np.asarray(voxelsize_mm)
        self.working_vs = np.asarray([1.5, 1.5, 1.5])
        if voxelsize_mm is None:
            self.data3dr = data3d
        else:
            self.data3dr = io3d.misc.resize_to_mm(data3d, voxelsize_mm, self.working_vs)
        self.lungs = None
        self.spine = None
        self.body = None
        self.orig_shape = data3d.shape
        self.diaphragm_mask = None
        self.angle = None


    def get_spine(self):

        spine = scipy.ndimage.filters.gaussian_filter(self.data3dr, sigma=[20, 5, 5]) > 200
        self.spine = spine

        self.spine_center = np.mean(np.nonzero(self.spine), 1)
        # self.center2 = np.mean(np.nonzero(self.spine), 2)
        return io3d.misc.resize_to_shape(spine, self.orig_shape)

    def get_body(self):
        body = scipy.ndimage.filters.gaussian_filter(self.data3dr, sigma=2) > -150
        body[0, :, :] = 1
        body[-1, :, :] = 1

        self.body = scipy.ndimage.morphology.binary_fill_holes(body)
        return io3d.misc.resize_to_shape(self.body, self.orig_shape)

    def get_lungs(self):
        lungs = scipy.ndimage.filters.gaussian_filter(self.data3dr, sigma=[4, 2, 2]) > -150
        lungs[0, :, :] = 1

        lungs = scipy.ndimage.morphology.binary_fill_holes(lungs)
        labs, n = scipy.ndimage.measurements.label(lungs==0)
        cornerlab = [
            labs[0,0,0],
            labs[0,0,-1],
            labs[0,-1,0],
            labs[0,-1,-1],
            labs[-1,0,0],
            labs[-1,0,-1],
            labs[-1,-1,0],
            labs[-1,-1,-1]
            ]

        lb = np.median(cornerlab)
        labs[labs==lb] = 0

        labs[labs==labs[0,0,0]] = 0
        labs[labs==labs[0,0,-1]] = 0
        labs[labs==labs[0,-1,0]] = 0
        labs[labs==labs[0,-1,-1]] = 0
        labs[labs==labs[-1,0,0]] = 0
        labs[labs==labs[-1,0,-1]] = 0
        labs[labs==labs[-1,-1,0]] = 0
        labs[labs==labs[-1,-1,-1]] = 0

        lungs = labs > 0
        self.lungs = lungs
        #self.body = (labs == 80)
        return misc.resize_to_shape(lungs, self.orig_shape)

    def dist_to_surface(self):
        if self.body is None:
            self.get_body()
        ld = scipy.ndimage.morphology.distance_transform_edt(self.body)

        return misc.resize_to_shape(ld, self.orig_shape)

    def dist_to_lungs(self):
        if self.lungs is None:
            self.get_lungs()

        ld = scipy.ndimage.morphology.distance_transform_edt(1 - self.lungs)
        return misc.resize_to_shape(ld, self.orig_shape)

    def dist_to_spine(self):
        if self.spine is None:
            self.get_spine()
        ld = scipy.ndimage.morphology.distance_transform_edt(1 - self.spine)
        return misc.resize_to_shape(ld, self.orig_shape)

    def find_symmetry(self, degrad=5, return_img=False):
        img = np.sum(self.data3dr > 430, axis=0)
        tr0, tr1, angle = find_symmetry(img, degrad)
        self.angle = angle
        self.symmetry_point = np.array([tr0, tr1])
        if return_img:
            return img


    def dist_sagittal(self, degrad=5):
        if self.angle is None:
            self.find_symmetry()
        rldst = np.ones(self.orig_shape, dtype=np.int16)
        symmetry_point_orig_res = self.symmetry_point * self.working_vs[1:] / self.voxelsize_mm[1:].astype(np.double)

        z = split_with_line(symmetry_point_orig_res, self.angle , self.orig_shape[1:])
        # print 'z  ', np.max(z), np.min(z)

        for i in range(self.orig_shape[0]):
            rldst[i ,: ,:] = z

        # rldst = scipy.ndimage.morphology.distance_transform_edt(rldst) - int(spine_mean[2])
        # return misc.resize_to_shape(rldst, self.orig_shape)
        return rldst

    def dist_coronal(self):
        if self.spine is None:
            self.get_spine()
        if self.angle is None:
            self.find_symmetry()
        spine_mean = np.mean(np.nonzero(self.spine), 1)
        rldst = np.ones(self.orig_shape, dtype=np.int16)
        # rldst = np.ones(self.data3dr.shape, dtype=np.int16)
        # rldst[:, 0, :] = 0

        spine_center = spine_mean * self.working_vs / self.voxelsize_mm.astype(np.double)

        # rldst = scipy.ndimage.morphology.distance_transform_edt(rldst) - int(spine_mean[1])

        z = split_with_line(spine_center[1:], self.angle + 90 , self.orig_shape[1:])
        for i in range(self.orig_shape[0]):
            rldst[i ,: ,:] = z

        return rldst

    def dist_axial(self):
        if self.diaphragm_mask is None:
            self.get_diaphragm_mask()
        axdst = np.ones(self.data3dr.shape, dtype=np.int16)
        axdst[0 ,: ,:] = 0
        iz, ix, iy = np.nonzero(self.diaphragm_mask)
        # print 'dia level ', self.diaphragm_mask_level

        axdst = scipy.ndimage.morphology.distance_transform_edt(axdst) - int(self.diaphragm_mask_level)
        return misc.resize_to_shape(axdst, self.orig_shape)



    def dist_diaphragm(self):
        if self.diaphragm_mask is None:
            self.get_diaphragm_mask()
        dst = (scipy.ndimage.morphology.distance_transform_edt(
                self.diaphragm_mask)
               -
               scipy.ndimage.morphology.distance_transform_edt(
                1 - self.diaphragm_mask)
              )
        return io3d.misc.resize_to_shape(dst, self.orig_shape)

    def _get_ia_ib_ic(self, axis):
        """
        according to axis gives order of of three dimensions
        :param axis: 0, 1 or 2 is allowed
        :return:
        """
        if axis == 0:
            ia = 0
            ib = 1
            ic = 2
        elif axis == 1:
            ia = 1
            ib = 0
            ic = 2
        elif axis == 2:
            ia = 2
            ib = 0
            ic = 1
        else:
            logger.error('Unrecognized axis')


        return ia, ib, ic

    def _filter_diaphragm_profile_image_remove_outlayers(self, profile, axis=0, tolerance=80):
        # import bottleneck
        # tolerance * 1.5mm

        med = np.median(profile[profile > 0])
        profile[np.abs(profile - med) > tolerance] = 0
        return profile

    def get_diaphragm_profile_image_with_empty_areas(self, axis=0):
        if self.lungs is None:
            self.get_lungs()
        if self.spine is None:
            self.get_spine()
        if self.angle is None:
            self.find_symmetry()
        axis = 0
        data = self.lungs
        ia, ib, ic = self._get_ia_ib_ic(axis)

        # gradient
        gr = scipy.ndimage.filters.sobel(data.astype(np.int16), axis=ia)
        # seg = (np.abs(ss.dist_coronal()) > 20).astype(np.uint8) + (np.abs(ss.dist_sagittal()) > 20).astype(np.uint8)
        grt = gr > 12
        # seg = (np.abs(self.dist_sagittal()) > zero_stripe_width).astype(np.uint8)
        # grt = grt * seg
        # nalezneme nenulove body
        nz = np.nonzero(grt)

        # udelame z 3d matice placku, kde jsou nuly tam, kde je nic a jinde jsou
        # z-tove souradnice
        flat = np.zeros([grt.shape[ib], grt.shape[ic]])
        flat[(nz[ib], nz[ic])] = [nz[ia]]


        # symmetry_point_orig_res = self.symmetry_point * self.working_vs[1:] / self.voxelsize_mm[1:].astype(np.double)
        # odstranime z dat pruh kolem srdce. Tam byva obcas jicen, nebo je tam oblast nad srdcem
        # symmetry_point_pixels = self.symmetry_point/ self.working_vs[1:]
        flat[flat==0] = np.NaN
        return flat

    def remove_pizza(self, flat, zero_stripe_width=10, alpha0=-20, alpha1=40 ):
        """
        Remove circular sector from the image with center in spine
        :param flat: input 2D image
        :param zero_stripe_width: offset to pivot point. Pizza line is zero_stripe_width far
        :param alpha0: Additional start angle relative to detected orientation
        :param alpha1: Additional end angle relative to detected orientation
        :return:
        """
        spine_mean = np.mean(np.nonzero(self.spine), 1)
        spine_mean = spine_mean[1:]

        z1 = split_with_line(spine_mean, self.angle + alpha1, flat.shape)
        z2 = split_with_line(spine_mean, self.angle + alpha0, flat.shape)

        z1 = (z1 > zero_stripe_width).astype(np.int8)
        z2 = (z2 < -zero_stripe_width).astype(np.int8)
        # seg = (np.abs(z) > zero_stripe_width).astype(np.int)
        seg = (z1 * z2)

        flat [seg>0]= np.NaN # + seg*10
        # print 'sp ', spine_mean
        # print 'sporig ', symmetry_point_orig_res
        # flat = seg

        return flat

    def filter_remove_outlayers(self, flat, minimum_value=0):
        """
        Remove outlayers using ellicptic envelope from scikits learn
        :param flat:
        :param minimum_value:
        :return:
        """
        from sklearn.covariance import EllipticEnvelope
        flat0 = flat.copy()
        flat0[np.isnan(flat)] = 0
        x,y = np.nonzero(flat0)
        # print np.prod(flat.shape)
        # print len(y)

        z = flat[(x,y)]

        data = np.asarray([x,y,z]).T

        clf = EllipticEnvelope(contamination=.1)
        clf.fit(data)
        y_pred = clf.decision_function(data)


        out_inds = y_pred < minimum_value
        flat[(x[out_inds], y[out_inds])] = np.NaN
        return flat


    def filter_ignoring_nan(self, flat, kernel_size_mm=None , max_dist_mm=30):
        """
        Compute filtered plane and removes pixels wiht distance grater then max_dist_mm

        :param flat:
        :param kernel_size_mm:
        :param max_dist_mm:
        :return:
        """

        if kernel_size_mm is None:
            kernel_size_mm = [150, 150]

        # kernel_size must be odd - lichý
        kernel_size = np.asarray(kernel_size_mm) / self.working_vs[1:]
        # print 'ks1 ', kernel_size
        odd = kernel_size % 2
        kernel_size = kernel_size + 1 - odd
        # print 'ks2 ', kernel_size

        # metoda 1
        kernel = np.ones(kernel_size)
        kernel = kernel / (1.0 * np.prod(kernel_size))
        # flat = scipy.ndimage.filters.convolve(flat, kernel)


        # metoda 2
        # # flat = flat.reshape([flat.shape[0], flat.shape[1], 1])
        import astropy.convolution
        flat_out = astropy.convolution.convolve(flat, kernel, boundary='extend')

        too_bad_pixels = np.abs(flat_out - flat) > (max_dist_mm/self.working_vs[0])


        flat[too_bad_pixels] = np.NaN
        # metoda 3
        # doplnime na nenulova mista střední hodnotu
        # flat_mask = np.isnan(flat)
        #
        # mn = np.mean(flat[flat_mask == False])
        #
        # flat_copy = flat.copy()
        # flat_copy[flat_mask] = mn
        #
        # flat_copy = scipy.ndimage.filters.gaussian_filter(flat_copy, sigma=sigma)
        # flat = flat_copy






        return flat


    def get_diaphragm_profile_image(self, axis=0, preprocessing=True):
        flat = self.get_diaphragm_profile_image_with_empty_areas(axis)

        if preprocessing:
            flat = self.remove_pizza(flat)
            flat = self._filter_diaphragm_profile_image_remove_outlayers(flat)
            flat = self.filter_ignoring_nan(flat)
            flat = self.filter_remove_outlayers(flat)



        # jeste filtrujeme ne jen podle stredni hodnoty vysky, ale i v prostoru
        # flat0 = flat==0
        # flat[flat0] = None
        #
        # flat = scipy.ndimage.filters.median_filter(flat, size=(40,40))
        # flat[flat0] = 0


        # something like interpolation
        # doplnime praznda mista v ploche mape podle nejblizsi oblasi
        # filter with big mask
        ou = fill_nan_with_nearest(flat.copy())
        ou = scipy.ndimage.filters.gaussian_filter(ou, sigma=5)


        # get back valid values on its places
        valid_flat_inds = 1 - np.isnan(flat)
        ou[valid_flat_inds] = flat[valid_flat_inds]
        # flat = ou

        ou = fill_nan_with_nearest(ou)
        # overal filter
        ou = scipy.ndimage.filters.median_filter(ou, size=(5,5))
        ou = scipy.ndimage.filters.gaussian_filter(ou, sigma=3)

        # ou = self.__filter_diaphragm_profile_image(ou, axis)
        return ou


    def __filter_diaphragm_profile_image(self, profile, axis=0):
        """
        filter do not go down in compare to pixel near to the back
        :param profile:
        :param axis:
        :return:
        """
        if axis == 0:

            profile_w = profile.copy()

            # profile_out = np.zeros(profile.shape)
            for i in range(profile_w.shape[0] -1 , 0 , -1):
                profile_line_0 = profile_w[i, :]
                profile_line_1 = profile_w[i - 1, :]
                where_is_bigger = profile_line_1  < (profile_line_0 - 0)
            #     profile_line_out[where_is_bigger] = profile_line_0[where_is_bigger]
                profile_w[i - 1, where_is_bigger] = profile_line_0[where_is_bigger]
                profile_w[i - 1, np.negative(where_is_bigger)] = profile_line_1[np.negative(where_is_bigger)]
            #     profile_out[where_is_bigger, :] = profile_line_1
        else:
            logger.error('other axis not implemented yet')

        return profile_w
        # plt.imshow(profile_w, cmap='jet')

    def get_diaphragm_mask(self, axis=0):
        if self.lungs is None:
            self.get_lungs()
        ia, ib, ic = self._get_ia_ib_ic(axis)
        data = self.lungs
        ou = self.get_diaphragm_profile_image(axis=axis)
        # reconstruction mask array
        mask = np.zeros(data.shape)
        for i in range(mask.shape[ia]):
            if ia == 0:
                mask[i,:,:] = ou > i
            elif ia == 1:
                mask[:,i,:] = ou > i
            elif ia == 2:
                mask[:,:,i] = ou > i

        self.diaphragm_mask = mask

        # maximal point is used for axial ze
        # ro plane
        self.diaphragm_mask_level = np.median(ou)
        self.center0 = self.diaphragm_mask_level * self.working_vs[0]

        return misc.resize_to_shape(self.diaphragm_mask, self.orig_shape)

    def get_center(self):
        self.get_diaphragm_mask()
        self.get_spine()

        self.center = np.array([self.diaphragm_mask_level, self.spine_center[0], self.spine_center[1]])
        self.center_mm = self.center * self.working_vs
        self.center_orig = self.center * self.voxelsize_mm / self.working_vs.astype(np.double)

        return self.center_orig


def prepare_images(imin0, pivot):
    imin1 = imin0[:,::-1]
    img=imin0

    padX0 = [img.shape[1] - pivot[1], pivot[1]]
    padY0 = [img.shape[0] - pivot[0], pivot[0]]
    imgP0 = np.pad(img, [padY0, padX0], 'constant')


    img = imin1
    padX1 = [pivot[1], img.shape[1] - pivot[1]]
    padY1 = [img.shape[0] - pivot[0], pivot[0]]
    imgP1 = np.pad(img, [padY1, padX1], 'constant')
    return imgP0, imgP1

def find_symmetry_parameters(imin0, trax, tray, angles):
    vals = np.zeros([len(trax), len(tray), len(angles)])
#     angles_vals = []



    for i, x in enumerate(trax):
#         print 'x ', x
        for j, y in enumerate(tray):
            try:
                img = imin0
                pivot=[x,y]
                imgP0, imgP1 = prepare_images(imin0, pivot)

    #             print 'y ', y
                for k, angle in enumerate(angles):

    #             print angle
#                 print imin0.shape, imin.shape, angle, x, y
#                     imr=rotateImage(imin.astype(int), angle, [imin0.shape[0] - x, y])
#                     dif = (imr-imin0)**2
#                     sm = np.sum(dif)
                    imr = scipy.ndimage.rotate(imgP1, angle, reshape=False)
                    dif = (imgP0-imr)**2
                    sm = np.sum(dif)
                    vals[i,j,k] = sm
            except:
                vals[i,j,:] = np.inf
    #             angles_vals.append(sm)

    # am = np.argmin(angles_vals)
    am = np.unravel_index(np.argmin(vals), vals.shape)
    # print am, ' min ', np.min(vals)

    return trax[am[0]], tray[am[1]], angles[am[2]]

def find_symmetry(img, degrad=5):
    imin0r = scipy.misc.imresize(img, np.asarray(img.shape)/degrad)

    angles = range(-180,180,15)
    trax = range(1, imin0r.shape[0],10)
    tray = range(1, imin0r.shape[1],10)



    tr0, tr1, ang = find_symmetry_parameters(imin0r, trax, tray, angles)


    # fine measurement
    angles = range(ang-20,ang+20, 3)
    trax = range(np.max([tr0-20, 1]), tr0+20, 3)
    tray = range(np.max([tr1-20, 1]), tr1+20, 3)

    tr0, tr1, ang = find_symmetry_parameters(imin0r, trax, tray, angles)

    angle = 90-ang/2.0

    return tr0*degrad, tr1*degrad, angle


# Rozděl obraz na půl
def split_with_line(point, orientation, imshape, degrees=True):
    """
    :arg point:
    :arg orientation: angle or oriented vector
    :arg degrees: if is set to True inptu angle is expected to be in radians, default True
    """

    if np.isscalar(orientation):
        angle = orientation
        if degrees:
            angle = np.radians(angle)

        # kvadranty
        angle = angle % (2* np.pi)
               # kvadranty
        angle = angle % (2* np.pi)
        # print np.degrees(angle)
        if (angle > (0.5*np.pi)) and (angle < (1.5*np.pi)):

            zn = -1
        else:
            zn = 1

        vector = [np.tan(angle), 1]
        # vector = [np.tan(angle), 1]
    else:
        vector = orientation

    vector = vector / np.linalg.norm(vector)
    x, y = np.mgrid[:imshape[0], :imshape[1]]
#     k = -vector[1]/vector[0]
#     z = ((k * (x - point[0])) + point[1] - y)
    a = vector[1]
    b = -vector[0]

    c = -a * point[0] - b * point[1]


    z = zn * (a * x + b * y + c) / (a**2 + b**2)**0.5
    return z

def fill_nan_with_nearest(flat):
    indices = scipy.ndimage.morphology.distance_transform_edt(np.isnan(flat), return_indices=True, return_distances=False)
    # indices = scipy.ndimage.morphology.distance_transform_edt(flat==0, return_indices=True, return_distances=False)
    ou = flat[(indices[0],indices[1])]
    return ou

def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description=
            'Segmentation of bones, lungs and heart.')
    parser.add_argument('-i','--datadir',
            default=None,
            help='path to data dir')
    parser.add_argument('-o','--output',
            default=None,
            help='output file')

    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-ss', '--segspine', action='store_true',
            help='run spine segmentaiton')
    parser.add_argument('-sb', '--segbody', action='store_true',
                        help='run body segmentaiton')
    parser.add_argument('-exd', '--exampledata', action='store_true',
            help='run unittest')
    parser.add_argument('-so', '--show_output', action='store_true',
            help='Show output data in viewer')
    args = parser.parse_args()



    if args.debug:
        logger.setLevel(logging.DEBUG)


    if args.exampledata:

        args.dcmdir = '../sample_data/liver-orig001.raw'

#    if dcmdir == None:

    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
    #data3d, metadata = dcmr.dcm_read_from_dir(args.dcmdir)
    import io3d

    data3d , metadata = io3d.datareader.read(args.datadir, dataplus_format=False)

    bn = BodyNavigation(data3d = data3d,
            voxelsize_mm = metadata['voxelsize_mm'],
            )


    seg = np.zeros(data3d.shape, dtype=np.int8)
    #sseg.orientation()
    if args.segspine:
        seg += bn.get_spine()
    if args.segspine:
        seg += bn.get_body()


    #print("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    #igc = pycut.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()

    #igc.make_gc()
    #igc.show_segmentation()

    # volume
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)


    #pyed = sed3.sed3(oseg.data3d, contour = oseg.segmentation)
    #pyed.show()

    if args.show_output:
        import sed3
        sed3.show_slices(data3d, contour=seg)


    #savestring = raw_input ('Save output data? (y/n): ')
    #sn = int(snstring)
    if args.output is not None: # savestring in ['Y','y']:
        md = {
            'voxelsize_mm': metadata['voxelsize_mm'],
            'segmentation': seg, }

        io3d.write(data3d, args.output, metadata=md)

    #output = segmentation.vesselSegmentation(oseg.data3d, oseg.orig_segmentation)


if __name__ == "__main__":
    main()
