#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Generator of histology report

"""
import logging
logger = logging.getLogger(__name__)

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))


import argparse
import numpy as np
import scipy.ndimage
from io3d import datawriter
import misc

# import datareader
import sed3 as se

import gen_vtk_tree

import vtk
from vtk.util import numpy_support

from datetime import datetime


class VolumeTreeGenerator:
    """
    This generator is called by generateTree() function as a general form.
    Other similar generator is used for generating LAR outputs.
    """
    def __init__(self, gtree):
        self.shape = gtree.shape
        self.data3d = np.zeros(gtree.shape, dtype=np.int)
        self.voxelsize_mm = gtree.voxelsize_mm

    def add_cylinder(self, p1m, p2m, rad, id):
        """
        Funkce na vykresleni jednoho segmentu do 3D dat
        """

        cyl_data3d = np.ones(self.shape, dtype=np.bool)
        # prvni a koncovy bod, ve pixelech
        p1 = [p1m[0] / self.voxelsize_mm[0], p1m[1] /
              self.voxelsize_mm[1], p1m[2] / self.voxelsize_mm[2]]
        p2 = [p2m[0] / self.voxelsize_mm[0], p2m[1] /
              self.voxelsize_mm[1], p2m[2] / self.voxelsize_mm[2]]
        logger.debug(
            "p1_px: " + str(p1[0]) + " " + str(p1[1]) + " " + str(p1[2]))
        logger.debug(
            "p2_px: " + str(p2[0]) + " " + str(p2[1]) + " " + str(p2[2]))
        logger.debug("radius_mm:" + str(rad))

        # vzdalenosti mezi prvnim a koncovim bodem (pro jednotlive osy)
        pdiff = [abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2])]

        # generovani hodnot pro osu segmentu
        num_points = max(pdiff) * \
            2  # na jeden "pixel nejdelsi osy" je 2 bodu primky (shannon)
        zvalues = np.linspace(p1[0], p2[0], num_points)
        yvalues = np.linspace(p1[1], p2[1], num_points)
        xvalues = np.linspace(p1[2], p2[2], num_points)

        # drawing a line
        for i in range(0, len(xvalues)):
            try:
                cyl_data3d[int(zvalues[i])][int(yvalues[i])][int(xvalues[i])] = 0
            except:
                import traceback
                traceback.print_exc()
                print "except in drawing line"
                import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

        # cuting size of 3d space needed for calculating distances (smaller ==
        # a lot faster)
        cut_up = max(
            0, round(min(p1[0], p2[0]) - (rad / min(self.voxelsize_mm)) - 2))
        # ta 2 je kuli tomu abyh omylem nurizl
        cut_down = min(self.shape[0], round(
            max(p1[0], p2[0]) + (rad / min(self.voxelsize_mm)) + 2))
        cut_yu = max(
            0, round(min(p1[1], p2[1]) - (rad / min(self.voxelsize_mm)) - 2))
        cut_yd = min(self.shape[1], round(
            max(p1[1], p2[1]) + (rad / min(self.voxelsize_mm)) + 2))
        cut_xl = max(
            0, round(min(p1[2], p2[2]) - (rad / min(self.voxelsize_mm)) - 2))
        cut_xr = min(self.shape[2], round(
            max(p1[2], p2[2]) + (rad / min(self.voxelsize_mm)) + 2))
        logger.debug("cutter_px: z_up-" + str(cut_up) + " z_down-" + str(cut_down) + " y_up-" + str(
            cut_yu) + " y_down-" + str(cut_yd) + " x_left-" + str(cut_xl) + " x_right-" + str(cut_xr))
        cyl_data3d_cut = cyl_data3d[
            int(cut_up):int(cut_down),
            int(cut_yu):int(cut_yd),
            int(cut_xl):int(cut_xr)]

        # calculating distances
        # spotrebovava naprostou vetsinu casu (pro 200^3  je to kolem 1.2
        # sekundy, proto jsou data osekana)
        lineDst = scipy.ndimage.distance_transform_edt(
            cyl_data3d_cut, self.voxelsize_mm)

        # zkopirovani vyrezu zpet do celeho rozsahu dat
        for z in xrange(0, len(cyl_data3d_cut)):
            for y in xrange(0, len(cyl_data3d_cut[z])):
                for x in xrange(0, len(cyl_data3d_cut[z][y])):
                    if lineDst[z][y][x] <= rad:
                        iX = int(z + cut_up)
                        iY = int(y + cut_yu)
                        iZ = int(x + cut_xl)
                        self.data3d[iX][iY][iZ] = 1

    def get_output(self):
        return self.data3d

    def save(self, outputfile, filetype='pklz'):
        data = {
            'data3d': self.data3d,
            'voxelsize_mm': self.voxelsize_mm
        }

        misc.obj_to_file(data, outputfile, filetype=filetype)
        print "saved"
        #dw = datawriter.DataWriter()
        #dw.Write3DData(self.data3d, outputfile, filetype)

    def show(self):
        pyed = se.sed3(self.data3d)
        pyed.show()


class TreeGenerator:

    def __init__(self, generator_class=VolumeTreeGenerator, generator_params=None):
        self.rawdata = None
        self.tree_data = None
        self.data3d = None
        self.voxelsize_mm = [1, 1, 1]
        self.shape = None
        self.use_lar = False
        self.generator_class = generator_class
        self.generator_params = generator_params

    def importFromYaml(self, filename):
        rawdata = misc.obj_from_file(filename=filename, filetype='yaml')
        self.rawdata = rawdata

        try:
            # key is usually "porta" or "microstructure"
            keys = self.rawdata['graph'].keys()
            self.tree_data = self.rawdata['graph'][keys[0]]
        except:
            self.tree_data = self.rawdata['Graph']

    def generateTree(self):
        """
        | Funkce na vygenerování objemu stromu ze zadaných dat.
        | Generates output by defined generator. If VolumeTreeGenerator is used, output is data3d.
        """
# LAR init
        if self.use_lar:
            import lar_vessels
            self.lv = lar_vessels.LarVessels()

        # use generator init
        if self.generator_params is None:
            self.generator = self.generator_class(self)
        else:
            self.generator = self.generator_class(self, **generator_params)

        for cyl_id in self.tree_data:
            logger.debug("CylinderId: " + str(cyl_id))
            cyl_data = self.tree_data[cyl_id]

            # try:
            #     cyl_data = self.data['graph']['porta'][cyl_id]
            # except:
            #     cyl_data = self.data['Graph'][cyl_id]

            # prvni a koncovy bod, v mm + radius v mm
            try:
                p1m = cyl_data['nodeA_ZYX_mm']  # souradnice ulozeny [Z,Y,X]
                p2m = cyl_data['nodeB_ZYX_mm']
                rad = cyl_data['radius_mm']
                self.generator.add_cylinder(p1m, p2m, rad, cyl_id)
            except Exception, e:
                # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT

                logger.error(
                    "Segment id " + str(cyl_id) + ": error reading data from yaml!: "+str(e))
                # return

            # if self.use_lar:
            #     self.generator.add_cylinder(p1m, p2m, rad, in)
        logger.debug("cylinders generated")

        try:
            # generator could have finish() function
            self.generator.finish()
            logger.debug("joints generated")
        except:
            import traceback
            logger.debug(traceback.format_exc())
            pass

        output = self.generator.get_output()

        logger.debug("before visualization - generateTree()")
        if self.use_lar:
            self.lv.show()
        return output

    def generateTree_vtk(self):
        """
        | Funkce na vygenerování objemu stromu ze zadaných dat.
        | Veze pro generování pomocí VTK
        | !!! funguje špatně -> vstupní data musí být pouze povrchové body, jinak generuje ve výstupních datech dutiny

        """
        # get vtkPolyData
        tree_data = gen_vtk_tree.process_tree(self.rawdata['Graph'])
        polyData = gen_vtk_tree.gen_tree(tree_data)

        polyData.GetBounds()
        # bounds = polyData.GetBounds()

        white_image = vtk.vtkImageData()
        white_image.SetSpacing(self.voxelsize_mm)
        white_image.SetDimensions(self.shape)
        white_image.SetExtent(
            [0, self.shape[0] - 1, 0, self.shape[1] - 1, 0, self.shape[2] - 1])
        # origin = [(bounds[0] + self.shape[0])/2, (bounds[1] + self.shape[1])/2, (bounds[2] + self.shape[2])/2]
        # white_image.SetOrigin(origin) #neni potreba?
        # white_image.SetScalarTypeToUnsignedChar()
        white_image.AllocateScalars()

        # fill the image with foreground voxels: (still black until stecil)
        inval = 255
        outval = 0
        count = white_image.GetNumberOfPoints()
        for i in range(0, count):
            white_image.GetPointData().GetScalars().SetTuple1(i, inval)

        pol2stencil = vtk.vtkPolyDataToImageStencil()
        pol2stencil.SetInput(polyData)

        # pol2stencil.SetOutputOrigin(origin) # TOHLE BLBNE
        pol2stencil.SetOutputSpacing(self.voxelsize_mm)
        pol2stencil.SetOutputWholeExtent(white_image.GetExtent())
        pol2stencil.Update()

        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInput(white_image)
        imgstenc.SetStencil(pol2stencil.GetOutput())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(outval)
        imgstenc.Update()

        # VTK -> Numpy
        vtk_img_data = imgstenc.GetOutput()
        vtk_data = vtk_img_data.GetPointData().GetScalars()
        numpy_data = numpy_support.vtk_to_numpy(vtk_data)
        numpy_data = numpy_data.reshape(
            self.shape[0], self.shape[1], self.shape[2])
        numpy_data = numpy_data.transpose(2, 1, 0)

        self.data3d = numpy_data

    def saveToFile(self, outputfile, filetype):
        self.generator.save(outputfile, filetype)

    def show(self):
        self.generator.show()


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser reporter. Try: \
python src/gen_volume_tree.py -i ./tests/hist_stats_test.yaml'
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file, yaml file'
    )
    parser.add_argument(
        '-o', '--outputfile',
        default=None,
        help='output file, .raw, .dcm, .tiff, given by extension '
    )
    parser.add_argument(
        '-ot', '--outputfiletype',
        default='pkl',
        help='output file type.  raw, dcm, tiff, or pkl,   default is pkl, '
    )
    parser.add_argument(
        '-vs', '--voxelsize',
        default=[1.0, 1.0, 1.0],
        type=float,
        metavar='N',
        nargs='+',
        help='size of voxel (ZYX)'
    )
    parser.add_argument(
        '-ds', '--datashape',
        default=[200, 200, 200],
        type=int,
        metavar='N',
        nargs='+',
        help='size of output data in pixels for each axis (ZYX)'
    )
    parser.add_argument(
        '-g', '--generator',
        default='vol',
        type=str,
        help='Volume or surface model can be generated by use this option. \
                Use "vol", "volume" for volumetric model. For LAR surface model\
                use "lar".'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    parser.add_argument(
        '-l', '--useLar', action='store_true',
        help='Use LAR')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    startTime = datetime.now()

    generator_params = None
    if args.generator in ['vol', 'volume']:
        generator_class = VolumeTreeGenerator
    elif args.generator in ['lar']:
        import gt_lar
        generator_class = gt_lar.GTLar
    elif args.generator in ['kunes']:
        import gt_lar_kunes
        generator_class = gt_lar_kunes.GTLar
    elif args.generator in ['larsm']:
        import gt_lar_smooth
        generator_class = gt_lar_smooth.GTLarSmooth
    elif args.generator in ['lar_nojoints']:
        import gt_lar
        generator_class = gt_lar.GTLar
        generator_params = {
            'endDistMultiplicator': 0,
            'use_joints': False
        }

    tg = TreeGenerator(generator_class, generator_params)
    tg.importFromYaml(args.inputfile)
    tg.voxelsize_mm = args.voxelsize
    tg.shape = args.datashape
    tg.use_lar = args.useLar
    data3d = tg.generateTree()

    logger.info("TimeUsed:" + str(datetime.now() - startTime))
    # volume_px = sum(sum(sum(data3d)))
    # volume_mm3 = volume_px * \
    #     (tg.voxelsize_mm[0] * tg.voxelsize_mm[1] * tg.voxelsize_mm[2])
    # logger.info("Volume px:" + str(volume_px))
    # logger.info("Volume mm3:" + str(volume_mm3))

# vizualizace
    logger.debug("before visualization")
    tg.show()
    logger.debug("after visualization")

# ukládání do souboru
    if args.outputfile is not None:
        tg.saveToFile(args.outputfile, args.outputfiletype)
