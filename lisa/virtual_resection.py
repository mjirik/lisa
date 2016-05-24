#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
# from ..extern.sed3 import sed3
# import featurevector

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.ndimage
# import vtk
import argparse


# from PyQt4 import QtCore, QtGui
# from PyQt4.QtGui import *
# from PyQt4.QtCore import Qt
# from PyQt4.QtGui import QApplication
# from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
#     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
#     QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup

# ----------------- my scripts --------
import misc
import sed3
# import show3
import qmisc


def Rez_podle_roviny(plane, data, voxel):

    a = plane.GetNormal()[0] * voxel[0]
    b = plane.GetNormal()[1] * voxel[1]
    c = plane.GetNormal()[2] * voxel[2]
    xx = plane.GetOrigin()[0] / voxel[0]
    yy = plane.GetOrigin()[1] / voxel[1]
    zz = plane.GetOrigin()[2] / voxel[2]
    d = -(a * xx) - (b * yy) - (c * zz)
    mensi = 0
    vetsi = 0
    mensi_objekt = 0
    vetsi_objekt = 0
    print 'x: ', a, ' y: ', b, ' z: ', c
    print('Pocitani rezu...')
    prava_strana = np.ones((data.shape[0], data.shape[1], data.shape[2]))
    leva_strana = np.ones((data.shape[0], data.shape[1], data.shape[2]))
    dimension = data.shape
    for x in range(dimension[0]):
        for y in range(dimension[1]):
            for z in range(dimension[2]):
                rovnice = a * x + b * y + c * z + d
                if((rovnice) <= 0):
                    mensi = mensi + 1
                    if(data[x][y][z] == 1):
                        mensi_objekt = mensi_objekt + 1
                    leva_strana[x][y][z] = 0
                else:
                    vetsi = vetsi + 1
                    if(data[x][y][z] == 1):
                        vetsi_objekt = vetsi_objekt + 1
                    prava_strana[x][y][z] = 0
    leva_strana = leva_strana * data
    objekt = mensi_objekt + vetsi_objekt
    odstraneni_procenta = ((100 * mensi_objekt) / objekt)
    print leva_strana

    return leva_strana, odstraneni_procenta


# ----------------------------------------------------------
def cut_editor_old(data):

    pyed = sed3.sed3qt(data['segmentation'])
    pyed.exec_()

    return pyed.seeds


def split_vessel(data, seeds):

    split_obj0 = seeds
    split_obj = split_obj0.copy()

    vessels = data['segmentation'] == data['slab']['porta']
    vesselstmp = vessels
    sumall = np.sum(vessels == 1)

    # split_obj = scipy.ndimage.binary_dilation(split_obj, iterations = 5 )
    # vesselstmp = vessels * (1 - split_obj)

    lab, n_obj = scipy.ndimage.label(vesselstmp)
    print(n_obj)

    # while n_obj < 2 :
# dokud neni z celkoveho objektu ustipnuto alespon 80 procent
    while np.sum(lab == qmisc.max_area_index(lab, n_obj)) > (0.95 * sumall):

        split_obj = scipy.ndimage.binary_dilation(split_obj, iterations=3)
        vesselstmp = vessels * (1 - split_obj)

        lab, n_obj = scipy.ndimage.label(vesselstmp)

    # všechny objekty, na které se to rozpadlo
    # pyed = sed3.sed3(lab)
    # pyed.show()
    obj1 = get_biggest_object(lab)

# vymaz nejvetsiho
    lab[obj1 == 1] = 0
    obj2 = get_biggest_object(lab)
    # from PyQt4.QtCore import pyqtRemoveInputHook
    # pyqtRemoveInputHook()
    # import ipdb; ipdb.set_trace() # BREAKPOINT

    lab = obj1 + 2 * obj2
    cut_by_user = split_obj0
    return lab, cut_by_user


def Resekce_podle_bodu(data, seeds):
    lab, cut = split_vessel(data, seeds)
    segm, dist1, dist2 = split_organ_by_two_vessels(data, lab)
    data = virtual_resection_visualization(data, segm, dist1, dist2, cut)
    return data


def cut_editor(data, inputfile):
    # @TODO ošetřit modul viewer viz issue #69
    import viewer3
    # global normal,coordinates
    viewer = viewer3.Viewer(inputfile, 'View')
    # zobrazovani jater v kodu
    viewer.prohlizej(data, 'View', 'liver')

    # mesh = viewer.generate_mesh(segmentation,voxelsize_mm,degrad)
    # viewer.View(mesh,False)
    # viewer.buttons(window,grid)
    # print(viewer.normal)
    # print(viewer.coordinates)

    '''
    Funkce vrací trojrozměrné porobné jako data['segmentation']
    v data['slab'] je popsáno, co která hodnota znamená
    labels = []
    segmentation = segmentation[::degrad,::degrad,::degrad]
    print("Generuji data...")
    segmentation = segmentation[:,::-1,:]
    mesh_data = seg2fem.gen_mesh_from_voxels_mc(segmentation,
        voxelsize_mm*degrad)
    print("Done")
    if True:
        mesh_data.coors = seg2fem.smooth_mesh(mesh_data)
    vtk_file = "mesh_geom.vtk"
    mesh_data.write(vtk_file)
    app = QApplication(sys.argv)
    #view = viewer3.QVTKViewer(vtk_file,'Cut')
    '''

    # normal = viewer3.normal_and_coordinates().set_normal()
    # coordinates = viewer3.normal_and_coordinates().set_coordinates()
    # return normal,coordinates
    pass


def change(data, name):
    # data['segmentation'][vessels == 2] = data['slab']['porta']
    segmentation = data['segmentation']
    cut_editor(segmentation == data['slab'][name])


def resection(data, name=None, use_old_editor=False,
              interactivity=True, seeds=None):
    if use_old_editor:
        return resection_old(data, interactivity=interactivity, seeds=seeds)
    else:
        return resection_new(data, name)


def resection_old(data, interactivity=True, seeds=None):
    if interactivity:
        print ("Select cut")
        seeds = cut_editor_old(data)
    elif seeds is None:
        logger.error('seeds is None and interactivity is False')
        return None

    # seeds[56][60][78] = 1
    lab, cut = split_vessel(data, seeds)
    segm, dist1, dist2 = split_organ_by_two_vessels(data, lab)
# TODO split this function from visualization
    data = virtual_resection_visualization(data, segm, dist1,
                                           dist2, cut,
                                           interactivity=interactivity)
    return data


def split_organ_by_two_vessels(data, lab):
    """
    Input of function is ndarray with 2 labeled vessels and data.
    Output is segmented organ by vessls using minimum distance criterium.
    """
    l1 = 1
    l2 = 2

    # dist se tady počítá od nul jenom v jedničkách
    # dist1 = scipy.ndimage.distance_transform_edt(
    #     lab != l1,
    #     sampling=data['voxelsize_mm']
    # )
    # dist2 = scipy.ndimage.distance_transform_edt(
    #     lab != l2,
    #     sampling=data['voxelsize_mm']
    # )
    import skfmm
    dist1 = skfmm.distance(
        lab != l1,
        dx=data['voxelsize_mm']
    )
    dist2 = skfmm.distance(
        lab != l2,
        dx=data['voxelsize_mm']
    )
    # print 'skfmm'
    # from PyQt4.QtCore import pyqtRemoveInputHook; pyqtRemoveInputHook()
    # import ipdb; ipdb.set_trace()

    # from PyQt4.QtCore import pyqtRemoveInputHook
    # pyqtRemoveInputHook()
    # import ipdb; ipdb.set_trace() # BREAKPOINT

    # segm = (dist1 < dist2) * (data['segmentation'] != data['slab']['none'])
    segm = (((data['segmentation'] != 0) * (dist1 < dist2)).astype('int8') +
            (data['segmentation'] != 0).astype('int8'))

    return segm, dist1, dist2


def virtual_resection_visualization(data, segm, dist1, dist2, cut,
                                    interactivity=True):
    v1, v2 = liver_spit_volume_mm3(segm, data['voxelsize_mm'])

    if interactivity:
        print "Liver volume: %.4g l" % ((v1 + v2) * 1e-6)
        print "volume1: %.4g l  (%.3g %%)" % (
            (v1) * 1e-6, 100 * v1 / (v1 + v2))
        print "volume2: %.4g l  (%.3g %%)" % (
            (v2) * 1e-6, 100 * v2 / (v1 + v2))

    # pyed = sed3.sed3(segm)
    # pyed.show()
    # import pdb; pdb.set_trace()
    linie = (((data['segmentation'] != 0) *
              (np.abs(dist1 - dist2) < 1))).astype(np.int8)
    linie_vis = 2 * linie
    linie_vis[cut == 1] = 1
    linie_vis = linie_vis.astype(np.int8)
    if interactivity:
        pyed = sed3.sed3qt(
            data['data3d'],
            seeds=linie_vis,
            contour=(data['segmentation'] != 0))
        # pyed.show()
        pyed.exec_()

    # import pdb; pdb.set_trace()

    # show3.show3(data['segmentation'])

    slab = {
        'liver': 1,
        'porta': 2,
        'resected_liver': 3,
        'resected_porta': 4}

    slab.update(data['slab'])

    data['slab'] = slab

    data['slab']['resected_liver'] = 3
    data['slab']['resected_porta'] = 4

    mask_resected_liver = (
        (segm == 1) & (data['segmentation'] == data['slab']['liver']))
    mask_resected_porta = (
        (segm == 1) & (data['segmentation'] == data['slab']['porta']))

    data['segmentation'][mask_resected_liver] = \
        data['slab']['resected_liver']
    data['segmentation'][mask_resected_porta] = \
        data['slab']['resected_porta']

    logger.debug('resection_old() end')
    return data


def resection_new(data, name):

    # data['segmentation'][vessels == 2] = data['slab']['porta']
    # segmentation = data['segmentation']
    # print(data['slab'])
    change(data, name)

    # print data["slab"]
    # change(segmentation == data['slab']['porta'])
    # lab = cut_editor(segmentation == data['slab']['porta'])


def get_biggest_object(data):
    return qmisc.get_one_biggest_object(data)


def liver_spit_volume_mm3(segm, voxelsize_mm):
    """
    segm: 0 - nothing, 1 - remaining tissue, 2 - resected tissue
    """
    voxelsize_mm3 = np.prod(voxelsize_mm)
    v1 = np.sum(segm == 1) * voxelsize_mm3
    v2 = np.sum(segm == 2) * voxelsize_mm3

    return v1, v2


def View(name):
    data = misc.obj_from_file("out", filetype='pickle')
    resection(data, name)


if __name__ == "__main__":
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

#    SectorDisplay2__()
    # logger.debug('input params')
    # input parser
    parser = argparse.ArgumentParser(description='Segment vessels from liver')
    parser.add_argument('-pkl', '--picklefile',
                        help='input file from organ_segmentation')
    parser.add_argument('-oe', '--use_old_editor',  action='store_true',
                        help='use an old editor for vessel cut')
    parser.add_argument('-o', '--outputfile',  default=None,
                        help='output file')
    parser.add_argument('-oo', '--defaultoutputfile',  action='store_true',
                        help='"vessels.pickle" as output file')
    parser.add_argument('-d', '--debug',  action='store_true',
                        help='Debug mode')
    args = parser.parse_args()

    if (args.picklefile or args.vtkfile) is None:
        raise IOError('No input data!')

    data = misc.obj_from_file(args.picklefile, filetype='pickle')
    ds = data['segmentation'] == data['slab']['liver']
    pozice = np.where(ds == 1)
    a = pozice[0][0]
    b = pozice[1][0]
    c = pozice[2][0]
    ds = False
    # print "vs ", data['voxelsize_mm']
    # print "vs ", data['voxelsize_mm']
    if args.debug:
        logger.setLevel(logging.DEBUG)
    # seg = np.zeros([100,100,100])
    # seg [50:80, 50:80, 60:75] = 1
    # seg[58:60, 56:72, 66:68]=2
    # dat = np.random.rand(100,100,100)
    # dat [50:80, 50:80, 60:75] =  dat [50:80, 50:80, 60:75] + 1
    # dat [58:60, 56:72, 66:68] =  dat  [58:60, 56:72, 66:68] + 1
    # slab = {'liver':1, 'porta':2, 'portaa':3, 'portab':4}
    # data = {'segmentation':seg, 'data3d':dat, 'slab':slab}
    name = 'porta'

    # cut_editor(data,args.inputfile)
    if args.use_old_editor:
        resection(data, name, use_old_editor=args.use_old_editor)
    else:
        cut_editor(data, args.picklefile)
    # print normal
    # print coordinates

    defaultoutputfile = "05-resection.pkl"
    if args.defaultoutputfile:
        args.outputfile = defaultoutputfile

    if args.outputfile is None:

        savestring = raw_input('Save output data? (y/n): ')
        if savestring in ['Y', 'y']:

            misc.obj_to_file(data, defaultoutputfile, filetype='pickle')
    else:
        misc.obj_to_file(data, args.outputfile, filetype='pickle')
