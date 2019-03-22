#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import os.path
import sys

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
# @TODO remove logger debug message from the header
logger.debug("before morphology import")
from skimage import morphology

# from PyQt4 import QtCore, QtGui
# from PyQt4.QtGui import *
# from PyQt4.QtCore import Qt
# from PyQt4.QtGui import QApplication
# from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
#     QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
#     QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup

# ----------------- my scripts --------
from . import misc
import sed3
# import show3
from . import qmisc
from . import data_manipulation
import imma.image_manipulation as ima


def resection(data, name=None, method='PV',
              interactivity=True, seeds=None, **kwargs):

    """
    Main resection function.

    :param data: dictionaru with data3d, segmentation and slab key.
    :param method: "PV", "planar"
    :param interactivity: True or False, use seeds if interactivity is False
    :param seeds: used as initial interactivity state
    :param kwargs: other parameters for resection algorithm
    :return:
    """
    if method is 'PV':
        return resection_old(data, interactivity=interactivity, seeds=seeds)
    elif method is 'planar':
        return resection_planar(data, interactivity=interactivity, seeds=seeds)
    elif method is "PV_new":
        return resection_portal_vein_new(data, interactivity=interactivity, seeds=seeds, organ_label=data["slab"]["liver"], vein_label=data["slab"]["porta"])
        # return resection_portal_vein_new(data, interactivity=interactivity, seeds=seeds, **kwargs)
    else:
        return resection_with_3d_visualization(data, **kwargs)


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
    print('x: ', a, ' y: ', b, ' z: ', c)
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
    print(leva_strana)

    return leva_strana, odstraneni_procenta


# ----------------------------------------------------------
def cut_editor_old(data, label=None):
    logger.debug("editor input label: " + str(label))

    if label is None:
        contour=data['segmentation']
    else:
        if type(label) == str:
            label = data['slab'][label]
        contour=(data['segmentation'] == label).astype(np.int8)
    pyed = sed3.sed3qt(data['data3d'], contour=contour)
    pyed.exec_()

    return pyed.seeds


def split_vessel(datap, seeds, vessel_volume_threshold=0.95, dilatation_iterations=1, input_label="porta",
                 output_label1 = 1, output_label2 = 2, input_seeds_cut_label=1,
                 input_seeds_separate_label=3,
                 input_seeds_label2=None,
                 method="reach volume",
                 ):
    """

    :param datap: data plus format with data3d, segmentation, slab ...
    :param seeds: 3d ndarray same size as data3d, label 1 is place where should be vessel cuted. Label 2 points to
    the vessel with output label 1 after the segmentation
    :param vessel_volume_threshold: this parameter defines the iteration stop rule if method "reach volume is selected
    :param dilatation_iterations:
    :param input_label: which vessel should be splited
    :param output_label1: output label for vessel part marked with right button (if it is used)
    :param output_label2: ouput label for not-marked vessel part
    :param method: "separate labels" or "reach volume". The first method needs 3 input seeds and it is more stable.
    :param input_seeds_separate_label: after the segmentation the object containing this label in seeds would be labeled with
    output_label1
    :param input_seeds_label2: This parameter is usedf the method is "separate labels". After the
    segmentation the object containing this label in seeds would be labeled with output_label1.
    :return:
    """
    split_obj0 = (seeds == input_seeds_cut_label).astype(np.int8)
    split_obj = split_obj0.copy()


    # numeric_label = imma.get_nlabel(datap["slab"], input_label)
    if method == "separate labels":
        input_label = np.max(datap["segmentation"][seeds == input_seeds_label2])

    vessels = ima.select_labels(datap["segmentation"], input_label, slab=datap["slab"])

    # if type(input_label) is str:
    #     numeric_label = datap['slab'][input_label]
    # else:
    #     numeric_label = input_label
    # vessels = datap['segmentation'] == numeric_label

    vesselstmp = vessels
    sumall = np.sum(vessels == 1)

    # split_obj = scipy.ndimage.binary_dilation(split_obj, iterations = 5 )
    # vesselstmp = vessels * (1 - split_obj)

    lab, n_obj = scipy.ndimage.label(vesselstmp)
    logger.debug("number of objects " + str(n_obj))

    # while n_obj < 2 :
# dokud neni z celkoveho objektu ustipnuto alespon 80 procent
    not_complete = True
    while not_complete:
        if method == "reach volume":
            not_complete = np.sum(lab == qmisc.max_area_index(lab, n_obj)) > (vessel_volume_threshold * sumall)
        elif method == "separate labels":
            # misc.
            # imma.get_nlabel(datap["slab"], )
            # imma.select_labels(seeds,input_seeds_separate_label)
            seglab1 = np.max(lab[seeds == input_seeds_separate_label])
            seglab2 = np.max(lab[seeds == input_seeds_label2])
            if (seglab1 > 0) and (seglab2 > 0) and (seglab1 != seglab2):
                not_complete = False
        else:
            IOError("Unknown method " + str(method))

        split_obj = scipy.ndimage.binary_dilation(split_obj, iterations=dilatation_iterations)
        vesselstmp = vessels * (1 - split_obj)

        lab, n_obj = scipy.ndimage.label(vesselstmp)

    if method == "reach volume":
        # všechny objekty, na které se to rozpadlo
        # pyed = sed3.sed3(lab)
        # pyed.show()
        obj1 = get_biggest_object(lab)

    # vymaz nejvetsiho
        lab[obj1 == 1] = 0
        obj2 = get_biggest_object(lab)

        pixel = 0
        pixels = obj1[seeds == input_seeds_separate_label]
        if len(pixels) > 0:
            pixel = pixels[0]

        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace() # BREAKPOINT

        if pixel > 0:
            ol1 = output_label1
            ol2 = output_label2
        else:
            ol2 = output_label1
            ol1 = output_label2

        # first selected pixel with right button
        lab = ol1 * obj1 + ol2 * obj2
    elif method == "separate labels":
        lab = (lab == seglab1) * output_label1 + (lab == seglab2) * output_label2
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

def velikosti(a):
    # a_index = [0, 0, 0]
    # for x in range(0, len(a)):
    #     for y in range(0, len(a[0])):
    #         for z in range(0, len(a[0][0])):
    #             if a[x][y][z] == 1:
    #                 a_index[0] += 1
    #             elif a[x][y][z] == 2:
    #                 a_index[1] += 1
    #             elif a[x][y][z] == 3:
    #                 a_index[2] += 1
    mx = np.max(a)
    a_index = []
    for i in range(1, 4): # for i in range(1, mx + 1):
        sm = np.sum(a == i)
        a_index.append(sm)

    return a_index

def nejnizsi(a, b, c):
    if a > b:
        if b > c:
            return 3
        else:
            return 2
    elif b > c:
        if c > a:
            return 1
        else:
            return 3
    elif c > a:
        if a > b:
            return 2
        else:
            return 1
    else:
        print("chyba")


def resection_portal_vein_new(data, interactivity=False, seeds=None, organ_label=1, vein_label=2):
    """
    New function for portal vein segmentation
    :param data:
    :param interactivity:
    :param seeds:
    :param kwargs:
    :return:
    """
    # ed = sed3.sed3(a)
    # ed.show()

    # from PyQt4 import QtGui
    # from PyQt4.QtGui import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, \
    # QFont, QPixmap, QFileDialog
    #
    # window = QtGui.QWidget()
    # mainLayout = QVBoxLayout()
    # window.setLayout(mainLayout)
    # mainLayout.addWidget(sed3.sed3qtWidget(data['data3d'], contour=data['segmentation']))

    # zachovani puvodnich dat
    segmentation = data["segmentation"]
    data3d = data["data3d"]

    # data pouze se segmentacemi
    segm = ((data["segmentation"] == organ_label) * organ_label +
            (data["segmentation"] == vein_label) * vein_label)


    # ed = sed3.sed3(segm)
    # ed.show()

    # ufiknutí segmentace
    crinfo = qmisc.crinfo_from_specific_data(segm, [0])
    data["segmentation"] = qmisc.crop(segm, crinfo)
    data["data3d"] = qmisc.crop(data3d, crinfo)
    if seeds is not None:
        seeds = qmisc.crop(seeds, crinfo)

    # @TODO zde nahradit střeve čímkoliv smysluplnějším
    if interactivity:
        print("Select cut")
        # seeds = cut_editor_old(data)
        seeds = cut_editor_old(data)
    elif seeds is None:
        logger.error('seeds is None and interactivity is False')
        return None

    lab, cut = split_vessel(data, seeds)
    segm, dist1, dist2 = split_organ_by_two_vessels(data, lab)

    # jatra rozdeleny na 3 kusy
    a = morphology.label(segm, background=0)
    ### podmínka nefunguje
    if 3 in a: # zda se v segmentaci objevuje 3. cast
        print("slape :) :) :P")
        a_index = velikosti(segm)
        print(a_index)
        i = nejnizsi(a_index[0], a_index[1], a_index[2])
        segm = ((a == i) * (segm == 1).astype('int8') +
                (a != i)*(segm == 2).astype('int8') +
                (segm != 0).astype('int8'))

    # TODO split this function from visualization
    data = virtual_resection_visualization(data, segm, dist1,
                                           dist2, cut,
                                           interactivity=interactivity)

    # vrácení původních dat a spojení s upravenými daty
    data["data3d"] = data3d
    # orig_shape = (len(segmentation), len(segmentation[0]), len(segmentation[1]))
    data["segmentation"] = qmisc.uncrop(data["segmentation"], crinfo, orig_shape=segmentation.shape)

    #segmentation = segmentation == vein
    data["segmentation"] = (data["segmentation"] +
                            (segmentation != organ_label) * segmentation) - (segmentation == vein_label) * vein_label
    return data

def resection_old(data, interactivity=True, seeds=None):
    if interactivity:
        print("Select cut")
        seeds = cut_editor_old(data)
    elif seeds is None:
        logger.error('seeds is None and interactivity is False')
        return None

    logger.debug("unique(seeds) " + str(np.unique(seeds)))
    # seeds[56][60][78] = 1
    lab, cut = split_vessel(data, seeds)
    segm, dist1, dist2 = split_organ_by_two_vessels(data, lab)
# TODO split this function from visualization
    data = virtual_resection_visualization(data, segm, dist1,
                                           dist2, cut,
                                           interactivity=interactivity)
    return data

def resection_planar(data, interactivity, seeds=None):
    """
    Based on input seeds the cutting plane is constructed

    :param data:
    :param interactivity:
    :param seeds:
    :return:
    """
    if seeds is None:
        if interactivity:
            print("Select cut")
            seeds = cut_editor_old(data)
        else:
            logger.error('seeds is None and interactivity is False')
            return None

    segm, dist1, dist2 = split_organ_by_plane(data, seeds)
    cut = dist1**2 < 2
    # TODO split this function from visualization
    data = virtual_resection_visualization(data, segm, dist1,
                                           dist2, cut,
                                           interactivity=interactivity)
    return data

def split_organ_by_plane(data, seeds):
    """
    Based on seeds split nonzero segmentation with plane

    :param data:
    :param seeds:
    :return:
    """
    from . import geometry3d
    from . import data_manipulation
    l1 = 1
    l2 = 2

    point, vector = geometry3d.plane_fit(seeds.nonzero())
    dist1 = data_manipulation.split_with_plane(point, vector, data['data3d'].shape)
    dist2 = dist1 * -1

    segm = (((data['segmentation'] != 0) * (dist1 < dist2)).astype('int8') +
            (data['segmentation'] != 0).astype('int8'))

    return segm, dist1, dist2


def split_tissue_on_labeled_tree(labeled_branches,
                                 trunk_label, branch_labels,
                                 tissue_segmentation, neighbors_list=None,
                                 ignore_labels=None,
                                 ignore_trunk=True,
                                 on_missed_branch="split",

                                 ):
    """
    Based on pre-labeled vessel tree split surrounding tissue into two part.
    The connected sub tree is computed and used internally.

    :param labeled_branches: ndimage with labeled volumetric vessel tree.
    :param trunk_label: int
    :param branch_labels: list of ints
    :param tissue_segmentation: ndimage with bool type. Organ is True, the rest is False.
    :param ignore_trunk: True or False
    :param ignore_labels: list of labels which will be ignored
    :param on_missed_branch: str, ["split", "organ_label", exception]. Missed label is label directly connected
    to trunk but with no branch label inside.
    "split" will ignore mised label.
    "orig" will leave the original area label.
    "exception", will throw the exception.
    :return:
    """
    # bl = lisa.virtual_resection.branch_labels(oseg, "porta")

    import imma.measure
    import imma.image_manipulation
    import imma.image_manipulation as ima

    if ignore_labels is None:
        ignore_labels = []

    ignore_labels = list(ignore_labels)
    if ignore_trunk:
        ignore_labels.append(trunk_label)

    if neighbors_list is None:
        exclude = [0]
        exclude.extend(ignore_labels)
        neighbors_list = imma.measure.neighbors_list(
            labeled_branches,
            None,
            # [seglabel1, seglabel2, seglabel3],
            exclude=exclude)
    #exclude=[imma.image_manipulation.get_nlabels(slab, ["liver"]), 0])
    # ex
    # print(neighbors_list)
    # find whole branche
    # segmentations = [None] * len(branch_labels)
    segmentation = np.zeros_like(labeled_branches, dtype=int)
    new_branches = []
    connected = [None] * len(branch_labels)

    for i, branch_label in enumerate(branch_labels):
        import copy

        ignore_other_branches = copy.copy(branch_labels)
        ignore_other_branches.pop(i)
        ignore_labels_i = [0]
        ignore_labels_i.extend(ignore_other_branches)
        ignore_labels_i.extend(ignore_labels)
        connected_i = imma.measure.get_connected_labels(
            neighbors_list, branch_label, ignore_labels_i)
        # segmentations[i] = ima.select_labels(labeled_branches, connected_i).astype(np.int8)
        select = ima.select_labels(labeled_branches, connected_i).astype(np.int8)
        select = select > 0
        if np.max(segmentation[select]) > 0:
            logger.debug("Missing branch connected to branch and other branch or trunk.")
            union = (segmentation * select) > 0
            segmentation[select] = i + 1
            if on_missed_branch == "split":
                segmentation[union] = 0
            elif on_missed_branch == "orig":
                new_branche_label = len(branch_labels) + len(new_branches) + 1
                logger.debug("new branch label {}".format(new_branche_label))
                segmentation[union] = new_branche_label
                new_branches.append(new_branche_label)
            elif on_missed_branch == "exception":
                raise ValueError("Missing one vessel")
            else:
                raise ValueError("Unknown 'on_missed_label' parameter.")
        else:
            segmentation[select] = i + 1
        # error
        # else:
        #   segmentation[select] = i + 1
        connected[i] = connected_i
    seg = segmentation
    # if np.max(np.sum(segmentations, 0)) > 1:
    #     raise ValueError("Missing one vessel")
    #
    # for i, branch_label in enumerate(branch_labels):
    #     segmentations[i] = segmentations[i] * (i + 1)
    # seg = np.sum(segmentations, 0)

    # ignore_labels1 = [0, trunk_label, branch_label2]
    # ignore_labels1.extend(ignore_labels)
    # ignore_labels2 = [0, trunk_label, branch_label]
    # ignore_labels2.extend(ignore_labels)
    # connected2 = imma.measure.get_connected_labels(
    #     neighbors_list, branch_label, ignore_labels1)
    # connected3 = imma.measure.get_connected_labels(
    #     neighbors_list, branch_label2, ignore_labels2)
    #
    # # seg = ima.select_labels(segmentation, organ_label, slab).astype(np.int8)
    # seg1 = ima.select_labels(labeled_branches, connected2).astype(np.int8)
    # seg2 = ima.select_labels(labeled_branches, connected3).astype(np.int8)
    # seg = seg1 + seg2 * 2
    # if np.max(seg) > 2:
    #     ValueError("Missing one vessel")

    dseg = ima.distance_segmentation(seg)
    logger.debug("output unique labels {}".format(np.unique(dseg)))
    # organseg = ima.select_labels(segmentation, organ_label, slab).astype(np.int8)
    dseg[~tissue_segmentation.astype(np.bool)] = 0

    return dseg, connected


def split_organ_by_two_vessels(datap,
                               seeds, organ_label=1,
                               seed_label1=1, seed_label2=2,
                               weight1=1, weight2=1):
    """

    Input of function is ndarray with 2 labeled vessels and data.
    Output is segmented organ by vessls using minimum distance criterium.

    :param datap: dictionary with 3d data, segmentation, and other information
           "data3d": 3d-ndarray with intensity data
           "voxelsize_mm",
           "segmentation": 3d ndarray with image segmentation
           "slab": segmentation labels
    :param seeds: ndarray with same size as data3d
            1: first part of portal vein (or defined in seed1_label)
            2: second part of portal vein (or defined in seed2_label)
    :param weight1: distance weight from seed_label1
    :param weight2: distance weight from seed_label2

    """
    weight1 = 1 if weight1 is None else weight1

    slab = datap["slab"]
    segmentation = datap["segmentation"]
    if type(seed_label1) != list:
        seed_label1 = [seed_label1]
    if type(seed_label2) != list:
        seed_label2 = [seed_label2]
    # dist se tady počítá od nul jenom v jedničkách
    dist1 = scipy.ndimage.distance_transform_edt(
        1 - ima.select_labels(seeds, seed_label1, slab),
        # seeds != seed_label1,
        sampling=datap['voxelsize_mm']
    )
    dist2 = scipy.ndimage.distance_transform_edt(
        1 - ima.select_labels(seeds, seed_label2, slab),
        # seeds != seed_label2,
        sampling=datap['voxelsize_mm']
    )
    # import skfmm
    # dist1 = skfmm.distance(
    #     labeled != l1,
    #     dx=datap['voxelsize_mm']
    # )
    # dist2 = skfmm.distance(
    #     labeled != l2,
    #     dx=datap['voxelsize_mm']
    # )
    # print 'skfmm'
    # from PyQt4.QtCore import pyqtRemoveInputHook; pyqtRemoveInputHook()
    # import ipdb; ipdb.set_trace()

    # from PyQt4.QtCore import pyqtRemoveInputHook
    # pyqtRemoveInputHook()
    # import ipdb; ipdb.set_trace() # BREAKPOINT

    # segm = (dist1 < dist2) * (data['segmentation'] != data['slab']['none'])
    target_organ_segmentation = ima.select_labels(segmentation, organ_label, slab)
    segm = ((target_organ_segmentation * ((dist1 / weight1) > (dist2 / weight2))).astype('int8') +
            target_organ_segmentation.astype('int8'))

    return segm, dist1, dist2


def virtual_resection_visualization(data, segm, dist1, dist2, cut,
                                    interactivity=True):
    v1, v2 = liver_spit_volume_mm3(segm, data['voxelsize_mm'])

    if interactivity:
        print("Liver volume: %.4g l" % ((v1 + v2) * 1e-6))
        print("volume1: %.4g l  (%.3g %%)" % (
            (v1) * 1e-6, 100 * v1 / (v1 + v2)))
        print("volume2: %.4g l  (%.3g %%)" % (
            (v2) * 1e-6, 100 * v2 / (v1 + v2)))

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


def resection_with_3d_visualization(data, name):

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


def label_volumetric_vessel_tree(oseg, vessel_label=None, write_to_oseg=True, new_label_str_format="{}{:03d}"):
    """
    Split vessel by branches and put it in segmentation and slab.

    :param oseg: OrganSegmentation object with segmentation, voxelsize_mm and slab
    :param vessel_label: int or string label with vessel. Everything above zero is used if vessel_label is set None.
    :param write_to_oseg: Store output into oseg.segmentation if True. The slab is also updated.
    :param new_label_str_format: format of new slab
    :return:
    """
    logger.debug("vessel_label {}".format(vessel_label))
    logger.debug("python version {} {}".format(sys.version_info, sys.executable))
    import skelet3d
    if vessel_label is None:
        vessel_volume = oseg.segmentation > 0
    else:
        vessel_volume = oseg.select_label(vessel_label)

    # print(np.unique(vessel_volume))
    skel = skelet3d.skelet3d(vessel_volume)
    skan = skelet3d.SkeletonAnalyser(skel, volume_data=vessel_volume)
    skan.skeleton_analysis()
    bl = skan.get_branch_label()
    un = np.unique(bl)
    logger.debug("skelet3d branch label min: {}, max: {}, dtype: {}".format(np.min(bl), np.max(bl), bl.dtype))
    if write_to_oseg:
        if 127 < np.max(bl) and ((oseg.segmentation.dtype == np.int8) or (oseg.segmentation.dtype == np.uint8)):
            oseg.segmentation = oseg.segmentation.astype(np.int16)
        for lb in un:
            if lb != 0:
                new_slabel = new_label_str_format.format(vessel_label, lb)
                new_nlabel = oseg.nlabels(new_slabel)
                oseg.segmentation[bl == lb] = new_nlabel

    # ima.distance_segmentation(oseg.select_label(vessel_label))
    return bl


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
        resection(data, name, method=args.use_old_editor)
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
