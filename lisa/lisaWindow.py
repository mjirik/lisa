# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
from loguru import logger



# logger = logging.getLogger()
# from lisa.logWindow import QVBoxLayout
import sys
import os
import numpy as np
import datetime
import functools
from pathlib import Path

from io3d import datareader

# Python 3
QString = str

# import segmentation
try:
    from viewer import QVTKViewer
    viewer3D_available = True

except ImportError:
    viewer3D_available = False

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/imcut/src"))

from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                         QLabel, QPushButton, QFrame, QDialog,
    # QFont, QPixmap,
                         QFileDialog, QInputDialog)
from PyQt5 import QtGui, QtWidgets, Qt
import os.path as op

import sys
Qstring = str

from seededitorqt import QTSeedEditor
import sed3
from . import loginWindow
from . import dictEditQt
from . import segmentationQt
from . import virtual_resection
from . import organ_segmentation
from io3d.network import download_file
from imtools.select_label_qt import SelectLabelWidget
from . import authors




def find_logo():
    # logopath = os.path.join(path_to_script, "./icons/LISA256.png")
    logopath = op.join(op.dirname(__file__), "LISA256.png")
    return logopath
    # if os.path.exists(logopath):
    #     return logopath
    # # lisa runtime directory
    # logopath = os.path.expanduser("~/lisa_data/.lisa/LISA256.png")
    # if not os.path.exists(logopath):
    #     try:
    #         # wget.download(
    #         download_file(
    #             "https://raw.githubusercontent.com/mjirik/lisa/master/lisa/icons/LISA256.png",
    #             filename=logopath
    #         )
    #     except:
    #         logger.warning('logo download failed')
    #         pass
    # if os.path.exists(logopath):
    #     return logopath
    #
    # pass


# GUI
class OrganSegmentationWindow(QMainWindow):

    def __init__(self, oseg=None, qapp=None):

        self.oseg:organ_segmentation.OrganSegmentation = oseg
        self.qapp = qapp


        QMainWindow.__init__(self)
        self._initUI()
        self._initMenu()

        # if oseg is not None:
        #     if oseg.data3d is not None:
        #         self.setLabelText(self.text_dcm_dir, self.oseg.datapath)
        #         self.setLabelText(self.text_dcm_data, self.getDcmInfo())

        self.statusBar().showMessage('Ready')

    def _initMenu(self):
        menubar = self.menuBar()

        ###### FILE MENU ######
        fileMenu = menubar.addMenu('&File')
        loadSubmenu = fileMenu.addMenu('&Load')
        # load dir
        loadDirAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Directory', self)
        loadDirAction.setStatusTip('Load data from directory (DICOM, jpg, png...)')
        loadDirAction.triggered.connect(self.loadDataDir)
        loadSubmenu.addAction(loadDirAction)
        # load file
        loadFileAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&File', self)
        loadFileAction.setStatusTip('Load data from file (pkl, 3D Dicom, tiff...)')
        loadFileAction.triggered.connect(self.loadDataFile)
        loadSubmenu.addAction(loadFileAction) 
        
        saveSubmenu = fileMenu.addMenu('&Save')
        # save file
        saveFileAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&File', self)
        saveFileAction.setStatusTip('Save data with segmentation')
        saveFileAction.triggered.connect(self.saveOut)
        saveSubmenu.addAction(saveFileAction)   
        # save dicom
        saveDicomAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&DICOM', self)
        saveDicomAction.setStatusTip('Save DICOM data')
        saveDicomAction.triggered.connect(self.btnSaveOutDcm)
        saveSubmenu.addAction(saveDicomAction)
        # save dicom overlay
        saveDicomOverlayAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&DICOM overlay', self)
        saveDicomOverlayAction.setStatusTip('Save overlay DICOM data')
        saveDicomOverlayAction.triggered.connect(self.btnSaveOutDcmOverlay)
        saveSubmenu.addAction(saveDicomOverlayAction)     
        # save JSON
        saveJSONAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&JSON file', self)
        saveJSONAction.setStatusTip('Save JSON file')
        saveJSONAction.triggered.connect(self.btnSaveJSON)
        saveSubmenu.addAction(saveJSONAction)     
        # save PV tree
        savePVAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&PV tree', self)
        savePVAction.setStatusTip('Save Portal Vein 1D model')
        savePVAction.triggered.connect(self.btnSavePortalVeinTree)
        saveSubmenu.addAction(savePVAction)     
        # save HV tree
        saveHVAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Vessel Tree', self)
        saveHVAction.setStatusTip('Save actual vessel tree 1D model')
        saveHVAction.triggered.connect(self.btnSaveActualVesselTree)
        saveSubmenu.addAction(saveHVAction)     

        separator = fileMenu.addAction("")
        separator.setSeparator(True)

        autoSeedsAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Automatic liver seeds', self)
        autoSeedsAction.setStatusTip('Automatic liver seeds')
        autoSeedsAction.triggered.connect(self.btnAutomaticLiverSeeds)
        fileMenu.addAction(autoSeedsAction)

        autoSegAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Automatic segmentation', self)
        autoSegAction.setStatusTip('Automatic segmentation')
        autoSegAction.triggered.connect(self.btnAutoSeg)
        fileMenu.addAction(autoSegAction)

        cropAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Crop', self)
        cropAction.setStatusTip('')
        cropAction.triggered.connect(self.cropDcm)
        fileMenu.addAction(cropAction)

        maskAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Mask region', self)
        maskAction.setStatusTip('')
        maskAction.triggered.connect(self.maskRegion)
        fileMenu.addAction(maskAction)

        segFromFile = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Segmentation from file', self)
        segFromFile.setStatusTip('Load segmentation from pkl file, raw, ...')
        segFromFile.triggered.connect(self.btnLoadSegmentationFromFile)
        fileMenu.addAction(segFromFile)

        # TODO use cache_name "segloaddir"
        self.__add_action_to_menu(fileMenu, "&Segmentation from dir",
                                  lambda: self.oseg.import_segmentation_from_file(self.ui_get_datadir(
                                      window_title="Select directory with seeds", cache_name="loaddir"), new_label_selection_method=self.ui_select_label),
                                  tip="Import segmentation from dir",
                                  finish_msg="Ready. Segmentation loaded."
                                  )

        segFromOverlay = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Segmentation from Dicom overlay', self)
        segFromOverlay.setStatusTip('Load segmentation from Dicom file stack')
        segFromOverlay.triggered.connect(self.btnLoadSegmentationFromDicomOverlay)
        fileMenu.addAction(segFromOverlay)

        view3DAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&View 3D', self)
        view3DAction.setStatusTip('View segmentation in 3D model')
        view3DAction.triggered.connect(self.view3D)
        fileMenu.addAction(view3DAction)

        debugAction= QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Debug terminal', self)
        debugAction.setStatusTip('Run interactive terminal debug')
        debugAction.triggered.connect(self.run_debug)
        fileMenu.addAction(debugAction)

        separator = fileMenu.addAction("")
        separator.setSeparator(True)


        exitAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtWidgets.QApplication.quit)
        fileMenu.addAction(exitAction)


        ###### IMAGE MENU ######
        imageMenu = menubar.addMenu('&Image')

        randomRotateAction= QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Random Rotate', self)
        # autoSeedsAction.setShortcut('Ctrl+Q')
        randomRotateAction.setStatusTip('Random rotation')
        randomRotateAction.triggered.connect(self.btnRandomRotate)
        imageMenu.addAction(randomRotateAction)

        mirrorZAxisAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Mirror Z-axis', self)
        mirrorZAxisAction.setStatusTip('Mirror Z-axis')
        mirrorZAxisAction.triggered.connect(self.oseg.mirror_z_axis)
        imageMenu.addAction(mirrorZAxisAction)

        segmentation_by_convex_areas = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Draw convex segmentation', self)
        segmentation_by_convex_areas.setStatusTip('Create segmentation by adding convex areas')
        segmentation_by_convex_areas.triggered.connect(self.action_add_segmentation_by_convex_areas)
        imageMenu.addAction(segmentation_by_convex_areas)

        self.__add_action_to_menu(imageMenu, "&Relabel segmentation",
                                  self.action_segmentation_relabel,
                                  tip="Change label of the segmentation",
                                  finish_msg="Ready. Segmentation relabeled"
                                  )
        # segmentation_relabel_action = QtGui.QAction(QtGui.QIcon('exit.png'), '&Relabel segmentation1', self)
        # segmentation_relabel_action.setStatusTip('Change label of the segmentation')
        # segmentation_relabel_action.triggered.connect(self.action_segmentation_relabel)
        # imageMenu.addAction(segmentation_relabel_action)

        branch_label_action = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Label vessel tree', self)
        branch_label_action.setStatusTip('Label volumetric vessel tree')
        branch_label_action.triggered.connect(self.action_label_volumetric_vessel_tree)
        imageMenu.addAction(branch_label_action)

        resize_to_mm_action = QtWidgets.QAction(QtGui.QIcon('exit.png'), "Resize to mm", self)
        resize_to_mm_action.setStatusTip('Resize data3d and segemntation to mm')
        resize_to_mm_action.triggered.connect(self.action_resize_mm)
        imageMenu.addAction(resize_to_mm_action)

        self.__add_action_to_menu(imageMenu, "&Split tissue", self.action_split_on_bifurcation,
                                  tip="Split tissue based on labeled vessel tree",
                                  init_msg="Split tissue...",
                                  finish_msg="Ready. Tissue split finished."
                                  )

        self.__add_action_to_menu(imageMenu, "&Minimize slab", self.oseg.minimize_slab,
                                  tip="Remove unused or redundant labels from segmentation labeling list (slab)")

        self.__add_action_to_menu(imageMenu, "&Store seeds",
                                  lambda: self.oseg.save_seeds(self.ui_select_from_list(
                                      self.oseg.get_list_of_saved_seeds(), "Save seeds as")),
                                  tip="Save seeds for later use")

        self.__add_action_to_menu(imageMenu, "&Load seeds",
                                  lambda: self.oseg.load_seeds(self.ui_select_from_list(
                                      self.oseg.get_list_of_saved_seeds(), "Save seeds as")),
                                  tip="Save seeds for later use")

        self.__add_action_to_menu(imageMenu, "&Export seeds fo files",
                                  lambda: self.oseg.export_seeds_to_files(self.ui_select_output_filename(
                                      "mhd", window_title="Select pattern for seed files")),
                                  tip="Export seeds and stored seeds to files",
                                  finish_msg="Ready. Seeds exported."
                                  )
        self.__add_action_to_menu(imageMenu, "&Export segmentations to files",
                                  lambda: self.oseg.export_segmentations_to_files(self.ui_select_output_filename(
                                      "mhd", suffix="_seg", window_title="Select pattern for segmentation files",
                                      filter="all")),
                                  tip="Export segmentation to files. Each label is stored in single file",
                                  finish_msg="Ready. Segmentions exported."
                                  )

        self.__add_action_to_menu(imageMenu, "&Export segmentation to file",
                                  lambda: self.oseg.export_segmentation_to_file(self.ui_select_output_filename(
                                      "mhd", suffix="_seg", window_title="Select segmentation file", filter="all")),
                                  tip="Export segmentation to files. All segmentation is stored in one file.",
                                  finish_msg="Ready. Segmention exported."
                                  )

        self.__add_action_to_menu(imageMenu, "&Import seeds from file",
                                  lambda: self.oseg.import_seeds_from_file(self.ui_select_output_filename(
                                      "*", window_title="Select file with seeds")),
                                  tip="Import seeds",
                                  finish_msg="Ready. Seeds imported."
                                  )


        self.__add_action_to_menu(imageMenu, "&Body navigation structures", self.oseg.get_body_navigation_structures,
                                  tip="Put structures located by bodynavigation into segmentataion",
                                  init_msg="Get body navigation structures ...",
                                  finish_msg="Ready. Body navigation structures are in segmentation now."
                                  )

        self.__add_action_to_menu(imageMenu, "&Precise body navigation structures", self.oseg.get_body_navigation_structures_precise,
                                  tip="Put structures located by bodynavigation into segmentataion",
                                  init_msg="Get precise body navigation structures ...",
                                  finish_msg="Ready. Body navigation structures are in segmentation now."
                                  )
        self.__add_action_to_menu(imageMenu, "&Segmentation to vessel tree",
                                  lambda: self.__saveVesselTreeGui(self.ui_select_label(
                                      "Select label", "selected segmentation will be turned into 1D vessel tree"
                                      )[0]),
                                  tip="Process segmentation with vessel tree algorithm and save 1D vessel tree",
                                  finish_msg="Ready. Vessel tree saved."
                                  )
        self.__add_action_to_menu(imageMenu, "&Compare with file",
                                  self.compareSegmentationWithFile,
                                  # self.compareSegmentationWithFile,
                                  # lambda: self.__saveVesselTreeGui(self.ui_select_label(
                                  #     "Select label", "selected segmentation will be turned into 1D vessel tree"
                                  # )[0]),
                                  tip="Process segmentation with vessel tree algorithm and save 1D vessel tree",
                                  finish_msg="Ready. Vessel tree saved."
                                  )

        ###### OPTION MENU ######
        optionMenu = menubar.addMenu('&Option')
        
        configAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Configuration', self)
        configAction.setStatusTip('Config settings')
        configAction.triggered.connect(self.btnConfig)
        optionMenu.addAction(configAction)

        editSlab = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Edit labels', self)
        editSlab.setStatusTip('Edit segmentation labels')
        editSlab.triggered.connect(self.btnEditSlab)
        optionMenu.addAction(editSlab)

        logAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Log', self)
        logAction.setStatusTip('See log file')
        logAction.triggered.connect(self.btnLog)
        optionMenu.addAction(logAction)

        syncAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Sync', self)
        syncAction.setStatusTip('Synchronize files from the server')
        syncAction.triggered.connect(self.sync_lisa_data)
        optionMenu.addAction(syncAction)

        unlockAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Unlock all buttons', self)
        unlockAction.setStatusTip('Unlock all locked buttons')
        unlockAction.triggered.connect(self.unlockAllButtons)
        optionMenu.addAction(unlockAction)

        updateAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&Update', self)
        updateAction.setStatusTip('Check new update')
        updateAction.triggered.connect(self.btnUpdate)
        optionMenu.addAction(updateAction)


        ###### CONFIG MENU ######
        configMenu = menubar.addMenu('&Config')
        # combo = QtGui.QComboBox(self)
        for text in self.oseg.segmentation_alternative_params.keys():
            iAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), '&' + text, self)
            iAction.setStatusTip('Use predefined config "%s"' % (text))
            # something like lambda
            fn = functools.partial(self.onAlternativeSegmentationParams, text)
            iAction.triggered.connect(fn)

            configMenu.addAction(iAction)
        # combo.activated[str].connect(self.onAlternativeSegmentationParams)
        # grid.addWidget(combo, 4, 1)

    def __add_action_to_menu(self, submenu, ampersand_name, triggered_connect, tip='', init_msg=None, finish_msg=None):
        """

        :param submenu:
        :param ampersand_name: name with ampersand like '&Directory
        :param triggered_connect:
        :param tip:
        :return:
        """
        this_action = QtWidgets.QAction(QtGui.QIcon('exit.png'), ampersand_name, self)
        this_action.setStatusTip(tip)
        this_action.triggered.connect(lambda: self._ui_run_with(
            triggered_connect, init_msg=init_msg, finish_msg=finish_msg
        ))
        submenu.addAction(this_action)

    def _add_button(
            self,
            text,
            callback,
            uiw_label=None,
            tooltip=None,
            icon=None
                    ):
        if uiw_label is None:
            uiw_label = text
        btn = QPushButton(text, self)
        btn.clicked.connect(callback)
        if icon is not None:
            btn.setIcon(btn.style().standardIcon(icon))
        self.uiw[uiw_label] = btn
        if tooltip is not None:
            btn.setToolTip(tooltip)

        return btn

    def _activate_widget(self, widget):
        self.currentWidget.hide()
        self.grid.addWidget(widget,6,2)
        widget.show()
        self.currentWidget = widget

    def _initUI(self):
        window = QtWidgets.QWidget()
        self.window = window
        self.setCentralWidget(window)
        self.resize(1024, 768)
        from . import __version__

        self.setWindowTitle('LISA ' + __version__)
        self.statusBar().showMessage('Ready')
        self.mainLayout = QHBoxLayout(window)
        window.setLayout(self.mainLayout)
        self.ui_widgets = {}
        self.ui_buttons = {}


        #### MENU ####
        self.initUIMenu()
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        self.mainLayout.addWidget(line)


        ##### BODY #####
        bodyLayout = QVBoxLayout()
        self.bodyLayout = bodyLayout
        self.mainLayout.addLayout(bodyLayout)

        #--- title ---
        self.infoBody = QtWidgets.QWidget()
        infoBodyLayout = QVBoxLayout()
        bodyLayout.addWidget(self.infoBody)
        self.infoBody.setLayout(infoBodyLayout)
        self.ui_widgets["Main"] = self.infoBody

        font_label = QFont()
        font_label.setBold(True)
        font_info = QFont()
        font_info.setItalic(True)
        font_info.setPixelSize(12)
        lisa_title = QLabel('Liver Surgery Analyser')
        # if sys.version_info.major == 2:
        #     names = QString.fromUtf8('M. Jiřík, V. Lukeš - 2013')
        # else:
        #     names = "M. Jiřík, V. Lukeš - 2013"
        info = QLabel('Developed by:\n' +
                      'University of West Bohemia\n' +
                      'Faculty of Applied Sciences\n' +
                      "Charles University\nFaculty of Medicine in Pilsen"
                      '\n\nVersion: ' + self.oseg.version # + "\n\nAuthors:\n" + str(authors.authors)
                      )
        info.setFont(font_info)
        lisa_title.setFont(font_label)
        infoBodyLayout.addWidget(lisa_title)
        infoBodyLayout.addWidget(info)
        # self.init_uni_logo_large(infoBodyLayout)

        #--- segmentation option ---
        self.segBody = segmentationQt.SegmentationWidget(oseg=self.oseg, lisa_window=self)
        # self.segBody.oseg = self.oseg
        bodyLayout.addWidget(self.segBody)
        self.ui_widgets["Segmentation"] = self.segBody
        # self.segBody.lblSegData.setText(self.text_seg_data)
        self.segBody.btnVirtualResectionPV.clicked.connect(self.btnVirtualResectionPV)
        self.segBody.btnVirtualResectionPlanar.clicked.connect(self.btnVirtualResectionPlanar)
        self.segBody.btnVirtualResectionPV_testing.clicked.connect(self.btnVirtualResectionPV_new)


        ###
        self.segBody.btnSegManual.clicked.connect(self.btnManualSeg)
        self.segBody.btnSegSemiAuto.clicked.connect(self.btnSemiautoSeg)
        self.segBody.btnSegMask.clicked.connect(self.maskRegion)
        self.segBody.btnSegPV.clicked.connect(self.btnPortalVeinSegmentation)
        self.segBody.btnSegHV.clicked.connect(self.btnHepaticVeinsSegmentation)

        #--- edit slab ---
        self.slabBody = dictEditQt.DictEdit(dictionary=self.oseg)
        bodyLayout.addWidget(self.slabBody)
        self.ui_widgets["EditSlab"] = self.slabBody
        self.slabBody.btnSaveSlab.clicked.connect(self.segBody.reinitLabels)

        # -- load widget
        import io3d.datareaderqt


        self.read_widget = io3d.datareaderqt.DataReaderWidget(
            before_function=self._before_read_callback,
            after_function=self._after_read_callback,
            qt_app=self.qapp
        )
        self.read_widget.cache = self.oseg.cache
        bodyLayout.addWidget(self.read_widget)
        self.ui_widgets["Load"] = self.read_widget

        #--- file info (footer) ---
        bodyLayout.addStretch()

        self.oseg.gui_update = self.gui_update # nefunguje automaticky !!

        ##### OTHERS #####
        self.mainLayout.addStretch()

        self.btnSave.setDisabled(True)
        self.btnSegmentation.setDisabled(True)
        self.btnCompare.setDisabled(True)
        self.changeWidget('Main')
        self.unlockAllButtons()
        self.show()

    def gui_update(self):
        self.segBody.reinitLabels()
        self.slabBody.discardChanges()

    def initLogo(self, layout):
        font_label = QFont()
        font_label.setBold(True)
        font_info = QFont()
        font_info.setItalic(True)
        font_info.setPixelSize(12)

        lisa_logo = QLabel()
        logopath = find_logo()
        logo = QPixmap(logopath)
        logo = logo.scaled(130, 130)
        lisa_logo.setPixmap(logo)  # scaledToWidth(128))
        lisa_logo.mousePressEvent = lambda event: self.changeWidget('Main')
        layout.addWidget(lisa_logo)
        return layout

    def init_uni_logo_large(self, layout):
        logo_fav = QLabel()
        logo_fav_path = op.join(op.dirname(__file__), "fav_en.png")
        logo_fav_im = QPixmap(logo_fav_path)
        logo_fav_im = logo_fav_im.scaledToWidth(300)
        logo_fav.setPixmap(logo_fav_im)  # scaledToWidth(128))
        layout.addWidget(logo_fav)

        logo_lfp = QLabel()
        logo_lfp_path = op.join(op.dirname(__file__), "logo_lf_uk_2.png")
        logo_lfp_im = QPixmap(logo_lfp_path)
        logo_lfp_im = logo_lfp_im.scaledToWidth(300)
        logo_lfp.setPixmap(logo_lfp_im)  # scaledToWidth(128))
        layout.addWidget(logo_lfp)

    def init_uni_logo_small(self, layout):
        logo_fav = QLabel()
        logo_fav_path = op.join(op.dirname(__file__), "logo_fav.png")
        logo_fav_im = QPixmap(logo_fav_path)
        logo_fav_im = logo_fav_im.scaledToWidth(130)
        logo_fav.setPixmap(logo_fav_im)  # scaledToWidth(128))
        layout.addWidget(logo_fav)

        logo_lfp = QLabel()
        logo_lfp_path = op.join(op.dirname(__file__), "logo_lfp.png")
        logo_lfp_im = QPixmap(logo_lfp_path)
        logo_lfp_im = logo_lfp_im.scaledToWidth(130)
        logo_lfp.setPixmap(logo_lfp_im)  # scaledToWidth(128))
        layout.addWidget(logo_lfp)

    def initUIMenu(self):
        ###### MAIN MENU ######
        self.mainLayoutRight = QVBoxLayout()
        mainLayoutRight = self.mainLayoutRight
        self.mainLayout.addLayout(mainLayoutRight)

        # ----- logo -----
        self.initLogo(mainLayoutRight)

        # --load--
        self.btnLoadWidget = QPushButton("Load", self)
        self.btnLoadWidget.clicked.connect(lambda: self.changeWidget('Load'))
        mainLayoutRight.addWidget(self.btnLoadWidget)

        # --save--
        self.btnSave = QPushButton("Save/Export")
        mainLayoutRight.addWidget(self.btnSave)

        saveFileAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), "File", self)
        saveFileAction.triggered.connect(self.saveOut)

        saveDicomAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), "Export Dicom", self)
        saveDicomAction.triggered.connect(self.btnSaveOutDcm)

        saveDicomOverlayAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), "Export Dicom overlay", self)
        saveDicomOverlayAction.triggered.connect(self.btnSaveOutDcmOverlay)

        saveJSONAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), "JSON file", self)
        saveJSONAction.triggered.connect(self.btnSaveJSON)

        saveImageStackAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), "Image stack", self)
        saveImageStackAction.triggered.connect(self.saveOutImageStack)

        savePVTreeAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), "PV Tree", self)
        savePVTreeAction.triggered.connect(self.btnSavePortalVeinTree)

        saveHVTreeAction = QtWidgets.QAction(QtGui.QIcon('exit.png'), "Vessel Tree", self)
        saveHVTreeAction.setStatusTip('Save actual vessel tree 1D model')
        saveHVTreeAction.triggered.connect(self.btnSaveActualVesselTree)


        menu = QtWidgets.QMenu(self.btnSave)
        menu.addAction(saveFileAction)
        menu.addAction(saveImageStackAction)
        menu.addAction(saveDicomAction)
        menu.addAction(saveDicomOverlayAction)
        menu.addAction(saveJSONAction)
        menu.addAction(savePVTreeAction)
        menu.addAction(saveHVTreeAction)
        self.btnSave.setMenu(menu)

        # --segmentation--
        self.btnSegmentation = QPushButton("Segmentation", self)
        self.btnSegmentation.setCheckable(True)
        self.btnSegmentation.clicked.connect(lambda: self.changeWidget('Segmentation'))
        self.btnSegmentation.setStyleSheet('QPushButton:checked,QPushButton:pressed {border: 1px solid #25101C; background-color: #BA5190; color: #FFFFFF}')
        mainLayoutRight.addWidget(self.btnSegmentation)

        # --others--
        self.btnCompare = QPushButton("Compare", self)
        self.btnCompare.clicked.connect(self.compareSegmentationWithFile)
        mainLayoutRight.addWidget(self.btnCompare)

        # --others--
        keyword = "3D Visualization"
        tmp = QPushButton(keyword, self)
        # tmp.clicked.connect(self.action3DVisualizationWidget)
        tmp.clicked.connect(lambda: self.changeWidget(keyword))
        mainLayoutRight.addWidget(tmp)
        self.ui_buttons[keyword] = tmp

        import imtools.show_segmentation_qt

        self.ui_widgets[keyword] = imtools.show_segmentation_qt.ShowSegmentationWidget(None)
        # import imtools.show_segmentation_qt as itss
        # itss.ShowSegmentationWidget()

        if self.oseg.debug_mode:
            btn_debug = QPushButton("Debug", self)
            btn_debug.clicked.connect(self.run_debug)
            mainLayoutRight.addWidget(btn_debug)

        btnQuit = QPushButton("Quit", self)
        btnQuit.clicked.connect(self.quit)
        mainLayoutRight.addWidget(btnQuit)

        mainLayoutRight.addStretch()
        self.init_uni_logo_small(mainLayoutRight)

    def unlockAllButtons(self):
        self.btnSave.setDisabled(False)
        self.btnSegmentation.setDisabled(False)
        self.btnCompare.setDisabled(False)

    def changeWidget(self, option):
        # widgets = [
        #     self.infoBody,
        #     self.segBody,
        #     self.slabBody,
        #     self.read_widget
        # ]
        for key, widget in self.ui_widgets.items():

            widget.hide()

        if option == 'EditSlab':
            # self.infoBody.hide()
            # self.segBody.hide()
            self.slabBody.show()
        elif option == 'Main':
            self.infoBody.show()
            # self.segBody.hide()
            # self.slabBody.hide()
        elif option == 'Segmentation':
            if self.btnSegmentation.isChecked() == True:
                # self.infoBody.hide()
                self.segBody.reinitLabels()
                self.segBody.show()
                # self.slabBody.hide()
            else:
                self.infoBody.show()
            return
        elif option == "3D Visualization":
            # remove old
            import imtools.show_segmentation_qt


            widget = self.ui_widgets[option]
            self.bodyLayout.removeWidget(widget)
            widget.deleteLater()
            widget = None

            # add new
            vtk_file = self.oseg.get_standard_ouptut_filename(suffix="_{}", filetype="vtk")
            widget = imtools.show_segmentation_qt.ShowSegmentationWidget(None, show_load_interface=True, vtk_file=vtk_file )
            self.ui_widgets[option] = widget
            self.bodyLayout.addWidget(widget)
            logger.debug("adding actual data to visualization widget")
            logger.debug(f"voxelsize_mm={self.oseg.voxelsize_mm}")
            logger.debug(f"segmentation.shape={self.oseg.segmentation.shape}")
            widget.add_data(self.oseg.segmentation, self.oseg.voxelsize_mm, self.oseg.slab)
            widget.show()

        elif option == 'Load':
            self.read_widget.show()
            # self.infoBody.hide()
            # self.segBody.show()
            # self.slabBody.hide()

        elif option in self.ui_widgets.keys():
            self.ui_widgets[option].show()

        self.btnSegmentation.setChecked(False)

    def quit(self, event):
        return self.close()

    def run_debug(self, event):
        """
        Stop processing and start debugger in console. Function started by
        pressing "Debug" button wich is availeble after starting Lisa with
        -d parameter.
        """
        logger.debug('== Starting debug mode, leave it with command "c" =')
        from PyQt5.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
        try:
            import ipdb; ipdb.set_trace()  # noqa BREAKPOINT
        except:
            import pdb; pdb.set_trace()  # noqa BREAKPOINT




    def changeVoxelSize(self, val):
        self.scaling_mode = str(val)

    def setLabelText(self, obj, text):
        dlab = str(obj.text())
        obj.setText(dlab[:dlab.find(':')] + ': %s' % text)

    def getDcmInfo(self):
        vx_size = self.oseg.voxelsize_mm
        vsize = tuple([float(ii) for ii in vx_size])
        ret = ' %dx%dx%d,  %fx%fx%f mm' % (self.oseg.data3d.shape + vsize)

        return ret

    # def setVoxelVolume(self, vxs):
    #     self.voxel_volume = np.prod(vxs)

    def __get_datafile(self, directory=None, window_title=None, app=None, cache_name="loadfiledir"):
        """
        Draw a dialog for directory selection.
        """
        if directory is None:
            if 'loaddir' in self.oseg.cache.data.keys():
                directory = self.oseg.cache.get(cache_name)
            else:
                directory = self.oseg.input_datapath_start

        if app is None:
            app=self.qapp
        if window_title is None:
            window_title = "Select Data File"
        from PyQt5.QtWidgets import QFileDialog
        # if app:
        dcmdir = QFileDialog.getOpenFileName(
            caption=window_title,
            directory=directory
            # ptions=QFileDialog.ShowDirsOnly,
        )[0]
        # else:
        #     app = QApplication(sys.argv)
        #     dcmdir = QFileDialog.getOpenFileName(
        #         caption=window_title,
        #         # caption='Select DICOM Folder',
        #         # ptions=QFileDialog.ShowDirsOnly,
        #         directory=directory
        #     )[0]
            # pp.exec_()
            # app.exit(0)
        if len(dcmdir) > 0:

            dcmdir = "%s" % (dcmdir)
            dcmdir = dcmdir.encode("utf8")
        else:
            dcmdir = None
        return dcmdir

    def loadDataFile(self):
        self.statusBar().showMessage('Reading data file...')
        QApplication.processEvents()

        oseg = self.oseg
        if 'loadfiledir' in self.oseg.cache.data.keys():
            directory = self.oseg.cache.get('loadfiledir')
        else:
            directory = self.oseg.input_datapath_start
        #
        oseg.datapath = self.__get_datafile(
            app=True,
            directory=directory
        )

        if oseg.datapath is None:
            self.statusBar().showMessage('No data path specified!')
            return
        head, teil = os.path.split(oseg.datapath)
        self.oseg.cache.update('loadfiledir', head)

        self.importDataWithGui()

    def _before_read_callback(self, readerWidget):
        self.statusBar().showMessage('Reading data file...')
        QApplication.processEvents()

    def _after_read_callback(self, readerWidget):
        logger.debug("after read callback started")
        self.statusBar().showMessage('Ready')
        QApplication.processEvents()
        self.oseg.datapath = readerWidget.datapath
        self.importDataWithGui(datap=readerWidget.datap)
        logger.debug("after read callback finished")
        # print readerWidget.loaddir
        # print readerWidget.loadfiledir
        # import ipdb; ipdb.set_trace()

    def loadDataDir(self):
        self.statusBar().showMessage('Reading DICOM directory...')
        QApplication.processEvents()

        oseg = self.oseg

        oseg.datapath = self.ui_get_datadir(
        )

        if oseg.datapath is None:
            self.statusBar().showMessage('No DICOM directory specified!')
            return
        # head, teil = os.path.split(oseg.datapath)

        self.importDataWithGui()

    def importDataWithGui(self, datap=None):
        logger.debug("import data with gui started")
        oseg = self.oseg

        if datap is None:
            reader = datareader.DataReader()

            # from PyQt4.QtCore import pyqtRemoveInputHook
            # pyqtRemoveInputHook()
            # seg.data3d, metadata =
            logger.debug("reading data")
            datap = reader.Get3DData(oseg.datapath, dataplus_format=True, gui=True, qt_app=self.qapp)
            logger.debug("datap readed")

        logger.debug("importing datap")
        oseg.import_dataplus(datap)
        self.statusBar().showMessage('Ready. Data loaded from ' + str(oseg.datapath))
        logger.debug("import data with gui finished")

        #### SET BUTTONS/MENU ####
        self.unlockAllButtons()


    def sync_lisa_data(self):
        """
        set sftp_username and sftp_password in ~/lisa_data/organ_segmentation.config
        """
        self.statusBar().showMessage('Sync in progress...')
        login = loginWindow.Login(checkLoginFcn=self.__loginCheckFcn)
        login.exec_()
        # print repr(self.oseg.sftp_username)
        # print repr(self.oseg.sftp_password)
        try:
            self.oseg.sync_lisa_data(self.oseg.sftp_username, self.oseg.sftp_password, callback=self._print_sync_progress)
        except:
            import traceback


            traceback.print_exc()
            logger.error(traceback.format_exc())

            QtWidgets.QMessageBox.warning(
                self, 'Error', 'Sync error')

        self.oseg.sftp_username = ''
        self.oseg.sftp_password = ''
        self.statusBar().showMessage('Sync finished')

    def __loginCheckFcn(self, textname, textpass):
        self.oseg.sftp_username = textname
        self.oseg.sftp_password = textpass
        return True

    def _print_sync_progress(self, transferred, toBeTransferred):
        self.statusBar().showMessage('Sync of current file {0} % '.format((100.0 * transferred) / toBeTransferred ))

        # print "Transferred: {0}\tOut of: {1}".format(transferred, toBeTransferred)

    def cropDcm(self):
        oseg = self.oseg

        if oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Cropping DICOM data...')
        QApplication.processEvents()

        pyed = QTSeedEditor(oseg.data3d, mode='crop',
                            voxelSize=oseg.voxelsize_mm)
        # @TODO
        mx = self.oseg.viewermax
        mn = self.oseg.viewermin
        width = mx - mn
        # enter = (float(mx)-float(mn))
        center = np.average([mx, mn])
        logger.debug("window params max %f min %f width, %f center %f" %
                     (mx, mn, width, center))
        pyed.changeC(center)
        pyed.changeW(width)
        pyed.exec_()

        crinfo = pyed.getROI()
        if crinfo is not None:
            tmpcrinfo = []
            for ii in crinfo:
                tmpcrinfo.append([ii.start, ii.stop])

            # seg.data3d = qmisc.crop(oseg.data3d, oseg.crinfo)
            oseg.crop(tmpcrinfo)

        # self.setLabelText(self.text_dcm_data, self.getDcmInfo())
        self.statusBar().showMessage('Ready. Image cropped to ' + str(self.getDcmInfo()) )

    def maskRegion(self):
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Mask region...')
        QApplication.processEvents()

        pyed = QTSeedEditor(
                self.oseg.data3d, mode='mask',
                voxelSize=self.oseg.voxelsize_mm,
                contours=((self.oseg.segmentation == 0).astype(np.int8)*2)
                )

        pyed.contours_old = pyed.contours.copy()
        # initial mask set
        # pyed.masked = np.ones(self.oseg.data3d.shape, np.int8)
        # pyed.masked = (self.oseg.segmentation == 0).astype(np.int8)

        mx = self.oseg.viewermax
        mn = self.oseg.viewermin
        width = mx - mn
        # enter = (float(mx)-float(mn))
        center = np.average([mx, mn])
        logger.debug("window params max %f min %f width, %f center %f" %
                     (mx, mn, width, center))
        pyed.changeC(center)
        pyed.changeW(width)
        pyed.exec_()

        self.statusBar().showMessage('Ready')

    def action_add_segmentation_by_convex_areas(self):


        # @todo add button for this functionality
        nlabel, slabel = self.ui_select_label("Select label for segmentation")

        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Draw segmentation by vonvex areas ...')

        QApplication.processEvents()

        target_segmentation = ((self.oseg.select_label(nlabel)).astype(np.int8)*2)
        # ed = sed3.sed3(target_segmentation)
        # ed.show()

        pyed = QTSeedEditor(
            self.oseg.data3d, mode='mask',
            voxelSize=self.oseg.voxelsize_mm,
            contours=target_segmentation
        )

        pyed.contours_old = pyed.contours.copy()
        # initial mask set
        # pyed.masked = np.ones(self.oseg.data3d.shape, np.int8)
        # pyed.masked = (self.oseg.segmentation == 0).astype(np.int8)

        mx = self.oseg.viewermax
        mn = self.oseg.viewermin
        width = mx - mn
        # enter = (float(mx)-float(mn))
        center = np.average([mx, mn])
        logger.debug("window params max %f min %f width, %f center %f" %
                     (mx, mn, width, center))
        pyed.changeC(center)
        pyed.changeW(width)
        pyed.exec_()

        contours = pyed.contours.copy()
        logger.debug("output contours unique" +
                     str(np.unique(contours)))
        self.segmentation_temp = contours
        # nlabel = self.oseg.nlabels(nlabel)
        self.oseg.segmentation_replacement(
            contours,
            label=nlabel,
            label_new=2,

        )
        # self.oseg.segmentation[contours == 2] = nlabel

        self.statusBar().showMessage('Ready')

    def _ui_run_with(self, fcn, init_msg=None, finish_msg=None):
        if init_msg is not None:
            self.statusBar().showMessage(init_msg)
        fcn()
        if finish_msg is not None:
            self.statusBar().showMessage(finish_msg)

    def action_segmentation_relabel(self):
        self.statusBar().showMessage('Relabelling segmentation')
        no, from_label = self.ui_select_label(
            "Select source", text_inside="Select source label(s) for relabel", multiple_choice=True)
        # val = self.lisa_window.ui_select_label("Rename from " + strlabel + "to to fallowing label")
        no, to_label = self.ui_select_label(
            "Select destination", text_inside=f"Select destination for renaming from {from_label}.\nNew label can be created by typing.")
        logger.debug(f"Relabeling from {from_label} to {to_label}")
        self.oseg.segmentation_relabel(from_label=from_label, to_label=to_label)
        self.statusBar().showMessage('Ready. Relabeled from ' + str(from_label) + " to " + str(to_label))

    def mask_segmentation(self):
        # @todo add button for this functionality
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.statusBar().showMessage('Mask region...')
        QApplication.processEvents()

        pyed = QTSeedEditor(
            self.oseg.segmentation, mode='mask',
            voxelSize=self.oseg.voxelsize_mm,
            contours=((self.oseg.segmentation == 0).astype(np.int8)*2)
        )

        pyed.contours_old = pyed.contours.copy()
        # initial mask set
        # pyed.masked = np.ones(self.oseg.data3d.shape, np.int8)
        # pyed.masked = (self.oseg.segmentation == 0).astype(np.int8)

        mx = self.oseg.viewermax
        mn = self.oseg.viewermin
        width = mx - mn
        # enter = (float(mx)-float(mn))
        center = np.average([mx, mn])
        logger.debug("window params max %f min %f width, %f center %f" %
                     (mx, mn, width, center))
        pyed.changeC(center)
        pyed.changeW(width)
        pyed.exec_()

        self.statusBar().showMessage('Ready')

    def btnLoadSegmentationFromFile(self):
        """
        Function make GUI for reading segmentaion file and calls
        organ_segmentation function to do the work.
        """
        self.statusBar().showMessage('Reading segmentation from file ...')
        QApplication.processEvents()
        logger.debug("import segmentation from file")
        logger.debug(str(self.oseg.crinfo))
        logger.debug(str(self.oseg.data3d.shape))
        logger.debug(str(self.oseg.segmentation.shape))
        path = self.oseg.cache.get_or_none('loadfiledir')
        if path is None:
            path = self.oseg.input_datapath_start
        seg_path = self.__get_datafile(
            # app=True,
            directory=path
        )
        if seg_path is None:
            self.statusBar().showMessage('No data path specified!')
            return
        self.oseg.import_segmentation_from_file(seg_path, new_label_selection_method=self.ui_select_label)
        self.segBody.enableSegType()
        self.statusBar().showMessage('Ready. Segmentation loaded from ' + str(seg_path))

    def action_split_on_bifurcation(self):

        self.statusBar().showMessage('Split vessel tree...')
        ed = QTSeedEditor(img=self.oseg.data3d, contours=self.oseg.segmentation,
                          voxelSize=self.oseg.voxelsize_mm, init_brush_index=0)
        ed.exec_()
        seeds = ed.getSeeds()
        # ed.see
        # ed = sed3.sed3(self.oseg.segmentation)
        # ed.show()

        seeds = ed.seeds
        labeled_branches = self.oseg.segmentation

        # trunk_label = labeled_branches[seeds == 1][0]
        # branch_label1 = labeled_branches[seeds == 2][0]
        # branch_label2 = labeled_branches[seeds == 3][0]
        organ_label, textl = self.ui_select_label("Organ to split")

        # split_label, textl = self.ui_select_label("Label for new split")

        un = np.unique(seeds)
        import imma.labeled


        unlab = imma.labeled.unique_labels_by_seeds(self.oseg.segmentation, seeds)

        self.oseg.split_tissue_with_labeled_volumetric_vessel_tree(organ_label, trunk_label=unlab[1][0], branch_labels=unlab[2])  # trunk_label, branch_label1, branch_label2)
        self.statusBar().showMessage('Ready.')

    def __evaluation_to_text(self, evaluation):
        overall_score = evaluation['sliver_overall_pts']
        # \
        #     volumetry_evaluation.sliver_overall_score_for_one_couple(
        #         score
        #     )

        logger.info('overall score: ' + str(overall_score))
        return "Sliver score: " + str(overall_score)

    def ui_select_label(self, headline, text_inside="select from existing labels or write a new one",
                        return_i=True, return_str=True, multiple_choice=False, datap=None):
        """ Get label with GUI.

        :return: numeric_label, string_label
        :param datap: if no datap is selected then the actual data are used and extracted by self.oseg.export()
        """

        # import copy
        # texts = copy.copy(self.oseg.slab.keys())

        if datap is None:
            datap = self.oseg.export()

        data3d = datap["data3d"]
        segmentation = datap["segmentation"]
        voxelsize_mm = datap["voxelsize_mm"]
        logger.debug(datap.keys())

        slab = datap["slab"] if "slab" in datap else {}
        logger.debug(f"slab={slab}")
        import imma.segmentation_labels
        imma.segmentation_labels.add_missing_labels(segmentation, slab=slab)
        logger.debug(f"slab={slab}")

        new_label_selected_from_viewer = None

        def _ui_get_labels_with_gui():
            nonlocal new_label_selected_from_viewer
            # oseg = self.oseg
            sgm = segmentation.astype(np.uint8)
            pyed = QTSeedEditor(data3d,
                                appmenu_text="Select one or more labels",
                                # seeds=sgm,
                                # mode='draw',
                                contours=sgm,
                                init_brush_index=0,
                                voxelSize=voxelsize_mm, volume_unit='ml')
            # self.qapp.exec_()
            logger.debug("before ui get seeds exec")
            pyed.show()
            pyed.exec_()
            logger.debug("after ui get seeds exec")
            seeds = pyed.getSeeds()

            unseeds = np.unique(segmentation[np.nonzero(seeds)])
            # unseeds = list(np.unique(seeds))
            # unseeds.pop[0]
            logger.debug("unique seeds {}".format(unseeds))
            new_label_selected_from_viewer = unseeds
            return unseeds

        if multiple_choice:
            add_button2 = True
            button2_label = "Select from image"
            button2_fcn = _ui_get_labels_with_gui
        else:
            add_button2 = False
            button2_label = None
            button2_fcn = None

        strlab = self.ui_select_from_list(
            slab,
            # list(slab.keys()),
            headline,
            text_inside=text_inside,
            multiple_choice=multiple_choice,
            add_button2=add_button2,
            button2_label=button2_label,
            button2_fcn=button2_fcn,
        )
        if new_label_selected_from_viewer:
            logger.debug(f"new_label_selected_from_viewer: {new_label_selected_from_viewer}")
            strlab = new_label_selected_from_viewer
        numlab = self.oseg.nlabels(strlab)
        if return_i and return_str:
            return numlab, strlab
        elif return_str:
            return strlab
        else:
            return numlab

    def ui_select_from_list(self, some_list, headline, text_inside="", multiple_choice=False, add_button2=False, button2_label=None, button2_fcn=None):
        """ Select string from list with GUI.

        :return: selected string
        """

        if multiple_choice:
            if type(some_list) == list:
                slab = {str(aa):aa for aa in list}
            else:
                slab = some_list
            slab_dialog = QtWidgets.QDialog(self)
            layout = QtWidgets.QGridLayout()
            slab_dialog.setLayout(layout)
            slab_wg = SelectLabelWidget(show_ok_button=False)
            slab_wg.init_slab(slab=slab, show_ok_button=False)
            slab_wg.update_slab_ui()

            layout.addWidget(slab_wg)

            if add_button2:
                def close_and_do(**kwargs):
                    logger.debug("ui sel close")
                    slab_dialog.close()
                    logger.debug("ui sel closed")
                    button2_ret_val = button2_fcn(**kwargs)
                    logger.debug(f"clicked on label {button2_ret_val}")
                    # slab_wg.overwrite_return_labeles(button2_ret_val)
                ok2 = QPushButton(button2_label)
                ok2.clicked.connect(close_and_do)
                layout.addWidget(ok2)

            ok = QPushButton("ok")
            ok.clicked.connect(slab_dialog.close)
            layout.addWidget(ok)
            self._ok_button_to_click_for_testing = ok
            slab_dialog.exec_()
            logger.debug("ui sel after exec")

            labels = slab_wg.get_selected_labels()
            labels = self.oseg.nlabels(labels, return_mode="str")
            strlab = labels
        else:
            if type(some_list) == dict:
                some_list = list(some_list.keys())

            logger.debug(f"some_list [{type(some_list)}={some_list}")
            strlab, ok = \
                QInputDialog.getItem(self,
                                     # self.qapp,
                                     headline,
                                     text_inside,
                                     some_list,
                                     editable=True)

            if not ok:
                raise ValueError("Selection canceled")
            strlab = str(strlab)

        # print("strlab", strlab)
        return strlab

    def ui_get_double(self, headline, value=0.0, **kwargs):
        """

        :param headline:
        :return: double_value
        """
        val, ok = \
            QInputDialog.getDouble(
                self,
                # self.qapp,
                headline,
                "insert number",
                value=value,
                **kwargs
            )
        return val, ok

    def compareSegmentationWithFile(self):
        """
        Function make GUI for reading segmentaion file to compare it with
        actual segmentation using Sliver methodics. It calls
        organ_segmentation function to do the work.
        """
        self.statusBar().showMessage('Reading segmentation from file ...')
        QApplication.processEvents()
        logger.debug("import segmentation from file to compare by sliver")
        logger.debug(str(self.oseg.crinfo))
        logger.debug(str(self.oseg.data3d.shape))
        logger.debug(str(self.oseg.segmentation.shape))
        cache_key = "loadcomparedir"
        directory = self.oseg.cache.get_or_none(cache_key)
        if directory is None:
            directory = self.oseg.input_datapath_start

        label1,_ = self.ui_select_label("Select label comparison", multiple_choice=True)
        seg_path = self.ui_get_path_file_or_dir(
            directory=directory
        )
        logger.debug(f"seg_path[{type(seg_path)}]={seg_path}")
        logger.debug(f"exist={Path(seg_path).exists()}")
        # seg_path = self.__get_datafile(
        #     directory=directory
        # )
        if seg_path is None:
            self.statusBar().showMessage('No data path specified!')
            return
        reader = datareader.DataReader()
        segmentation_datap = reader.Get3DData(seg_path, dataplus_format=True)
        if "segmentation" in segmentation_datap:
            pass
        else:
            segmentation_datap["segmentation"] = segmentation_datap["data3d"]
        label2,_ = self.ui_select_label("Select label from loaded data", datap=segmentation_datap, multiple_choice=True)
        evaluation, segdiff = \
            self.oseg.sliver_compare_with_other_volume(segmentation_datap, label1=label1, label2=label2)
        print('Evaluation: ', evaluation)
        # print 'Score: ', score

        text = self.__evaluation_to_text(evaluation)

        segdiff[segdiff == -1] = 2
        logger.debug('segdif unique ' + str(np.unique(segdiff)))

        QApplication.processEvents()
        import seededitorqt.seed_editor_qt
        seeds_colortable = seededitorqt.seed_editor_qt.SEEDS_COLORTABLE.copy()
        # seeds_colortable[0] = [255, 0, 64, 128]
        # seeds_colortable[1] = [255, 64, 0, 128]
        # seeds_colortable[2] = [255, 64, 0, 128]
        seeds_colortable[0] = [0, 64, 255, 64]
        seeds_colortable[1] = [64, 0, 255, 64]
        seeds_colortable[2] = [0, 0, 255, 64]
        seeds_colortable[3] = [0, 255, 255, 64]
        seeds_colortable[4] = [255, 0, 255, 64]
        seeds_colortable[5] = [255, 255, 0, 64]
        se = seededitorqt.QTSeedEditor(
            self.oseg.data3d,
            seeds=segdiff.astype(np.uint8),
            contours=self.oseg.segmentation.astype(np.uint8),
            seeds_colortable=seeds_colortable,
            appmenu_text=f"Error={evaluation['voe']:.2f}[%]",
        )
        se.exec_()
        # try:
        #
        #     ed = sed3.sed3qt(
        #         self.oseg.data3d,
        #         seeds=segdiff,
        #         # contour=(self.oseg.segmentation == self.oseg.slab['liver'])
        #         contour=self.oseg.segmentation
        #     )
        #     ed.exec_()
        # except:
        #     ed = sed3.sed3(
        #         self.oseg.data3d,
        #         seeds=segdiff,
        #         contour=(self.oseg.segmentation == self.oseg.slab['liver'])
        #     )
        #     ed.show()

        head, teil = os.path.split(seg_path)
        self.oseg.cache.update(cache_key, head)
        self.setLabelText(self.segBody.lblSegData, text=text)
        self.statusBar().showMessage('Ready')

    def action_resize_mm(self):
        self.statusBar().showMessage('Performing resize ...')
        val, ok = self.ui_get_double(headline="Set new voxelsize [mm]", value=5.0)
        self.oseg.resize_to_mm(val)

        self.statusBar().showMessage('Ready')

    def btnSemiautoSeg(self):
        self.statusBar().showMessage('Performing liver segmentation ...')
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.oseg.interactivity(
            min_val=self.oseg.viewermin,
            max_val=self.oseg.viewermax),
            # TODO
            # self.sublayout_segmentace
        self.checkSegData('auto. seg., ')
        self.statusBar().showMessage('Ready')

    def btnAutoSeg(self):
        self.statusBar().showMessage('Performing automatic segmentation...')
        QApplication.processEvents()
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.oseg.run_sss()
        self.statusBar().showMessage('Automatic segmentation finished')



    def btnManualSeg(self):
        oseg = self.oseg
        # rint 'ms d3d ', oseg.data3d.shape
        # rint 'ms seg ', oseg.segmentation.shape
        # rint 'crinfo ', oseg.crinfo
        if oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return
        sgm = oseg.segmentation.astype(np.uint8)

        pyed = QTSeedEditor(oseg.data3d,
                            seeds=sgm,
                            mode='draw',
                            voxelSize=oseg.voxelsize_mm, volume_unit='ml')
        pyed.exec_()

        oseg.segmentation = pyed.getSeeds()
        self.oseg.processing_time = \
            datetime.datetime.now() - self.oseg.time_start
        self.checkSegData('manual seg., ')

    def checkSegData(self, msg):
        oseg = self.oseg
        if oseg.segmentation is None:
            self.statusBar().showMessage('No segmentation!')
            return

        nzs = oseg.segmentation.nonzero()
        nn = nzs[0].shape[0]
        if nn > 0:
            voxelvolume_mm3 = np.prod(oseg.voxelsize_mm)
            tim = self.oseg.processing_time

            if self.oseg.volume_unit == 'ml':
                # import datetime
                # timstr = str(datetime.timedelta(seconds=round(tim)))
                timstr = str(tim)
                # logger.debug('tim = ' + str(tim))
                aux = 'volume = %.2f [ml] , time = %s' %\
                      (nn * voxelvolume_mm3 / 1000, timstr)
            else:
                aux = 'volume = %.6e mm3' % (nn * voxelvolume_mm3, )
            self.setLabelText(self.segBody.lblSegData, msg + aux)
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No segmentation!')

    def saveOutImageStack(self, event=None):
        """
        Open dialog with filename suggestion to save output as image stack
        :param event:
        :return:
        """
        self.saveOut(image_stack=True)

    def ui_select_output_filename(self, filetype="pklz", suffix="_seeds", window_title="Save file", filter=None,
                                  # cache_name="saveoutputfile"
                                  ):
        """

        :param filetype: extension without dot
        :param suffix: what is the last part of filename before extension
        :param window_title:
        :param filter: string in format 'Dicom (*.dcm);;' or one of predefined: pklz, h5, dicom, all
        :return:
        """
        if filter is None:
            if filetype is None:
                filter = "Former Lisa Format (*.pklz);;New Lisa format HDF5 (*.h5 *.hdf5);; Dicom (*.dcm);; All files(*.*)"
            elif filetype in ("pklz"):
                filter = "Former Lisa Format (*.pklz);; All files (*.*)"
            elif filetype in ("h5", "hdf5"):
                filter = "New Lisa format HDF5 (*.h5 *.hdf5);; All files (*.*)"
            elif filetype in ("dcm", "dicom"):
                filter = "Dicom (*.dcm);; All files (*.*)"
            elif filetype in ("*", "all"):
                filter = "All files (*.*);;Dicom (*.dcm);;Former Lisa Format (*.pklz);;New Lisa format HDF5 (*.h5 *.hdf5)"
            else:
                filter = "(*.{});; All files (*.*)".format(filetype)
        ofilename = self.oseg.get_standard_ouptut_filename(filetype=filetype, suffix=suffix)
        filename = str(QFileDialog.getSaveFileName(
            self,
            window_title,
            ofilename,
            filter=filter)[0])

        logger.info(f'Selected file: {filename}')
        return filename

    def ui_get_path_file_or_dir(self, directory=None, window_title=None):
        d = QDialog()
        file_dialog = True

        def click_file():
            nonlocal file_dialog
            file_dialog = True
            d.close()

        def clicked_dir():
            nonlocal file_dialog
            file_dialog = False
            d.close()

        b1 = QPushButton("File", d)
        b1.clicked.connect(click_file)
        b1.move(10, 10)
        b2 = QPushButton("Directory", d)
        b2.clicked.connect(clicked_dir)
        b2.move(10, 50)
        d.setWindowTitle("File or Directory")
        # d.setWindowModality(Qt.ApplicationModal)
        d.exec_()

        if file_dialog:
            # datafile is for some strange reason bytes not str
            pth = self.__get_datafile(directory=directory, window_title=window_title)
            # from PyQt5.QtCore import pyqtRemoveInputHook
            # pyqtRemoveInputHook()
            # import pdb;
            # pdb.set_trace()
            return pth.decode()
        else:
            return self.ui_get_datadir(directory=directory, window_title=window_title)


    def ui_get_datadir(self, directory=None, window_title="Load from directory",
                       cache_name="loaddir"):
        """
        Create window to select directory
        :param directory:
        :param window_title:
        :param cache_name: name to store path for next use.
        :return:
        """

        if directory is None:
            if 'loaddir' in self.oseg.cache.data.keys():
                directory = self.oseg.cache.get(cache_name)
            else:
                directory = self.oseg.input_datapath_start

        from PyQt5.QtWidgets import QFileDialog
        if self.qapp is not None:
            dcmdir = QFileDialog.getExistingDirectory(
                self,
                caption=window_title,
                options=QFileDialog.ShowDirsOnly,
                directory=directory,
            )
        else:
            app = QApplication(sys.argv)
            dcmdir = QFileDialog.getExistingDirectory(
                self,
                window_title,
                caption=window_title,
                options=QFileDialog.ShowDirsOnly,
                directory=directory,
            )
            # pp.exec_()
            app.exit(0)
        dcmdir = get_str(dcmdir)

        if len(dcmdir) == 0:
            dcmdir = None
        else:
            self.oseg.cache.update('loaddir', dcmdir)
        return dcmdir

    def ui_select_input_dir(self, filetype="pklz", suffix="_seeds", window_title="Load dir", filter=None):
        ofilename = self.oseg.get_standard_ouptut_filename(filetype=filetype, suffix=suffix)
        ofilename = self.oseg.get_stan(filetype=filetype, suffix=suffix)

        if 'loaddir' in self.oseg.cache.data.keys():
            directory = self.oseg.cache.get('loaddir')
        else:
            directory = self.oseg.input_datapath_start

        filename = str(QFileDialog.getO(
            self,
            window_title,
            ofilename,
            filter=filter))

        logger.info('Selected file: %s', filename)
        return filename

    def saveOut(self, event=None, filename=None, image_stack=False):
        """
        Open dialog for selecting file output filename. Uniqe name is as
        suggested.
        """
        logger.debug(f"saveOut, filename: {filename}")
        if self.oseg.segmentation is not None:
            self.statusBar().showMessage('Saving segmentation data...')
            QApplication.processEvents()
            if image_stack:
                suffix = "{:04d}"
                ofilename = self.oseg.get_standard_ouptut_filename(filetype="dcm", suffix=suffix)
            else:
                ofilename = self.oseg.get_standard_ouptut_filename()
            qf_dialog = QFileDialog.getSaveFileName(
                self,
                "Save file",
                ofilename,
                filter="Former Lisa Format (*.pklz);;New Lisa format HDF5 (*.h5 *.hdf5);; Dicom (*.dcm)")

            # from PyQt5.QtCore import pyqtRemoveInputHook
            # pyqtRemoveInputHook()
            # import pdb; pdb.set_trace()
            filename = str(qf_dialog[0])

            logger.info('Data saved to: ' + filename)

            self.oseg.save_outputs(filename)
            self.statusBar().showMessage('Ready. Data saved to ' + str(filename))

        else:
            self.statusBar().showMessage('No segmentation data!')

    def btnLoadSegmentationFromDicomOverlay(self, event=None):

        self.statusBar().showMessage('Reading dicom overlay')
        dirpath = self.oseg.load_segmentation_from_dicom_overlay(None)
        self.statusBar().showMessage('Ready. Dicom overlay loaded from ' + str(dirpath))

    def btnUpdate(self, event=None):

        self.statusBar().showMessage('Checking for update ...')
        self.oseg.update()
        self.statusBar().showMessage('Update finished. Please restart Lisa')

    def btnAutomaticLiverSeeds(self, event=None):
        self.statusBar().showMessage('Automatic liver seeds...')
        self.oseg.automatic_liver_seeds()
        self.statusBar().showMessage('Ready')

    def btnSaveSegmentation(self, event=None):
        """
        Not fully implemented yet
        :param event:
        :return:
        """
        import dictGUI


        slab_selection = {}
        for label, value in  self.oseg.slab.items():
            slab_selection[label] = True

        slab_selection = dictGUI.dictGui(slab_selection)
        # TODO use some function from oseg to store

    def btnConfig(self, event=None):
        import config


        import organ_segmentation as los


        import lisaConfigGui as lcg


        d = los.lisa_config_init()

        newconf = lcg.configGui(d)

        if newconf is None:
# reset config
            os.remove(
                os.path.join(
                    d['output_datapath'],
                    'organ_segmentation.config')
            )
        else:
            config.save_config(
                newconf,
                os.path.join(
                    newconf['output_datapath'],
                    'organ_segmentation.config')
            )
        
        self.quit(event)

    def btnSaveOutDcmOverlay(self, event=None, filename=None):
        if self.oseg.segmentation is not None:
            self.statusBar().showMessage('Saving segmentation data...')
            QApplication.processEvents()

            output_dicom_dir = self.oseg.save_outputs_dcm_overlay()
            self.statusBar().showMessage('Ready. Data saved to ' + output_dicom_dir)

        else:
            self.statusBar().showMessage('No segmentation data!')

    def btnSaveJSON(self, event=None, filename=None):
        if self.oseg.segmentation is not None:
            self.statusBar().showMessage('Saving json data...')
            QApplication.processEvents()
            if filename==None:
                ofilename = self.oseg.get_standard_ouptut_filename(filetype='json')
                filename = str(QFileDialog.getSaveFileName(
                    self,
                    "Save file",
                    ofilename,
                    filter="*.*")[0])

            self.oseg.output_annotaion_file = filename
            self.oseg.json_annotation_export()
            self.statusBar().showMessage('Ready')
        else:
            self.statusBar().showMessage('No segmentation data!')

    def btnSaveOutDcm(self, event=None, filename=None):
        logger.info('Pressed button "Save Dicom"')
        self.statusBar().showMessage('Saving input data...')
        QApplication.processEvents()
        ofilename = self.oseg.get_standard_ouptut_filename(filetype='dcm')
        filename = str(QFileDialog.getSaveFileName(
            self,
            "Save file",
            ofilename,
            filter="*.*")[0])

        self.oseg.save_input_dcm(filename)
        logger.info('Input data saved to: ' + filename)
        self.statusBar().showMessage('Ready')
        if self.oseg.segmentation is not None:
            self.statusBar().showMessage('Saving segmentation data...')
            QApplication.processEvents()
            filename = filename[:-4] + '-seg' + filename[-4:]
            logger.debug('saving to file: ' + filename)
            # osfilename = self.oseg.get_standard_ouptut_filename(
            #     filetype='dcm')
            self.oseg.save_outputs_dcm(filename)

            self.statusBar().showMessage('Ready')
        else:
            self.statusBar().showMessage('No segmentation data!')

    def action_label_volumetric_vessel_tree(self):
        self.statusBar().showMessage('Performing branch label...')
        nlabel, slabel = self.ui_select_label("Select label with vessel")
        # print("label", slabel)
        self.oseg.label_volumetric_vessel_tree(vessel_label=slabel)
        self.statusBar().showMessage('Ready. Vessel {} branches labeled. '.format(str(slabel)))

    def btnVirtualResectionPV(self):
        # self.statusBar().showMessage('Performing Portal Vein Virtual Resection')
        self._virtual_resection('PV', )

    def btnVirtualResectionPV_new(self):
        self._virtual_resection('PV_new',
                                label=self.oseg.get_slab_value("liver"),
                                vein=2
                                )

    def btnVirtualResectionPlanar(self):
        self._virtual_resection('planar')

    def _virtual_resection(self, method='planar', **kwargs):
        # mport vessel_cut

        self.statusBar().showMessage('Performing virtual resection ...')
        data = {'data3d': self.oseg.data3d,
                'segmentation': self.oseg.segmentation,
                'slab': self.oseg.slab,
                'voxelsize_mm': self.oseg.voxelsize_mm
                }
        cut = virtual_resection.resection(data, method=method)
        self.oseg.segmentation = cut['segmentation']
        self.oseg.slab = cut['slab']

        # rint

        voxelvolume_mm3 = np.prod(self.oseg.voxelsize_mm)
        v1 = np.sum(cut['segmentation'] == self.oseg.slab['liver'])
        v2 = np.sum(cut['segmentation'] == self.oseg.slab['resected_liver'])
        v1 = (v1) * voxelvolume_mm3
        v2 = (v2) * voxelvolume_mm3
        aux = "volume = %.4g l, %.4g/%.4g (%.3g/%.3g %% )" % (
            (v1 + v2) * 1e-6,
            (v1) * 1e-6,
            (v2) * 1e-6,
            100 * v1 / (v1 + v2),
            100 * v2 / (v1 + v2)
        )
        self.setLabelText(self.segBody.lblSegData, aux)


    def btnLesionLocalization(self):
        self.oseg.lesionsLocalization()

    def btnPortalVeinSegmentation(self):
        """
        Function calls segmentation.vesselSegmentation function.
        """

        self.statusBar().showMessage('Vessel segmentation ...')
        # self.oseg.nlabels(2, "porta")
        # self.oseg.nlabels(3, "hepatic_veins")
        self.oseg.nlabels("porta")
        self.oseg.nlabels("hepatic_veins")
        organ_numeric_label, string_label = self.ui_select_label("Organ label")
        vessel_numeric_label, string_label = self.ui_select_label("Vessel label")
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import ipdb; ipdb.set_trace()
        self.oseg.portalVeinSegmentation(inner_vessel_label=vessel_numeric_label, organ_label=organ_numeric_label)
        self.statusBar().showMessage('Ready')

    def __saveVesselTreeGui(self, textLabel):
        # TODO fix names in output files

        textLabel = self.oseg.nlabels(textLabel, return_mode="str")
        fn_yaml = self.oseg.get_standard_ouptut_filename(filetype='yaml', suffix='-vt-' + textLabel+".yaml")
        fn_vtk = self.oseg.get_standard_ouptut_filename(filetype='vtk', suffix='-vt-' + textLabel+".vtk")

        fn_yaml = str(QFileDialog.getSaveFileName(
            self,
            "Save YAML file ",
            fn_yaml,
            filter="*.yaml"))[0]
        fn_vtk = str(QFileDialog.getSaveFileName(
            self,
            "Save VTK file",
            fn_vtk,
            filter="*.vtk"))[0]
        self.oseg.saveVesselTree(textLabel, fn_yaml=fn_yaml, fn_vtk=fn_vtk)
        self.statusBar().showMessage('Ready')

    def btnSavePortalVeinTree(self):
        self.statusBar().showMessage('Saving vessel tree ...')
        QApplication.processEvents()
        textLabel = 'porta'
        self.__saveVesselTreeGui(textLabel)
        self.statusBar().showMessage('Ready')

    def btnSaveActualVesselTree(self):
        self.statusBar().showMessage('Saving vessel tree ...')
        QApplication.processEvents()
        textLabel = self.oseg.nlabels(self.oseg.output_label, return_mode="str")
        msg = 'Saving "' + textLabel +'" vessel tree ...'
        logger.debug(msg)
        self.statusBar().showMessage(msg)
        self.__saveVesselTreeGui(textLabel)
        self.statusBar().showMessage('Ready')

    def btnHepaticVeinsSegmentation(self):
        """
        Function calls segmentation.vesselSegmentation function.
        """
        self.statusBar().showMessage('Vessel segmentation ...')
        self.oseg.hepaticVeinsSegmentation()
        self.statusBar().showMessage('Ready')

    def btnEditSlab(self):
        # run gui
        self.changeWidget('EditSlab')
        # predtim
        print(self.oseg.slab)

        #from PyQt4.QtCore import pyqtRemoveInputHook
        #pyqtRemoveInputHook()
        #import ipdb; ipdb.set_trace()

        # potom
        print(self.oseg.slab)


    def action3DVisualizationWidget(self):
        self.changeWidget("3D Visualization")

    def btnLog(self):
        import logWindow


        import os.path as op


        fn = op.expanduser("~/lisa_data/lisa.log")
        form = logWindow.LogViewerForm(fn) #, qapp=self.app)
        form.show()
        form.exec_()

    def btnRandomRotate(self):
        self.oseg.random_rotate()

    def btnRotateZ(self):
        pass


    def onAlternativeSegmentationParams(self, text):
        self.oseg.update_parameters_based_on_label(str(text))

    def view3D(self):
        # rom seg2mesh import gen_mesh_from_voxels, mesh2vtk, smooth_mesh
        # rom viewer import QVTKViewer
        oseg = self.oseg
        if oseg.segmentation is not None:
            import show_segmentation



            show_segmentation.showSegmentation(
                oseg.segmentation==1,
                voxelsize_mm=oseg.voxelsize_mm,
                degrad=1,
                resize_mm=1.5
            )
            # pts, els, et = gen_mesh_from_voxels(oseg.segmentation,
            #                                     oseg.voxelsize_mm,
            #                                     etype='q', mtype='s')
            # pts = smooth_mesh(pts, els, et,
            #                   n_iter=10)
            # vtkdata = mesh2vtk(pts, els, et)
            # view = QVTKViewer(vtk_data=vtkdata)
            # view.exec_()

        else:
            self.statusBar().showMessage('No segmentation data!')

def get_str(text):
    if sys.version_info.major == 2:
        import PyQt5.QtCore


        if type(text) is QString:
            text = str(text)

    return text
