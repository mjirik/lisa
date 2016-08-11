# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
import logging
from lisa.logWindow import QVBoxLayout
logger = logging.getLogger(__name__)

import sys
import os
import numpy as np
import subprocess

import datetime
import functools

from io3d import datareader
# import segmentation
from seg2mesh import gen_mesh_from_voxels, mesh2vtk, smooth_mesh
import virtual_resection

try:
    from viewer import QVTKViewer
    viewer3D_available = True

except ImportError:
    viewer3D_available = False

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pysegbase/src"))

from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, \
    QFont, QPixmap, QFileDialog, QStyle
from PyQt4 import QtGui
from PyQt4.Qt import QString
try:
    from pysegbase.seed_editor_qt import QTSeedEditor
except:
    logger.warning("Deprecated of pyseg_base as submodule")
    try:
        from pysegbase.seed_editor_qt import QTSeedEditor
    except:
        logger.warning("Deprecated of pyseg_base as submodule")
        from seed_editor_qt import QTSeedEditor

import sed3
import loginWindow

def find_logo():
    import wget
    logopath = os.path.join(path_to_script, "./icons/LISA256.png")
    if os.path.exists(logopath):
        return logopath
    # lisa runtime directory
    logopath = os.path.expanduser("~/lisa_data/.lisa/LISA256.png")
    if not os.path.exists(logopath):
        try:
            wget.download(
                "https://raw.githubusercontent.com/mjirik/lisa/master/lisa/icons/LISA256.png",
                out=logopath
            )
        except:
            logger.warning('logo download failed')
            pass
    if os.path.exists(logopath):
        return logopath

    pass

# GUI
class OrganSegmentationWindow(QMainWindow):

    def __init__(self, oseg=None):

        self.oseg = oseg

        QMainWindow.__init__(self)
        self._initUI()
        self._initMenu()

        if oseg is not None:
            if oseg.data3d is not None:
                self.setLabelText(self.text_dcm_dir, self.oseg.datapath)
                self.setLabelText(self.text_dcm_data, self.getDcmInfo())

        self.statusBar().showMessage('Ready')


    def _initMenu(self):
        menubar = self.menuBar()

        ###### FILE MENU ######
        fileMenu = menubar.addMenu('&File')
        loadSubmenu = fileMenu.addMenu('&Load')
        # load dir
        loadDirAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Directory', self)
        loadDirAction.setStatusTip('Load data from directory (DICOM, jpg, png...)')
        loadDirAction.triggered.connect(self.loadDataDir)
        loadSubmenu.addAction(loadDirAction)
        # load file
        loadFileAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&File', self)
        loadFileAction.setStatusTip('Load data from file (pkl, 3D Dicom, tiff...)')
        loadFileAction.triggered.connect(self.loadDataFile)
        loadSubmenu.addAction(loadFileAction) 
        
        saveSubmenu = fileMenu.addMenu('&Save')
        # save file
        saveFileAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&File', self)
        saveFileAction.setStatusTip('Save data with segmentation')
        saveFileAction.triggered.connect(self.saveOut)
        saveSubmenu.addAction(saveFileAction)   
        # save dicom
        saveDicomAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&DICOM', self)
        saveDicomAction.setStatusTip('Save DICOM data')
        saveDicomAction.triggered.connect(self.btnSaveOutDcm)
        saveSubmenu.addAction(saveDicomAction)
        # save dicom overlay
        saveDicomOverlayAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&DICOM overlay', self)
        saveDicomOverlayAction.setStatusTip('Save overlay DICOM data')
        saveDicomOverlayAction.triggered.connect(self.btnSaveOutDcmOverlay)
        saveSubmenu.addAction(saveDicomOverlayAction)     
        # save PV tree
        savePVAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&PV tree', self)
        savePVAction.setStatusTip('Save Portal Vein 1D model')
        savePVAction.triggered.connect(self.btnSavePortalVeinTree)
        saveSubmenu.addAction(savePVAction)     
        # save HV tree
        saveHVAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&HV tree', self)
        saveHVAction.setStatusTip('Save Hepatic Veins 1D model')
        saveHVAction.triggered.connect(self.btnSaveHepaticVeinsTree)
        saveSubmenu.addAction(saveHVAction)     

        separator = fileMenu.addAction("")
        separator.setSeparator(True)

        autoSeedsAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Automatic liver seeds', self)
        autoSeedsAction.setStatusTip('Automatic liver seeds')
        autoSeedsAction.triggered.connect(self.btnAutomaticLiverSeeds)
        fileMenu.addAction(autoSeedsAction)

        autoSegAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Automatic segmentation', self)
        autoSegAction.setStatusTip('Automatic segmentation')
        autoSegAction.triggered.connect(self.autoSeg)
        fileMenu.addAction(autoSegAction)

        cropAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Crop', self)
        cropAction.setStatusTip('')
        cropAction.triggered.connect(self.cropDcm)
        fileMenu.addAction(cropAction)

        maskAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Mask region', self)
        maskAction.setStatusTip('')
        maskAction.triggered.connect(self.maskRegion)
        fileMenu.addAction(maskAction)

        segFromFile = QtGui.QAction(QtGui.QIcon('exit.png'), '&Segmentation from file', self)
        segFromFile.setStatusTip('Load segmentation from pkl file, raw, ...')
        segFromFile.triggered.connect(self.loadSegmentationFromFile)
        fileMenu.addAction(segFromFile)

        view3DAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&View 3D', self)
        view3DAction.setStatusTip('View segmentation in 3D model')
        view3DAction.triggered.connect(self.view3D)
        fileMenu.addAction(view3DAction)

        separator = fileMenu.addAction("")
        separator.setSeparator(True)

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)
        fileMenu.addAction(exitAction)


        ###### IMAGE MENU ######
        imageMenu = menubar.addMenu('&Image')

        randomRotateAction= QtGui.QAction(QtGui.QIcon('exit.png'), '&Random Rotate', self)
        # autoSeedsAction.setShortcut('Ctrl+Q')
        randomRotateAction.setStatusTip('Random rotation')
        randomRotateAction.triggered.connect(self.btnRandomRotate)
        imageMenu.addAction(randomRotateAction)

        mirrorZAxisAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Mirror Z-axis', self)
        mirrorZAxisAction.setStatusTip('Mirror Z-axis')
        mirrorZAxisAction.triggered.connect(self.oseg.mirror_z_axis)
        imageMenu.addAction(mirrorZAxisAction)


        ###### OPTION MENU ######
        optionMenu = menubar.addMenu('&Option')
        
        configAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Configuration', self)
        configAction.setStatusTip('Config settings')
        configAction.triggered.connect(self.btnConfig)
        optionMenu.addAction(configAction)

        logAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Log', self)
        logAction.setStatusTip('See log file')
        logAction.triggered.connect(self.btnLog)
        optionMenu.addAction(logAction)

        syncAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Sync', self)
        syncAction.setStatusTip('Synchronize files from the server')
        syncAction.triggered.connect(self.sync_lisa_data)
        optionMenu.addAction(syncAction)        

        updateAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Update', self)
        updateAction.setStatusTip('Check new update')
        updateAction.triggered.connect(self.btnUpdate)
        optionMenu.addAction(updateAction)


        ###### CONFIG MENU ######
        configMenu = menubar.addMenu('&Config')
        # combo = QtGui.QComboBox(self)
        for text in self.oseg.segmentation_alternative_params.keys():
            iAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&' + text, self)
            iAction.setStatusTip('Use predefined config "%s"' % (text))
            # something like lambda
            fn = functools.partial(self.onAlternativeSegmentationParams, text)
            iAction.triggered.connect(fn)

            configMenu.addAction(iAction)
        # combo.activated[str].connect(self.onAlternativeSegmentationParams)
        # grid.addWidget(combo, 4, 1)

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


    def _initUI(self):
        window = QtGui.QWidget()
        self.window = window
        self.setCentralWidget(window)
        self.resize(800, 600)
        self.setWindowTitle('LISA')
        self.statusBar().showMessage('Ready')
        mainLayout = QHBoxLayout(window)

        ###### MAIN MENU ######
        menuLayout = QVBoxLayout()
        mainLayout.addLayout(menuLayout)

        #----- logo -----
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
        menuLayout.addWidget(lisa_logo)

        #self.text_dcm_dir = QLabel('DICOM dir:')
        #self.text_dcm_data = QLabel('DICOM data:')
        #grid.addWidget(self.text_dcm_dir, 3, 1, 1, 4)
        #grid.addWidget(self.text_dcm_data, 4, 1, 1, 4)

        #----- menu -----
        #btnLoad = QPushButton("Load", self)
        #btnLoad.clicked.connect(self.btnLoadEvent)
        #menuLayout.addWidget(btnLoad)
        
        #btnBackLoad = QPushButton("Load", self)
        #btnBackLoad.setStyleSheet('QPushButton {background-color: #BA5190; color: #BBBBBB}')
        #btnBackLoad.clicked.connect(self.btnBackLoadEvent)
        #menuLayout.addWidget(btnBackLoad)

        #btnSave = QPushButton("Save", self)
        #btnSave.clicked.connect(self.btnSaveEvent)
        #menuLayout.addWidget(btnSave)

        #btnBackSave = QPushButton("Save", self)
        #btnBackSave.setStyleSheet('QPushButton {background-color: #BA5190; color: #BBBBBB}')
        #btnBackSave.clicked.connect(self.btnBackSaveEvent)
        #menuLayout.addWidget(btnBackSave)

        #btnSegmentation = QPushButton("Segmentation", self)
        #btnSegmentation.clicked.connect(self.btnSegmentationEvent)
        #menuLayout.addWidget(btnSegmentation)

        #btnBackSegmentation = QPushButton("Segmentation", self)
        #btnBackSegmentation.setStyleSheet('QPushButton {background-color: #BA5190; color: #BBBBBB}')
        #btnBackSegmentation.clicked.connect(self.btnBackSegmentationEvent)
        #menuLayout.addWidget(btnBackSegmentation)

        #--load--
        btnLoad = QPushButton("Load", self)
        menuLayout.addWidget(btnLoad)
        menu = QtGui.QMenu(btnLoad)
        group = QtGui.QActionGroup(btnLoad)
        group.setExclusive(True)

        loadFileAction = group.addAction("File")
        loadFileAction.setCheckable(False)
        loadFileAction.triggered.connect(self.loadDataFile)
        menu.addAction(loadFileAction)

        loadDirAction = group.addAction("Directory")
        loadDirAction.setCheckable(False)
        loadDirAction.triggered.connect(self.loadDataDir)
        menu.addAction(loadDirAction)
        btnLoad.setMenu(menu)
        #group.triggered.connect(self.btnLoadEvent)

        #--save--
        btnSave = QPushButton("Save", self)
        menuLayout.addWidget(btnSave)
        menu = QtGui.QMenu(btnSave)
        group = QtGui.QActionGroup(btnSave)
        group.setExclusive(True)

        saveFileAction = group.addAction("File")
        saveFileAction.setCheckable(False)
        saveFileAction.triggered.connect(self.saveOut)
        menu.addAction(saveFileAction)

        saveDicomAction = group.addAction("Dicom")
        saveDicomAction.setCheckable(False)
        saveDicomAction.triggered.connect(self.btnSaveOutDcm)
        menu.addAction(saveDicomAction)

        saveDicomOverlayAction = group.addAction("Dicom overlay")
        saveDicomOverlayAction.setCheckable(False)
        saveDicomOverlayAction.triggered.connect(self.btnSaveOutDcmOverlay)
        menu.addAction(saveDicomOverlayAction)

        savePVTreeAction = group.addAction("PV Tree")
        savePVTreeAction.setCheckable(False)
        savePVTreeAction.triggered.connect(self.btnSavePortalVeinTree)
        menu.addAction(savePVTreeAction)

        saveHVTreeAction = group.addAction("HV Tree")
        saveHVTreeAction.setCheckable(False)
        saveHVTreeAction.triggered.connect(self.btnSaveHepaticVeinsTree)
        menu.addAction(saveHVTreeAction)
        btnSave.setMenu(menu)

        #--segmentation--
        btnSegmentation = QPushButton("Segmentation", self)
        btnSegmentation.clicked.connect(self.btnSegmentationEvent)
        menuLayout.addWidget(btnSegmentation)

        btnBackSegmentation = QPushButton("Segmentation", self)
        btnBackSegmentation.setStyleSheet('QPushButton {background-color: #BA5190; color: #FFFFFF}')
        btnBackSegmentation.clicked.connect(self.btnBackSegmentationEvent)
        menuLayout.addWidget(btnBackSegmentation)
        ####
        #btnSeg = QPushButton("Segmentation", self)
        #menuLayout.addWidget(btnSeg)
        #menu = QtGui.QMenu(btnSeg)
        #group = QtGui.QActionGroup(btnSeg)
        #group.setExclusive(True)

        #segManualAction = group.addAction("Manual")
        #segManualAction.setCheckable(False)
        #segManualAction.triggered.connect(self.liverSeg)
        #menu.addAction(segManualAction)

        #segMaskAction = group.addAction("Mask")
        #segMaskAction.setCheckable(False)
        #segMaskAction.triggered.connect(self.maskRegion)
        #menu.addAction(segMaskAction)

        #segPVAction = group.addAction("Portal Vein")
        #segPVAction.setCheckable(False)
        #segPVAction.triggered.connect(self.btnPortalVeinSegmentation)
        #menu.addAction(segPVAction)

        #segHVAction = group.addAction("Hepatic Vein")
        #segHVAction.setCheckable(False)
        #segHVAction.triggered.connect(self.btnHepaticVeinsSegmentation)
        #menu.addAction(segHVAction)
        #btnSeg.setMenu(menu)
        
        #--others--
        btnCompare = QPushButton("Compare", self)
        btnCompare.clicked.connect(self.compareSegmentationWithFile)
        menuLayout.addWidget(btnCompare)

        btnQuit = QPushButton("Quit", self)
        btnQuit.clicked.connect(self.quit)
        menuLayout.addWidget(btnQuit)


        ###### LOAD MENU #####
        #loadMenu = QtGui.QWidget()
        #loadMenuLayout = QVBoxLayout()
        #mainLayout.addWidget(loadMenu)
        #loadMenu.setLayout(loadMenuLayout)
        #mainLayout.addLayout(loadMenuLayout)
        #loadMenuLayout.addSpacing(126)

        #btnLoadFile = QPushButton("File", self)
        #btnLoadFile.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnLoadFile.clicked.connect(self.loadDataFile)
        #loadMenuLayout.addWidget(btnLoadFile)

        #btnLoadDir = QPushButton("Directory", self)
        #btnLoadDir.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnLoadDir.clicked.connect(self.loadDataDir)
        #loadMenuLayout.addWidget(btnLoadDir)
        

        ####### SAVE MENU #####
        #saveMenu = QWidget()
        #saveMenuLayout = QVBoxLayout()
        #mainLayout.addWidget(saveMenu)
        #saveMenu.setLayout(saveMenuLayout)
        #mainLayout.addLayout(saveMenuLayout)
        #saveMenuLayout.addSpacing(126)

        #btnSaveFile = QPushButton("File", self)
        #btnSaveFile.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSaveFile.clicked.connect(self.saveOut)
        #saveMenuLayout.addWidget(btnSaveFile)

        #btnSaveDcm = QPushButton("Dicom", self)
        #btnSaveDcm.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSaveDcm.clicked.connect(self.btnSaveOutDcm)
        #saveMenuLayout.addWidget(btnSaveDcm)

        #btnSaveDcmOverlay = QPushButton("Dicom overlay", self)
        #btnSaveDcmOverlay.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSaveDcmOverlay.clicked.connect(self.btnSaveOutDcmOverlay)
        #saveMenuLayout.addWidget(btnSaveDcmOverlay)

        #btnSavePV = QPushButton("PV tree", self)
        #btnSavePV.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSavePV.clicked.connect(self.btnSavePortalVeinTree)
        #saveMenuLayout.addWidget(btnSavePV)

        #btnSaveHV = QPushButton("HV tree", self)
        #btnSaveHV.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSaveHV.clicked.connect(self.btnSaveHepaticVeinsTree)
        #saveMenuLayout.addWidget(btnSaveHV)


        ####### SEGMENTATION MENU #####
        #segMenu = QWidget()
        #segMenuLayout = QVBoxLayout()
        #mainLayout.addWidget(segMenu)
        #segMenu.setLayout(segMenuLayout)
        #mainLayout.addLayout(segMenuLayout)
        #segMenuLayout.addSpacing(126)

        #btnSegManual = QPushButton("Manual", self)
        #btnSegManual.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSegManual.clicked.connect(self.liverSeg)
        #segMenuLayout.addWidget(btnSegManual)

        #btnSegMask = QPushButton("Mask", self)
        #btnSegMask.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSegMask.clicked.connect(self.maskRegion)
        #segMenuLayout.addWidget(btnSegMask)

        #btnSegPV = QPushButton("Portal Vein", self)
        #btnSegPV.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSegPV.clicked.connect(self.btnPortalVeinSegmentation)
        #segMenuLayout.addWidget(btnSegPV)

        #btnSegHV = QPushButton("Hepatic Vein", self)
        #btnSegHV.setStyleSheet('QPushButton {background-color: #e600e6; color: #FFFFFF}\n'
        #                            'QPushButton:hover {background-color: #FF44FF; color: #000000;}')
        #btnSegHV.clicked.connect(self.btnHepaticVeinsSegmentation)
        #segMenuLayout.addWidget(btnSegHV)


        ##### SEPARATING LINE #####
        line = QFrame()
        line.setFrameShape(QFrame.VLine)

        mainLayout.addWidget(line)
        window.setLayout(mainLayout)


        ##### BODY #####
        bodyLayout = QVBoxLayout()
        mainLayout.addLayout(bodyLayout)

        #--- title ---
        infoBody = QtGui.QWidget()
        infoBodyLayout = QVBoxLayout()
        bodyLayout.addWidget(infoBody)
        infoBody.setLayout(infoBodyLayout)
        bodyLayout.addLayout(infoBodyLayout)

        lisa_title = QLabel('Liver Surgery Analyser')
        info = QLabel('Developed by:\n' +
                      'University of West Bohemia\n' +
                      'Faculty of Applied Sciences\n' +
                      QString.fromUtf8('M. Jiřík, V. Lukeš - 2013') +
                      '\n\nVersion: ' + self.oseg.version
                      )
        info.setFont(font_info)
        lisa_title.setFont(font_label)
        infoBodyLayout.addWidget(lisa_title)
        infoBodyLayout.addWidget(info)

        #--- segmentation option ---
        segBody = QtGui.QWidget()
        segBodyLayout = QVBoxLayout()
        bodyLayout.addWidget(segBody)
        segBody.setLayout(segBodyLayout)
        bodyLayout.addLayout(segBodyLayout)

        lblSegConfig = QLabel('Choose configure')
        lblSegConfig.setFont(font_label)
        segBodyLayout.addWidget(lblSegConfig)

        segConfig = QtGui.QWidget()
        segConfigLayout = QHBoxLayout()
        segBodyLayout.addWidget(segConfig)
        segConfig.setLayout(segConfigLayout)
        segBodyLayout.addLayout(segConfigLayout)

        btnHearth = QPushButton("Hearth", self)
        btnHearth.setCheckable(True)
        btnHearth.clicked.connect(self.btnHearthEvent)
        segConfigLayout.addWidget(btnHearth)

        btnKidneyL = QPushButton("Kidney Left", self)
        btnKidneyL.setCheckable(True)
        btnKidneyL.clicked.connect(self.btnKidneyLEvent)
        segConfigLayout.addWidget(btnKidneyL)

        btnKidneyR = QPushButton("Kidney Right", self)
        btnKidneyR.setCheckable(True)
        btnKidneyR.clicked.connect(self.btnKidneyREvent)
        segConfigLayout.addWidget(btnKidneyR)
        
        btnLiver = QPushButton("Liver", self)
        btnLiver.setCheckable(True)
        btnLiver.clicked.connect(self.btnLiverEvent)
        segConfigLayout.addWidget(btnLiver)

        btnSimple20 = QPushButton("Simple 2.0 mm", self)
        btnSimple20.setCheckable(True)
        btnSimple20.clicked.connect(self.btnSimple20Event)
        segConfigLayout.addWidget(btnSimple20)

        btnSimple25 = QPushButton("Simple 2.5 mm", self)
        btnSimple25.setCheckable(True)
        btnSimple25.clicked.connect(self.btnSimple25Event)
        segConfigLayout.addWidget(btnSimple25)

        ###
        lblSegType = QLabel('Choose type of segmentation')
        lblSegType.setFont(font_label)
        segBodyLayout.addWidget(lblSegType)

        segType = QtGui.QWidget()
        segTypeLayout = QHBoxLayout()
        segBodyLayout.addWidget(segType)
        segType.setLayout(segTypeLayout)
        segBodyLayout.addLayout(segTypeLayout)
        
        btnSegManual = QPushButton("Manual", self)
        btnSegManual.clicked.connect(self.liverSeg)
        segTypeLayout.addWidget(btnSegManual)

        btnSegMask = QPushButton("Mask", self)
        btnSegMask.clicked.connect(self.maskRegion)
        segTypeLayout.addWidget(btnSegMask)

        btnSegPV = QPushButton("Portal Vein", self)
        btnSegPV.clicked.connect(self.btnPortalVeinSegmentation)
        segTypeLayout.addWidget(btnSegPV)

        btnSegHV = QPushButton("Hepatic Vein", self)
        btnSegHV.clicked.connect(self.btnHepaticVeinsSegmentation)
        segTypeLayout.addWidget(btnSegHV)
        

        #--- file info (footer) ---
        bodyLayout.addStretch()
        self.text_dcm_dir = QLabel('DICOM dir:')
        self.text_dcm_data = QLabel('DICOM data:')
        bodyLayout.addWidget(self.text_dcm_dir)
        bodyLayout.addWidget(self.text_dcm_data)

        #if self.oseg.debug_mode:
        #    btn_debug = QPushButton("Debug", self)
        #    btn_debug.clicked.connect(self.run_debug)
        #    grid.addWidget(btn_debug, rstart - 2, 4)


        ##### OTHERS #####
        mainLayout.addStretch()
        menuLayout.addStretch()
        #loadMenuLayout.addStretch()
        #saveMenuLayout.addStretch()
        #segMenuLayout.addStretch()

        self.btnLoad = btnLoad
        self.btnSave = btnSave
        #self.btnSeg = btnSeg
        #self.btnBackLoad = btnBackLoad
        #self.btnBackSave = btnBackSave
        self.btnBackSegmentation = btnBackSegmentation
        self.btnSegmentation = btnSegmentation
        self.btnCompare = btnCompare
        self.btnHearth = btnHearth
        self.btnKidneyL = btnKidneyL
        self.btnKidneyR = btnKidneyR
        self.btnLiver = btnLiver
        self.btnSimple20 = btnSimple20
        self.btnSimple25 = btnSimple25
        self.btnSegManual = btnSegManual
        self.btnSegMask = btnSegMask
        self.btnSegPV = btnSegPV
        self.btnSegHV = btnSegHV
        #self.loadMenu = loadMenu
        #self.saveMenu = saveMenu
        #self.segMenu = segMenu
        self.infoBody = infoBody
        self.segBody = segBody
        self.segConfig = segConfig
        self.segType = segType

        self.btnSave.setDisabled(True)
        #self.btnSeg.setDisabled(True)
        self.btnSegmentation.setDisabled(True)
        self.btnCompare.setDisabled(True)
        self.btnSegManual.setDisabled(True)
        self.btnSegMask.setDisabled(True)
        self.btnSegPV.setDisabled(True)
        self.btnSegHV.setDisabled(True)

        #self.btnBackLoad.hide()
        #self.btnBackSave.hide()
        self.btnBackSegmentation.hide()
        #self.loadMenu.hide()
        #self.saveMenu.hide()
        #self.segMenu.hide()
        self.segBody.hide()
        self.segConfig.hide()
        self.segType.hide()
        self.show()


    def enableSegType(self):
        self.btnSegManual.setDisabled(False)
        self.btnSegMask.setDisabled(False)
        self.btnSegPV.setDisabled(False)
        self.btnSegHV.setDisabled(False)

    def btnHearthEvent(self, event):
        functools.partial(self.onAlternativeSegmentationParams, "label hearth")
        self.enableSegType()
        self.btnHearth.setChecked(True)
        self.btnKidneyL.setChecked(False)
        self.btnKidneyR.setChecked(False)
        self.btnLiver.setChecked(False)
        self.btnSimple20.setChecked(False)
        self.btnSimple25.setChecked(False)

    def btnKidneyLEvent(self, event):
        functools.partial(self.onAlternativeSegmentationParams, "label kidney L")
        self.enableSegType()
        self.btnHearth.setChecked(False)
        self.btnKidneyL.setChecked(True)
        self.btnKidneyR.setChecked(False)
        self.btnLiver.setChecked(False)
        self.btnSimple20.setChecked(False)
        self.btnSimple25.setChecked(False)

    def btnKidneyREvent(self, event):
        functools.partial(self.onAlternativeSegmentationParams, "label kidney R")
        self.enableSegType()
        self.btnHearth.setChecked(False)
        self.btnKidneyL.setChecked(False)
        self.btnKidneyR.setChecked(True)
        self.btnLiver.setChecked(False)
        self.btnSimple20.setChecked(False)
        self.btnSimple25.setChecked(False)

    def btnLiverEvent(self, event):
        functools.partial(self.onAlternativeSegmentationParams, "label liver")
        self.enableSegType()
        self.btnHearth.setChecked(False)
        self.btnKidneyL.setChecked(False)
        self.btnKidneyR.setChecked(False)
        self.btnLiver.setChecked(True)
        self.btnSimple20.setChecked(False)
        self.btnSimple25.setChecked(False)

    def btnSimple20Event(self, event):
        functools.partial(self.onAlternativeSegmentationParams, "simple 2 mm")
        self.enableSegType()
        self.btnHearth.setChecked(False)
        self.btnKidneyL.setChecked(False)
        self.btnKidneyR.setChecked(False)
        self.btnLiver.setChecked(False)
        self.btnSimple20.setChecked(True)
        self.btnSimple25.setChecked(False)

    def btnSimple25Event(self, event):
        functools.partial(self.onAlternativeSegmentationParams, "simple 2.5 mm")
        self.enableSegType()
        self.btnHearth.setChecked(False)
        self.btnKidneyL.setChecked(False)
        self.btnKidneyR.setChecked(False)
        self.btnLiver.setChecked(False)
        self.btnSimple20.setChecked(False)
        self.btnSimple25.setChecked(True)


    def btnSegmentationEvent(self, event):
        self.btnSegmentation.hide()
        self.btnBackSegmentation.show()
        self.infoBody.hide()
        self.segBody.show()
        self.segConfig.show()
        self.segType.show()

    def btnBackSegmentationEvent(self, event):
        self.btnSegmentation.show()
        self.btnBackSegmentation.hide()
        self.infoBody.show()
        self.segBody.hide()
        self.segConfig.hide()
        self.segType.hide()


    #def btnBackLoadEvent(self, event):
    #    self.btnLoad.show()
    #    self.btnBackLoad.hide()

    #    self.loadMenu.hide()

    #def btnBackSaveEvent(self, event):
    #    self.btnSave.show()
    #    self.btnBackSave.hide()

    #    self.saveMenu.hide()

    #def btnBackSegmentationEvent(self, event):
    #    self.btnSegmentation.show()
    #    self.btnBackSegmentation.hide()

    #    self.segMenu.hide()


    #def btnLoadEvent(self, event):
    #    self.btnBackLoad.show()
    #    self.btnLoad.hide()
    #    self.btnBackSave.hide()
    #    self.btnSave.show()
    #    self.btnBackSegmentation.hide()
    #    self.btnSegmentation.show()
        
    #    self.loadMenu.show()
    #    self.saveMenu.hide()
    #    self.segMenu.hide()

    #def btnSaveEvent(self, event):
    #    self.btnBackLoad.hide()
    #    self.btnLoad.show()
    #    self.btnBackSave.show()
    #    self.btnSave.hide()
    #    self.btnBackSegmentation.hide()
    #    self.btnSegmentation.show()

    #    self.loadMenu.hide()
    #    self.saveMenu.show()
    #    self.segMenu.hide()

    #def btnSegmentationEvent(self, event):
    #    self.btnBackLoad.hide()
    #    self.btnLoad.show()
    #    self.btnBackSave.hide()
    #    self.btnSave.show()
    #    self.btnBackSegmentation.show()
    #    self.btnSegmentation.hide()

    #    self.loadMenu.hide()
    #    self.saveMenu.hide()
    #    self.segMenu.show()

    #def btnMainMenu(self, event):
    #    self.btnLoad.clicked.connect(self.btnLoadEvent)
    #    self.btnSave.clicked.connect(self.btnSaveEvent)
    #    self.btnSegmentation.clicked.connect(self.btnSegmentationEvent)
    #    self.btnLoad.setStyleSheet('')
    #    self.btnSave.setStyleSheet('')
    #    self.btnSegmentation.setStyleSheet('')
    #    self.loadMenu.hide()
    #    self.saveMenu.hide()
    #    self.segMenu.hide()

    def quit(self, event):
        return self.close()

    def run_debug(self, event):
        """
        Stop processing and start debugger in console. Function started by
        pressing "Debug" button wich is availeble after starting Lisa with
        -d parameter.
        """
        logger.debug('== Starting debug mode, leave it with command "c" =')
        from PyQt4.QtCore import pyqtRemoveInputHook
        pyqtRemoveInputHook()
        import ipdb; ipdb.set_trace()  # noqa BREAKPOINT

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

    def __get_datafile(self, app=False, directory=''):
        """
        Draw a dialog for directory selection.
        """

        from PyQt4.QtGui import QFileDialog
        if app:
            dcmdir = QFileDialog.getOpenFileName(
                caption='Select Data File',
                directory=directory
                # ptions=QFileDialog.ShowDirsOnly,
            )
        else:
            app = QApplication(sys.argv)
            dcmdir = QFileDialog.getOpenFileName(
                caption='Select DICOM Folder',
                # ptions=QFileDialog.ShowDirsOnly,
                directory=directory
            )
            # pp.exec_()
            app.exit(0)
        if len(dcmdir) > 0:

            dcmdir = "%s" % (dcmdir)
            dcmdir = dcmdir.encode("utf8")
        else:
            dcmdir = None
        return dcmdir

    def __get_datadir(self, app=False, directory=''):
        """
        Draw a dialog for directory selection.
        """

        from PyQt4.QtGui import QFileDialog
        if app:
            dcmdir = QFileDialog.getExistingDirectory(
                caption='Select DICOM Folder',
                options=QFileDialog.ShowDirsOnly,
                directory=directory
            )
        else:
            app = QApplication(sys.argv)
            dcmdir = QFileDialog.getExistingDirectory(
                caption='Select DICOM Folder',
                options=QFileDialog.ShowDirsOnly,
                directory=directory
            )
            # pp.exec_()
            app.exit(0)
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
        # f oseg.datapath is None:
        #     seg.datapath = dcmreader.get_dcmdir_qt(
        #        app=True,
        #        directory=self.oseg.input_datapath_start
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

            QtGui.QMessageBox.warning(
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

    def loadDataDir(self):
        self.statusBar().showMessage('Reading DICOM directory...')
        QApplication.processEvents()

        oseg = self.oseg
        if 'loaddir' in self.oseg.cache.data.keys():
            directory = self.oseg.cache.get('loaddir')
        else:
            directory = self.oseg.input_datapath_start

        oseg.datapath = self.__get_datadir(
            app=True,
            directory=directory
        )

        if oseg.datapath is None:
            self.statusBar().showMessage('No DICOM directory specified!')
            return
        # head, teil = os.path.split(oseg.datapath)
        self.oseg.cache.update('loaddir', oseg.datapath)

        self.importDataWithGui()

    def importDataWithGui(self):
        oseg = self.oseg

        reader = datareader.DataReader()

        # seg.data3d, metadata =
        datap = reader.Get3DData(oseg.datapath, dataplus_format=True)
        # rint datap.keys()
        # self.iparams['series_number'] = self.metadata['series_number']
        # self.iparams['datapath'] = self.datapath
        oseg.import_dataplus(datap)
        self.setLabelText(self.text_dcm_dir, oseg.datapath)
        self.setLabelText(self.text_dcm_data, self.getDcmInfo())
        self.statusBar().showMessage('Ready')

        #### SET BUTTONS/MENU ####
        #self.btnLoad.show()
        #self.btnBackLoad.hide()
        self.btnSave.setDisabled(False)
        #self.btnSeg.setDisabled(False)
        self.btnSegmentation.setDisabled(False)
        self.btnCompare.setDisabled(False)

        #self.loadMenu.hide()

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

        self.setLabelText(self.text_dcm_data, self.getDcmInfo())
        self.statusBar().showMessage('Ready')

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

    def loadSegmentationFromFile(self):
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
        seg_path = self.__get_datafile(
            app=True,
            directory=self.oseg.input_datapath_start
        )
        if seg_path is None:
            self.statusBar().showMessage('No data path specified!')
            return
        self.oseg.import_segmentation_from_file(seg_path)
        self.statusBar().showMessage('Ready')

    def __evaluation_to_text(self, evaluation):
        overall_score = evaluation['sliver_overall_pts']
        # \
        #     volumetry_evaluation.sliver_overall_score_for_one_couple(
        #         score
        #     )

        logger.info('overall score: ' + str(overall_score))
        return "Sliver score: " + str(overall_score)

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
        if 'loadcomparedir' in self.oseg.cache.data.keys():
            directory = self.oseg.cache.get('loadcomparedir')
        else:
            directory = self.oseg.input_datapath_start
        #
        seg_path = self.__get_datafile(
            app=True,
            directory=directory
        )
        if seg_path is None:
            self.statusBar().showMessage('No data path specified!')
            return
        evaluation, segdiff = \
            self.oseg.sliver_compare_with_other_volume_from_file(seg_path)
        print 'Evaluation: ', evaluation
        # print 'Score: ', score

        text = self.__evaluation_to_text(evaluation)

        segdiff[segdiff == -1] = 2
        logger.debug('segdif unique ' + str(np.unique(segdiff)))

        QApplication.processEvents()
        try:

            ed = sed3.sed3qt(
                self.oseg.data3d,
                seeds=segdiff,
                # contour=(self.oseg.segmentation == self.oseg.slab['liver'])
                contour=self.oseg.segmentation
            )
            ed.exec_()
        except:
            ed = sed3.sed3(
                self.oseg.data3d,
                seeds=segdiff,
                contour=(self.oseg.segmentation == self.oseg.slab['liver'])
            )
            ed.show()

        head, teil = os.path.split(seg_path)
        self.oseg.cache.update('loadcomparedir', head)
        self.setLabelText(self.text_seg_data, text)
        self.statusBar().showMessage('Ready')

    def liverSeg(self):
        self.statusBar().showMessage('Performing liver segmentation ...')
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.oseg.interactivity(
            min_val=self.oseg.viewermin,
            max_val=self.oseg.viewermax)
        self.checkSegData('auto. seg., ')
        self.statusBar().showMessage('Ready')

    def autoSeg(self):
        self.statusBar().showMessage('Performing automatic segmentation...')
        QApplication.processEvents()
        if self.oseg.data3d is None:
            self.statusBar().showMessage('No DICOM data!')
            return

        self.oseg.run_sss()
        self.statusBar().showMessage('Automatic segmentation finished')

    def manualSeg(self):
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
            self.setLabelText(self.text_seg_data, msg + aux)
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No segmentation!')

    def saveOut(self, event=None, filename=None):
        """
        Open dialog for selecting file output filename. Uniqe name is as
        suggested.
        """
        if self.oseg.segmentation is not None:
            self.statusBar().showMessage('Saving segmentation data...')
            QApplication.processEvents()
            ofilename = self.oseg.get_standard_ouptut_filename()
            filename = str(QFileDialog.getSaveFileName(
                self,
                "Save file",
                ofilename,
                filter="*.pklz"))

            logger.info('Data saved to: ' + ofilename)

            self.oseg.save_outputs(filename)
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('No segmentation data!')

    def btnUpdate(self, event=None):

        self.statusBar().showMessage('Checking for update ...')
        self.oseg.update()
        self.statusBar().showMessage('Update finished. Please restart Lisa')

    def btnAutomaticLiverSeeds(self, event=None):
        self.statusBar().showMessage('Automatic liver seeds...')
        self.oseg.automatic_liver_seeds()
        self.statusBar().showMessage('Ready')

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

            self.oseg.save_outputs_dcm_overlay()
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
            filter="*.*"))

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

    def btnVirtualResectionPV(self):
        self._virtual_resection('PV')

    def btnVirtualResectionPlanar(self):
        self._virtual_resection('planar')

    def _virtual_resection(self, method='planar'):
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
        self.setLabelText(self.text_seg_data, aux)


    def btnLesionLocalization(self):
        self.oseg.lesionsLocalization()

    def btnPortalVeinSegmentation(self):
        """
        Function calls segmentation.vesselSegmentation function.
        """

        self.statusBar().showMessage('Vessel segmentation ...')
        self.oseg.portalVeinSegmentation()
        self.statusBar().showMessage('Ready')

    def btnSavePortalVeinTree(self):
        self.statusBar().showMessage('Saving vessel tree ...')
        QApplication.processEvents()
        self.oseg.saveVesselTree('porta')
        self.statusBar().showMessage('Ready')

    def btnSaveHepaticVeinsTree(self):
        self.statusBar().showMessage('Saving vessel tree ...')
        QApplication.processEvents()
        self.oseg.saveVesselTree('hepatic_veins')
        self.statusBar().showMessage('Ready')

    def btnHepaticVeinsSegmentation(self):
        """
        Function calls segmentation.vesselSegmentation function.
        """
        self.statusBar().showMessage('Vessel segmentation ...')
        self.oseg.hepaticVeinsSegmentation()
        self.statusBar().showMessage('Ready')

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
            pts, els, et = gen_mesh_from_voxels(oseg.segmentation,
                                                oseg.voxelsize_mm,
                                                etype='q', mtype='s')
            pts = smooth_mesh(pts, els, et,
                              n_iter=10)
            vtkdata = mesh2vtk(pts, els, et)
            view = QVTKViewer(vtk_data=vtkdata)
            view.exec_()

        else:
            self.statusBar().showMessage('No segmentation data!')
