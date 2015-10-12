#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Histology analyser GUI
"""

import logging
logger = logging.getLogger(__name__)

import sys
import os.path
path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/dicom2fem/src"))

from PyQt4 import QtCore
from PyQt4.QtCore import pyqtSignal, QObject, QRunnable, QThreadPool, Qt
from PyQt4.QtGui import QMainWindow, QWidget, QDialog, QLabel, QFont,\
    QGridLayout, QFrame, QPushButton, QSizePolicy, QProgressBar, QSpacerItem,\
    QCheckBox, QLineEdit, QApplication, QHBoxLayout, QFileDialog, QMessageBox

import numpy as np
from io3d import datareader

try:
    from pysegbase.seed_editor_qt import QTSeedEditor
except:
    logger.warning("Deprecated of pyseg_base as submodule")
    try:
        from pysegbase.seed_editor_qt import QTSeedEditor
    except:
        logger.warning("Deprecated of pyseg_base as submodule")
        from seed_editor_qt import QTSeedEditor

import misc
import histology_analyser as HA
from histology_report import HistologyReport
from histology_report import HistologyReportDialog


class HistologyAnalyserWindow(QMainWindow):
    HEIGHT = 350 #600
    WIDTH = 800

    def __init__(self, inputfile = None, voxelsize = None, crop = None, args=None, qapp=None):
        self.qapp = qapp
        QMainWindow.__init__(self)
        self.initUI()
        
        self.args = args
        self.showLoadDialog(inputfile = inputfile, voxelsize = voxelsize, crop = crop)

    def initUI(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        self.ui_gridLayout = QGridLayout()
        self.ui_gridLayout.setSpacing(15)

        #self.ui_gridLayout.setColumnMinimumWidth(1, 500)

        # status bar
        self.statusBar().showMessage('Ready')

        rstart = 0

        ### embeddedAppWindow
        self.ui_helpWidget = None
        self.ui_helpWidget_pos = rstart
        self.ui_embeddedAppWindow = QLabel('Default window')
        self.ui_embeddedAppWindow_pos = rstart + 1

        self.ui_gridLayout.addWidget(self.ui_embeddedAppWindow, rstart + 1, 1)
        rstart +=2

        cw.setLayout(self.ui_gridLayout)
        self.setWindowTitle('LISA - Histology Analyser')
        self.show()

    def closeEvent(self, event):
        """
        Runs when user tryes to close main window.

        sys.exit(0) - to fix wierd bug, where process is not terminated.
        """
        sys.exit(0)

    def processDataGUI(self, inputfile=None, data3d=None, metadata=None, crgui=True):
        """
        GUI version of histology analysation algorithm
        """
        self.inputfile = inputfile
        self.data3d = data3d
        self.masked = None
        self.metadata = metadata
        self.crgui = crgui

        ### Gui Crop data
        if self.crgui is True: 
            logger.debug('Gui data crop')
            self.data3d = self.showCropDialog(self.data3d)

        ### Init HistologyAnalyser object
        logger.debug('Init HistologyAnalyser object')
        self.ha = HA.HistologyAnalyser(
                self.data3d, self.metadata,
                nogui=False, qapp=self.qapp,
                aggregate_near_nodes_distance=self.args.aggregatenearnodes)

        self.ha.set_anotation(inputfile)
        ### Remove Area (mask)
        logger.debug('Remove area')
        bad_mask = True
        if self.args.maskfile is not None: # try to load mask from file
            logger.debug('Loading mask from file...')
            try:
                mask = misc.obj_from_file(filename=self.args.maskfile, filetype='pickle')
                if self.ha.data3d.shape == mask.shape:
                    self.ha.data3d_masked = mask
                    self.ha.data3d[mask == 0] = np.min(self.ha.data3d)
                    bad_mask = False
                else:
                    logger.error('Mask file has wrong dimensions '+str(mask.shape))
            except Exception, e:
                logger.error('Error when processing mask file: '+str(e))
            if bad_mask == True: logger.debug('Falling back to GUI mask mode')
            
        if bad_mask == True: # gui remove area (mask)
            self.setStatusBarText('Remove area')
            self.showRemoveDialog(self.ha.data3d)
            self.ha.data3d_masked = self.masked
            
        if self.args.savemask and bad_mask == True: # save mask to file
            logger.debug('Saving mask to file...')
            misc.obj_to_file(self.masked, filename='mask.pkl', filetype='pickle')

        ### Segmentation
        self.showSegmQueryDialog()

    def runSegmentation(self, default=False):
        logger.debug('Segmentation')

        # show segmentation wait screen
        self.setStatusBarText('Segmentation')
        self.showSegmWaitDialog()

        # use default segmentation parameters
        if default is True:
            self.ha.nogui = True
            self.ha.threshold = 2800 #7000
            self.ha.binaryClosing = 1 #2
            self.ha.binaryOpening = 1 #1
        else:
            self.ha.nogui = False
            self.ha.threshold = -1

        # run segmentation
        self.ha.data_to_skeleton()
        if default is True:
            self.ha.nogui = False

        self.fixWindow()
        self.setStatusBarText('Ready')

        ### Show segmented data
        self.showSegmResultDialog()


    def setStatusBarText(self,text=""):
        """
        Changes status bar text
        """
        self.statusBar().showMessage(text)
        QtCore.QCoreApplication.processEvents()

    def fixWindow(self,width=None,height=None):
        """
        Resets Main window size, and makes sure all events (gui changes) were processed
        """
        if width is None:
            width = self.WIDTH
        if height is None:
            height = self.HEIGHT

        self.resize(width, height)
        QtCore.QCoreApplication.processEvents() # this is very important

    def embedWidget(self, widget=None):
        """
        Replaces widget embedded that is in gui
        """
        # removes old widget
        self.ui_gridLayout.removeWidget(self.ui_embeddedAppWindow)
        self.ui_embeddedAppWindow.close()

        # init new widget
        if widget is None:
            self.ui_embeddedAppWindow = QLabel()
        else:
            self.ui_embeddedAppWindow = widget

        # add new widget to layout and update
        self.ui_gridLayout.addWidget(self.ui_embeddedAppWindow, self.ui_embeddedAppWindow_pos, 1)
        self.ui_gridLayout.update()

        self.fixWindow()

    def changeHelpWidget(self, widget=None):
        # removes old widget
        if self.ui_helpWidget is not None:
            self.ui_gridLayout.removeWidget(self.ui_helpWidget)
            self.ui_helpWidget.close()

        # init new widget
        if widget is None:
            self.ui_helpWidget = None
        else:
            self.ui_helpWidget = widget

        # add new widget to layout and update
        if self.ui_helpWidget is not None:
            self.ui_gridLayout.addWidget(self.ui_helpWidget, self.ui_helpWidget_pos, 1)
            self.ui_gridLayout.update()

        self.fixWindow()

    def showSegmQueryDialog(self):
        logger.debug('Segmentation Query Dialog')
        newapp = SegmQueryDialog(self)
        newapp.signal_finished.connect(self.runSegmentation)
        self.embedWidget(newapp)
        self.fixWindow()
        # TODO odstrani exec_() . Melo by to byt patrne volano vne tridy a jen
        # jednou pro celou aplikaci.

        newapp.exec_()

    def showSegmWaitDialog(self):
        newapp = SegmWaitDialog(self)
        self.embedWidget(newapp)
        self.fixWindow()

    def showStatsRunDialog(self):
        newapp = StatsRunDialog(self.ha, mainWindow=self)
        self.embedWidget(newapp)
        self.fixWindow()
        newapp.start()

    def showSegmResultDialog(self):
        newapp = SegmResultDialog(self, histologyAnalyser=self.ha)
        self.embedWidget(newapp)
        self.fixWindow()
        newapp.exec_()

    def showHistologyReportDialog(self):
        newapp = HistologyReportDialog(self, histologyAnalyser=self.ha )
        self.embedWidget(newapp)
        self.fixWindow(height = 600)
        newapp.exec_()

    def showRemoveDialog(self, data3d=None):
        if data3d is None:
            data3d=self.ha.data3d

        helpW = QLabel('Remove unneeded data')
        self.changeHelpWidget(widget=helpW)
        newapp = QTSeedEditor(data3d, mode='mask', 
                              voxelSize=self.metadata['voxelsize_mm'])
        newapp.status_bar.hide()
        self.embedWidget(newapp)

        newapp.exec_()
        self.masked = newapp.masked

        self.changeHelpWidget(widget=None) # removes help

        self.fixWindow()

    def showCropDialog(self,data3d=None):
        if data3d is None:
            data3d=self.data3d

        helpW = QLabel('Crop data')
        self.changeHelpWidget(widget=helpW)

        newapp = QTSeedEditor(data3d, mode='crop',
                              voxelSize=self.metadata['voxelsize_mm'])
        newapp.status_bar.hide()
        self.embedWidget(newapp)

        newapp.exec_()
        self.changeHelpWidget(widget=None) # removes help

        self.fixWindow()

        return newapp.img

    def showLoadDialog(self, inputfile = None, voxelsize = None, crop = None):
        newapp = LoadDialog(mainWindow = self,
                            inputfile = inputfile,
                            voxelsize = voxelsize,
                            crop = crop)
        self.embedWidget(newapp)
        newapp.signal_finished.connect(self.processDataGUI)
        self.fixWindow()
        newapp.exec_()

# TODO - nicer look
class SegmQueryDialog(QDialog):
    signal_finished = pyqtSignal(bool)

    def __init__(self, mainWindow=None):
        self.mainWindow = mainWindow

        QDialog.__init__(self)
        self.initUI()

    def initUI(self):
        self.ui_gridLayout = QGridLayout()
        self.ui_gridLayout.setSpacing(15)

        rstart = 0

        info_label = QLabel('Default segmentation settings?')

        self.ui_gridLayout.addWidget(info_label, rstart + 0, 0, 1, 3)
        rstart +=1

        ### Buttons
        btn_default = QPushButton("Use default parameters", self)
        btn_default.clicked.connect(self.runSegmDefault)
        btn_manual = QPushButton("Set segmentation parameters and area manualy", self)
        btn_manual.clicked.connect(self.runSegmManual)

        self.ui_gridLayout.addWidget(btn_default, rstart + 0, 1)
        self.ui_gridLayout.addWidget(btn_manual, rstart + 1, 1)
        rstart +=2

        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0,)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1

        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()

    def runSegmDefault(self):
        self.signal_finished.emit(True)

    def runSegmManual(self):
        self.signal_finished.emit(False)

# TODO - more detailed info about what is happening + dont show help when using default parameters + nicer look
class SegmWaitDialog(QDialog):
    def __init__(self, mainWindow=None):
        self.mainWindow = mainWindow

        QDialog.__init__(self)
        self.initUI()

    def initUI(self):
        self.ui_gridLayout = QGridLayout()
        self.ui_gridLayout.setSpacing(15)

        rstart = 0


        font_info = QFont()
        font_info.setBold(True)
        font_info.setPixelSize(15)
        info_label = QLabel('Segmentation\n1. Select segmentation Area\n2. Select finer segmentation settings\n3. Wait until segmentation is finished')
        info_label.setFont(font_info)

        self.ui_gridLayout.addWidget(info_label, rstart + 0, 0, 1, 3)
        rstart +=1

        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1

        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()

# TODO - go back to crop/mask...
class SegmResultDialog(QDialog):
    def __init__(self, mainWindow=None, histologyAnalyser=None):
        self.mainWindow = mainWindow
        self.ha = histologyAnalyser

        QDialog.__init__(self)
        self.initUI()

    def initUI(self):
        self.ui_gridLayout = QGridLayout()
        self.ui_gridLayout.setSpacing(15)

        rstart = 0

        font_info = QFont()
        font_info.setBold(True)
        font_info.setPixelSize(20)
        info_label = QLabel('Segmentation finished')
        info_label.setFont(font_info)

        self.ui_gridLayout.addWidget(info_label, rstart + 0, 0, 1, 3)
        rstart += 1

        btn_preview = QPushButton("Show segmentation result", self)
        btn_preview.clicked.connect(self.showSegmentedData)
        btn_segm = QPushButton("Go back to segmentation", self)
        btn_segm.clicked.connect(self.mainWindow.showSegmQueryDialog)
        btn_stats = QPushButton("Compute Statistics", self)
        btn_stats.clicked.connect(self.mainWindow.showStatsRunDialog)

        self.ui_gridLayout.addWidget(btn_preview, rstart + 0, 1)
        self.ui_gridLayout.addWidget(btn_segm, rstart + 1, 1)
        self.ui_gridLayout.addWidget(btn_stats, rstart + 2, 1)
        rstart += 3

        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1

        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()

    def showSegmentedData(self):
        logger.debug('Preview of segmented data')
        self.ha.showSegmentedData()

# Worker signals for computing statistics
class StatsWorkerSignals(QObject):
    update = pyqtSignal(int,int,str)
    finished = pyqtSignal()

# Worker for computing statistics
class StatsWorker(QRunnable):
    def __init__(self, ha):
        super(StatsWorker, self).__init__()
        self.ha = ha

        self.signals = StatsWorkerSignals()

    def run(self):
        self.ha.data_to_statistics(guiUpdateFunction=self.signals.update.emit)
        logger.debug('data to statistics finished')
        self.signals.finished.emit()

class StatsRunDialog(QDialog):
    def __init__(self, ha, mainWindow=None):
        self.mainWindow = mainWindow
        self.ha = ha

        QDialog.__init__(self)
        self.initUI()

        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)

        if self.mainWindow is not None:
            self.mainWindow.setStatusBarText('Computing Statistics...')

    def initUI(self):
        self.ui_gridLayout = QGridLayout()
        self.ui_gridLayout.setSpacing(15)

        rstart = 0

        ### Info
        font_info = QFont()
        font_info.setBold(True)
        font_info.setPixelSize(20)
        info_label=QLabel('Computing Statistics:')
        info_label.setFont(font_info)

        self.ui_gridLayout.addWidget(info_label, rstart + 0, 0)
        rstart +=1

        ### Progress bar
        self.pbar=QProgressBar(self)
        self.pbar.setValue(0)
        self.pbar.setGeometry(30, 40, 200, 25)

        self.ui_gridLayout.addWidget(self.pbar, rstart + 0, 0)
        rstart +=1

        ### Progress info
        self.ui_partInfo_label=QLabel('Processing part: -')
        self.ui_progressInfo_label=QLabel('Progress: -/-')

        self.ui_gridLayout.addWidget(self.ui_partInfo_label, rstart + 0, 0)
        self.ui_gridLayout.addWidget(self.ui_progressInfo_label, rstart + 1, 0)
        rstart +=2

        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1

        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()

    def start(self):
        logger.info("Computing Statistics")
        worker = StatsWorker(self.ha)
        worker.signals.update.connect(self.updateInfo)
        worker.signals.finished.connect(self.mainWindow.showHistologyReportDialog)

        self.pool.start(worker)

    def updateInfo(self, part=0, whole=1, processPart="-"):
        # update progress bar
        step = int((part/float(whole))*100)
        self.pbar.setValue(step)
        # update progress info
        self.ui_partInfo_label.setText('Processing part: '+str(processPart))
        self.ui_progressInfo_label.setText('Progress: '+str(part)+'/'+str(whole))

class LoadDialog(QDialog):
    signal_finished = pyqtSignal(str,np.ndarray,dict,bool)

    def __init__(self, mainWindow=None, inputfile=None, voxelsize=None, crop=None):
        self.mainWindow = mainWindow

        self.inputfile = inputfile
        self.data3d = None
        self.metadata = None

        self.box_vs = False
        self.box_crop = False
        self.box_crgui = False

        QDialog.__init__(self)
        self.initUI()

        if voxelsize is not None:
            self.vs_box.setCheckState(Qt.Checked)
            self.vs_box.stateChanged.emit(Qt.Checked)
            self.manual_vs_z.setText(str(voxelsize[0]))
            self.manual_vs_y.setText(str(voxelsize[1]))
            self.manual_vs_x.setText(str(voxelsize[2]))

        if crop is not None:
            self.crop_box.setCheckState(Qt.Checked)
            self.crop_box.stateChanged.emit(Qt.Checked)
            self.manual_crop_z_s.setText(str(crop[0]))
            self.manual_crop_z_e.setText(str(crop[1]))
            self.manual_crop_y_s.setText(str(crop[2]))
            self.manual_crop_y_e.setText(str(crop[3]))
            self.manual_crop_x_s.setText(str(crop[4]))
            self.manual_crop_x_e.setText(str(crop[5]))

        if self.inputfile is not None:
            self.importDataWithGui()

    def initUI(self):
        self.ui_gridLayout = QGridLayout()
        self.ui_gridLayout.setSpacing(15)

        rstart = 0

        ### Title
        font_label = QFont()
        font_label.setBold(True)
        font_label.setPixelSize(20)
        ha_title = QLabel('Histology analyser')
        ha_title.setFont(font_label)
        ha_title.setAlignment(Qt.AlignCenter)

        self.ui_gridLayout.addWidget(ha_title, rstart + 0, 1)
        rstart +=1

        ### Load files buttons etc.
        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        font_info = QFont()
        font_info.setBold(True)
        info = QLabel('Load Data:')
        info.setFont(font_info)

        btn_dcmdir = QPushButton("Load directory (DICOM)", self)
        btn_dcmdir.clicked.connect(self.loadDataDir)
        btn_datafile = QPushButton("Load file", self)
        btn_datafile.clicked.connect(self.loadDataFile)
        btn_dataclear = QPushButton("Generated data", self)
        btn_dataclear.clicked.connect(self.loadDataClear)

        self.text_dcm_dir = QLabel('Data path: -')
        self.text_dcm_data = QLabel('Data info: -')

        hr2 = QFrame()
        hr2.setFrameShape(QFrame.HLine)

        self.ui_gridLayout.addWidget(hr, rstart + 0, 0, 1, 3)
        self.ui_gridLayout.addWidget(info, rstart + 1, 0, 1, 3)
        self.ui_gridLayout.addWidget(btn_dcmdir, rstart + 2, 0)
        self.ui_gridLayout.addWidget(btn_datafile, rstart + 2, 1)
        self.ui_gridLayout.addWidget(btn_dataclear, rstart + 2, 2)
        self.ui_gridLayout.addWidget(self.text_dcm_dir, rstart + 3, 0, 1, 3)
        self.ui_gridLayout.addWidget(self.text_dcm_data, rstart + 4, 0, 1, 3)
        self.ui_gridLayout.addWidget(hr2, rstart + 5, 0, 1, 3)
        rstart += 6

        # settings layout
        layout_settings = QGridLayout()
        layout_settings.setSpacing(15)

        # Manual setting of voxelsize
        self.vs_box = QCheckBox('Manual voxel size', self)
        self.vs_box.stateChanged.connect(self.vsBox)

        self.manual_vs_z = QLineEdit()
        self.manual_vs_y = QLineEdit()
        self.manual_vs_x = QLineEdit()

        self.vs_box.setCheckState(Qt.Unchecked)
        self.vs_box.stateChanged.emit(Qt.Unchecked)

        layout_settings.addWidget(self.vs_box, 0, 0)

        layout_vs = QHBoxLayout()
        layout_vs.setSpacing(0)
        layout_vs.addWidget(QLabel('Z: '))
        layout_vs.addWidget(self.manual_vs_z)
        layout_settings.addLayout(layout_vs, 0, 1)

        layout_vs = QHBoxLayout()
        layout_vs.setSpacing(0)
        layout_vs.addWidget(QLabel('Y: '))
        layout_vs.addWidget(self.manual_vs_y)
        layout_settings.addLayout(layout_vs, 0, 2)

        layout_vs = QHBoxLayout()
        layout_vs.setSpacing(0)
        layout_vs.addWidget(QLabel('X: '))
        layout_vs.addWidget(self.manual_vs_x)
        layout_settings.addLayout(layout_vs, 0, 3)

        # Manual setting of crop
        self.crop_box = QCheckBox('Manual crop data', self)
        self.crop_box.stateChanged.connect(self.cropBox)

        self.manual_crop_z_s = QLineEdit()
        self.manual_crop_z_e = QLineEdit()
        self.manual_crop_y_s = QLineEdit()
        self.manual_crop_y_e = QLineEdit()
        self.manual_crop_x_s = QLineEdit()
        self.manual_crop_x_e = QLineEdit()

        self.crop_box.setCheckState(Qt.Unchecked)
        self.crop_box.stateChanged.emit(Qt.Unchecked)

        layout_settings.addWidget(self.crop_box, 1, 0)

        layout_crop = QHBoxLayout()
        layout_crop.setSpacing(0)
        layout_crop.addWidget(QLabel('Z: '))
        layout_crop.addWidget(self.manual_crop_z_s)
        layout_crop.addWidget(QLabel('-'))
        layout_crop.addWidget(self.manual_crop_z_e)
        layout_settings.addLayout(layout_crop, 1, 1)

        layout_crop = QHBoxLayout()
        layout_crop.setSpacing(0)
        layout_crop.addWidget(QLabel('Y: '))
        layout_crop.addWidget(self.manual_crop_y_s)
        layout_crop.addWidget(QLabel('-'))
        layout_crop.addWidget(self.manual_crop_y_e)
        layout_settings.addLayout(layout_crop, 1, 2)

        layout_crop = QHBoxLayout()
        layout_crop.setSpacing(0)
        layout_crop.addWidget(QLabel('X: '))
        layout_crop.addWidget(self.manual_crop_x_s)
        layout_crop.addWidget(QLabel('-'))
        layout_crop.addWidget(self.manual_crop_x_e)
        layout_settings.addLayout(layout_crop, 1, 3)

        # GUI crop checkbox
        crop_gui_box = QCheckBox('GUI Crop data', self)
        crop_gui_box.setCheckState(Qt.Unchecked)
        crop_gui_box.stateChanged.connect(self.cropGuiBox)

        layout_settings.addWidget(crop_gui_box, 2, 0)

        # add settings layout to main layout
        self.ui_gridLayout.addLayout(layout_settings, rstart + 0, 0, 1, 3)
        rstart += 1

        # process button
        hr3 = QFrame()
        hr3.setFrameShape(QFrame.HLine)

        self.btn_process = QPushButton("Continue", self)
        self.btn_process.clicked.connect(self.finished)

        self.ui_gridLayout.addWidget(hr3, rstart + 0, 0, 1, 3)
        self.ui_gridLayout.addWidget(self.btn_process, rstart + 1, 1)
        rstart +=2

        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0,)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1

        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()

    def finished(self,event):
        if (self.data3d is not None) and (self.metadata is not None):
            # if manually set voxelsize
            if self.box_vs is True:
                try:
                    manual_vs = [float(self.manual_vs_z.text()),
                                 float(self.manual_vs_y.text()),
                                 float(self.manual_vs_x.text())]
                    logger.debug('Manual voxel size: %s', str(manual_vs))
                    self.metadata['voxelsize_mm'] = manual_vs
                except:
                    logger.warning('Error when setting manual voxel size - bad parameters')
                    QMessageBox.warning(self, 'Error', 'Bad manual voxelsize parameters!!!')
                    return

            # if manually set data crop (--crop, -cr)
            if self.box_crop is True:
                try:
                    crop_raw = [self.manual_crop_z_s.text(),
                            self.manual_crop_z_e.text(),
                            self.manual_crop_y_s.text(),
                            self.manual_crop_y_e.text(),
                            self.manual_crop_x_s.text(),
                            self.manual_crop_x_e.text()]
                    crop = []
                    for c in crop_raw:
                        if str(c) == '' or str(c).lower() == 'none' or str(c).lower() == 'end' or str(c).lower() == 'start':
                            crop.append(None)
                        else:
                            crop.append(int(c))

                    logger.debug('Croping data: %s', str(crop))
                    self.data3d = self.data3d[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]].copy()
                except:
                    logger.warning('Error when manually croping data - bad parameters')
                    QMessageBox.warning(self, 'Error', 'Bad manual crop parameters!!!')
                    return
            
            if self.inputfile is None:
                self.inputfile = ""
            self.signal_finished.emit(self.inputfile, self.data3d, self.metadata, self.box_crgui)

        else:
            # if no data3d or metadata loaded
            logger.warning('No input data or metadata')
            QMessageBox.warning(self, 'Error', 'No input data or metadata!!!')

    def vsBox(self, state):
        if state == QtCore.Qt.Checked:
            self.box_vs = True
            self.manual_vs_z.setEnabled(True)
            self.manual_vs_y.setEnabled(True)
            self.manual_vs_x.setEnabled(True)
        else:
            self.box_vs = False
            self.manual_vs_z.setEnabled(False)
            self.manual_vs_y.setEnabled(False)
            self.manual_vs_x.setEnabled(False)

    def cropBox(self, state):
        if state == QtCore.Qt.Checked:
            self.box_crop = True
            self.manual_crop_z_s.setEnabled(True)
            self.manual_crop_z_e.setEnabled(True)
            self.manual_crop_y_s.setEnabled(True)
            self.manual_crop_y_e.setEnabled(True)
            self.manual_crop_x_s.setEnabled(True)
            self.manual_crop_x_e.setEnabled(True)
        else:
            self.box_crop = False
            self.manual_crop_z_s.setEnabled(False)
            self.manual_crop_z_e.setEnabled(False)
            self.manual_crop_y_s.setEnabled(False)
            self.manual_crop_y_e.setEnabled(False)
            self.manual_crop_x_s.setEnabled(False)
            self.manual_crop_x_e.setEnabled(False)

    def cropGuiBox(self, state):
        if state == QtCore.Qt.Checked:
            self.box_crgui = True
        else:
            self.box_crgui = False

    def loadDataDir(self,event):
        self.mainWindow.setStatusBarText('Reading directory...')
        self.inputfile = self.__get_datadir(
            app=True,
            directory=''
        )
        if self.inputfile is None:
            self.mainWindow.setStatusBarText('No directory specified!')
            return
        self.importDataWithGui()

    def loadDataFile(self,event):
        self.mainWindow.setStatusBarText('Reading data file...')
        self.inputfile = self.__get_datafile(
            app=True,
            directory=''
        )
        if self.inputfile is None:
            self.mainWindow.setStatusBarText('No data path specified!')
            return
        self.importDataWithGui()

    def loadDataClear(self,event):
        self.inputfile=None
        self.importDataWithGui()
        self.mainWindow.setStatusBarText('Ready')

    def __get_datafile(self, app=False, directory=''):
        """
        Draw a dialog for file selection.
        """

        from PyQt4.QtGui import QFileDialog
        if app:
            dcmdir = QFileDialog.getOpenFileName(
                caption='Select Data File',
                directory=directory
                #options=QFileDialog.ShowDirsOnly,
            )
        else:
            app = QApplication(sys.argv)
            dcmdir = QFileDialog.getOpenFileName(
                caption='Select Data File',
                #options=QFileDialog.ShowDirsOnly,
                directory=directory
            )
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
                caption='Select Folder',
                options=QFileDialog.ShowDirsOnly,
                directory=directory
            )
        else:
            app = QApplication(sys.argv)
            dcmdir = QFileDialog.getExistingDirectory(
                caption='Select Folder',
                options=QFileDialog.ShowDirsOnly,
                directory=directory
            )
            app.exit(0)
        if len(dcmdir) > 0:
            dcmdir = "%s" % (dcmdir)
            dcmdir = dcmdir.encode("utf8")
        else:
            dcmdir = None

        return dcmdir

    def importDataWithGui(self):
        if self.inputfile is None:
            ### Generating data if no input file
            logger.info('Generating sample data...')
            self.mainWindow.setStatusBarText('Generating sample data...')
            self.metadata = {'voxelsize_mm': [1, 1, 1]}
            self.data3d = HA.generate_sample_data(1,0,0)
            self.text_dcm_dir.setText('Data path: '+'Generated sample data')
        else:
            try:
                reader = datareader.DataReader()
                self.data3d, self.metadata = reader.Get3DData(self.inputfile)
            except:
                logger.error("Unexpected error: "+str(sys.exc_info()[0]))
                self.mainWindow.setStatusBarText('Bad file/folder!!!')
                return

            self.text_dcm_dir.setText('Data path: '+str(self.inputfile))

        voxelsize = self.metadata['voxelsize_mm']
        shape = self.data3d.shape
        self.text_dcm_data.setText('Data info: '+str(shape[0])+'x'+str(shape[1])+'x'+str(shape[2])+', '+str(voxelsize))

        self.mainWindow.setStatusBarText('Ready')

if __name__ == "__main__":
    HA.main()
