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
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy as np
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import datareader
from seed_editor_qt import QTSeedEditor
import misc

import histology_analyser as HA
from histology_report import HistologyReport

class HistologyAnalyserWindow(QMainWindow): 
    HEIGHT = 350 #600
    WIDTH = 800
    
    def __init__(self,inputfile=None,skeleton=False,crop=None,voxelsize=None):
        self.args_inputfile=inputfile
        self.args_skeleton=skeleton
        self.args_crop=crop
        self.args_voxelsize=voxelsize
        
        QMainWindow.__init__(self)   
        self.initUI()
        
        self.showLoadDialog()
        
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
    
    def processDataGUI(self, data3d=None, metadata=None, crgui=True):
        """
        GUI version of histology analysation algorithm
        """
        self.data3d = data3d
        self.metadata = metadata
        self.crgui = crgui
        
        ### when input is just skeleton
        # TODO - edit input_is_skeleton mode to run in gui + test if it works
        if self.args_skeleton: 
            logger.info("input is skeleton")
            struct = misc.obj_from_file(filename='tmp0.pkl', filetype='pickle')
            self.data3d_skel = struct['skel']
            self.data3d_thr = struct['thr']
            self.data3d = struct['data3d']
            self.metadata = struct['metadata']
            self.ha = HA.HistologyAnalyser(self.data3d, self.metadata, nogui=False)
            logger.info("end of is skeleton")
            self.fixWindow() # just to be sure
        else:
            ### Crop data
            self.setStatusBarText('Crop Data')
            if self.args_crop is not None: # --crop cli parameter crop
                crop = self.args_crop
                logger.debug('Croping data: %s', str(crop))
                self.data3d = self.data3d[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
            if self.crgui is True: # --crgui gui crop
                logger.debug('Gui data crop')
                self.data3d = self.showCropDialog(self.data3d)
            
            ### Init HistologyAnalyser object
            logger.debug('Init HistologyAnalyser object')
            self.ha = HA.HistologyAnalyser(self.data3d, self.metadata, nogui=False)
            
            ### Remove Area
            logger.debug('Remove area')
            self.setStatusBarText('Remove area')
            self.showRemoveDialog(self.ha.data3d)

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
        self.data3d_thr, self.data3d_skel = self.ha.data_to_skeleton()
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
        self.embedWidget(newapp)
        self.fixWindow()
        
        newapp.exec_()
        
    def showSegmWaitDialog(self):
        newapp = SegmWaitDialog(self)
        self.embedWidget(newapp)
        self.fixWindow()
        
    def showStatsRunDialog(self):
        newapp = StatsRunDialog(self.ha, self.data3d_thr, self.data3d_skel, mainWindow=self)
        self.embedWidget(newapp)
        self.fixWindow()
        newapp.start()
    
    def showSegmResultDialog(self):
        newapp = SegmResultDialog(self, 
                            histologyAnalyser=self.ha,
                            data3d_thr=self.data3d_thr,
                            data3d_skel=self.data3d_skel
                            )
        self.embedWidget(newapp)
        self.fixWindow()
        newapp.exec_()
    
    def showStatsResultDialog(self,histologyAnalyser=None):
        newapp = StatsResultDialog(self, histologyAnalyser=self.ha )
        self.embedWidget(newapp)
        self.fixWindow(height = 600)
        newapp.exec_()
        
    def showRemoveDialog(self, data3d=None):
        if data3d is None:
            data3d=self.ha.data3d
            
        helpW = QLabel('Remove unneeded data')
        self.changeHelpWidget(widget=helpW)
        
        newapp = QTSeedEditor(data3d, mode='mask')
        newapp.status_bar.hide()
        self.embedWidget(newapp)

        newapp.exec_()
        self.changeHelpWidget(widget=None) # removes help
        
        self.fixWindow()
        
    def showCropDialog(self,data3d=None):
        if data3d is None:
            data3d=self.data3d
            
        helpW = QLabel('Crop data')
        self.changeHelpWidget(widget=helpW)
            
        newapp = QTSeedEditor(data3d, mode='crop')
        newapp.status_bar.hide()
        self.embedWidget(newapp)
        
        newapp.exec_()
        self.changeHelpWidget(widget=None) # removes help
        
        self.fixWindow()
        
        return newapp.img
        
    def showLoadDialog(self):
        newapp = LoadDialog(mainWindow=self, inputfile=self.args_inputfile, voxelsize=self.args_voxelsize)
        self.embedWidget(newapp)
        self.fixWindow()
        newapp.exec_()
        
# TODO - nicer look
class SegmQueryDialog(QDialog):
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
        self.mainWindow.runSegmentation(default=True)
        
    def runSegmManual(self):
        self.mainWindow.runSegmentation(default=False)

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

# TODO - go back to segmentation/crop/mask...
class SegmResultDialog(QDialog):
    def __init__(self, mainWindow=None, histologyAnalyser=None, data3d_thr=None, data3d_skel=None):
        self.mainWindow = mainWindow
        self.ha = histologyAnalyser
        self.data3d_thr=data3d_thr
        self.data3d_skel=data3d_skel
        
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
        btn_write = QPushButton("Write segmented data to file", self)
        btn_write.clicked.connect(self.writeSegmentedData)
        btn_stats = QPushButton("Compute Statistics", self)
        btn_stats.clicked.connect(self.mainWindow.showStatsRunDialog)
        
        self.ui_gridLayout.addWidget(btn_preview, rstart + 0, 1)
        self.ui_gridLayout.addWidget(btn_segm, rstart + 1, 1)
        self.ui_gridLayout.addWidget(btn_write, rstart + 2, 1)
        self.ui_gridLayout.addWidget(btn_stats, rstart + 3, 1)
        rstart += 4
        
        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1
        
        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()
        
    def writeSegmentedData(self): # TODO - choose save path + or maybe just remove
        logger.debug("Writing pickle file")
        self.mainWindow.setStatusBarText('Writing pickle file')
        struct = {'skel': self.data3d_skel, 'thr': self.data3d_thr, 'data3d': self.mainWindow.data3d, 'metadata':self.mainWindow.metadata}
        misc.obj_to_file(struct, filename='tmp0.pkl', filetype='pickle')
        self.mainWindow.setStatusBarText('Ready')
    
    def showSegmentedData(self):
        logger.debug('Preview of segmented data')
        self.ha.showSegmentedData(self.data3d_thr, self.data3d_skel)

# Worker signals for computing statistics
class StatsWorkerSignals(QObject):
    update = pyqtSignal(int,int,int)
    finished = pyqtSignal()

# Worker for computing statistics
class StatsWorker(QRunnable):        
    def __init__(self, ha, data3d_thr, data3d_skel):
        super(StatsWorker, self).__init__()
        self.ha = ha
        self.data3d_thr = data3d_thr
        self.data3d_skel = data3d_skel
        
        self.signals = StatsWorkerSignals()

    def run(self):        
        self.ha.skeleton_to_statistics(self.data3d_thr, self.data3d_skel, guiUpdateFunction=self.signals.update.emit) 
        self.signals.finished.emit()

class StatsRunDialog(QDialog):
    def __init__(self, ha, data3d_thr, data3d_skel, mainWindow=None):
        self.mainWindow = mainWindow
        self.ha = ha
        self.data3d_thr = data3d_thr
        self.data3d_skel = data3d_skel
        
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
        worker = StatsWorker(self.ha, self.data3d_thr, self.data3d_skel)
        worker.signals.update.connect(self.updateInfo)
        worker.signals.finished.connect(self.mainWindow.showStatsResultDialog)

        self.pool.start(worker)
        
        #self.pool.waitForDone()
        
        ### Saving files
        # TODO - move this somewhere else / or delete
        #logger.info("##### write to file")
        #self.setStatusBarText('Statistics - write file')
        #self.showMessage('Writing files (Pickle) \nPlease wait...') 
        #self.ha.writeSkeletonToPickle('skel.pkl')
    
    def updateInfo(self, part=0, whole=1, processPart=1):
        logger.info('processed '+str(part)+'/'+str(whole)+', part '+str(processPart))
        # update progress bar
        step = int((part/float(whole))*100)
        self.pbar.setValue(step)
        # update progress info
        self.ui_partInfo_label.setText('Processing part: '+str(processPart))
        self.ui_progressInfo_label.setText('Progress: '+str(part)+'/'+str(whole))

class StatsResultDialog(QDialog):
    def __init__(self, mainWindow=None, histologyAnalyser=None):
        self.mainWindow = mainWindow
        self.ha = histologyAnalyser
        
        self.hr = HistologyReport()
        self.hr.data = self.ha.stats
        self.hr.generateStats()
        
        QDialog.__init__(self)
        self.initUI()
        
        self.mainWindow.setStatusBarText('Finished')
    
    def initUI(self):
        self.ui_gridLayout = QGridLayout()
        self.ui_gridLayout.setSpacing(15)

        rstart = 0
        
        font_info = QFont()
        font_info.setBold(True)
        font_info.setPixelSize(20)
        label = QLabel('Finished')
        label.setFont(font_info)
        
        self.ui_gridLayout.addWidget(label, rstart + 0, 0, 1, 1)
        rstart +=1
        
        ### histology report
        report = self.hr.stats['Report']
        report_label = QLabel('Total length mm: '+str(report['Total length mm'])+'\n'
                        +'Avg length mm: '+str(report['Avg length mm'])+'\n'
                        +'Avg radius mm: '+str(report['Avg radius mm'])
                        )
        histogram_radius = HistogramMplCanvas(report['Radius histogram'][0],
                                        report['Radius histogram'][1],
                                        title='Radius histogram',
                                        xlabel="Blood-vessel radius [mm]",
                                        ylabel="Number"
                                        )
        histogram_length = HistogramMplCanvas(report['Length histogram'][0],
                                        report['Length histogram'][1],
                                        title='Length histogram',
                                        xlabel="Blood-vessel length [mm]",
                                        ylabel="Number"
                                        )
        
        self.ui_gridLayout.addWidget(report_label, rstart + 0, 0, 1, 3)
        self.ui_gridLayout.addWidget(histogram_radius, rstart + 1, 0, 1, 3)
        self.ui_gridLayout.addWidget(histogram_length, rstart + 2, 0, 1, 3)
        rstart +=3
        
        ### buttons
        btn_yaml = QPushButton("Write statistics to YAML", self)
        btn_yaml.clicked.connect(self.writeYAML)
        btn_csv = QPushButton("Write statistics to CSV", self)
        btn_csv.clicked.connect(self.writeCSV)
        btn_rep_yaml = QPushButton("Write report to YAML", self)
        btn_rep_yaml.clicked.connect(self.writeReportYAML)
        btn_rep_csv = QPushButton("Write report to CSV", self)
        btn_rep_csv.clicked.connect(self.writeReportCSV)
        
        self.ui_gridLayout.addWidget(btn_yaml, rstart + 0, 0)
        self.ui_gridLayout.addWidget(btn_csv, rstart + 0, 1)
        self.ui_gridLayout.addWidget(btn_rep_yaml, rstart + 1, 0)
        self.ui_gridLayout.addWidget(btn_rep_csv, rstart + 1, 1)
        rstart +=2
        
        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0,)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1
        
        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()
    
    def writeYAML(self):
        # TODO - choose save path
        logger.info("Writing statistics YAML file")
        self.mainWindow.setStatusBarText('Statistics - writing YAML file')
        self.ha.writeStatsToYAML()
        self.mainWindow.setStatusBarText('Ready')
    
    def writeCSV(self):
        # TODO - choose save path
        logger.info("Writing statistics CSV file")
        self.mainWindow.setStatusBarText('Statistics - writing CSV file')
        self.ha.writeStatsToCSV()
        self.mainWindow.setStatusBarText('Ready')
        
    def writeReportYAML(self):
        # TODO - choose save path
        logger.info("Writing report YAML file")
        self.mainWindow.setStatusBarText('Report - writing YAML file')
        self.hr.writeReportToYAML()
        self.mainWindow.setStatusBarText('Ready')
        
    def writeReportCSV(self):
        # TODO - choose save path
        logger.info("Writing report CSV file")
        self.mainWindow.setStatusBarText('Report - writing CSV file')
        self.hr.writeReportToCSV()
        self.mainWindow.setStatusBarText('Ready')

class HistogramMplCanvas(FigureCanvas):
    def __init__(self, histogramNumbers, histogramBins, title='', xlabel='', ylabel=''):
        self.histNum =  histogramNumbers
        self.histBins = histogramBins
        self.text_title = title
        self.text_xlabel = xlabel
        self.text_ylabel = ylabel
        
        # init figure
        fig = Figure(figsize=(5, 2.5))
        self.axes = fig.add_subplot(111)
        
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)
        
        # plot data
        self.compute_initial_figure()

        # init canvas (figure -> canvas)
        FigureCanvas.__init__(self, fig)
        #self.setParent(parent)
        
        # setup
        fig.tight_layout()
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        # width of bar
        width = self.histBins[1] - self.histBins[0]
        
        # start values of bars
        pos = np.round(self.histBins,2)
        pos = pos[:-1]  # end value is redundant
        
        # heights of bars
        height = self.histNum
        
        # plot data to figure
        self.axes.bar(pos, height=height, width=width, align='edge')
        
        # set better x axis size
        xaxis_min = np.round(min(self.histBins),2)
        xaxis_max = np.round(max(self.histBins),2)
        self.axes.set_xlim([xaxis_min,xaxis_max])
        
        # better x axis numbering
        spacing = width*4
        start = np.ceil(xaxis_min) + spacing
        end = np.ceil(xaxis_max) + 0.1
        
        xticks_values = np.arange(start,end, spacing)
        xticks_values = np.round(xticks_values, 2)
        xticks = [xaxis_min] + xticks_values.tolist()

        self.axes.set_xticks(xticks)
        
        # labels
        if self.text_title is not '':
            self.axes.set_title(self.text_title)
        if self.text_xlabel is not '':
            self.axes.set_xlabel(self.text_xlabel)
        if self.text_ylabel is not '':
            self.axes.set_ylabel(self.text_ylabel)
        
class LoadDialog(QDialog):
    def __init__(self, mainWindow=None, inputfile=None, voxelsize=None):
        self.mainWindow = mainWindow
        self.inputfile = inputfile
        self.voxelsize = voxelsize
        self.data3d = None
        self.metadata = None
        self.crgui = False
        
        QDialog.__init__(self)
        self.initUI()
        
        if self.inputfile is not None:
            self.importDataWithGui(self.voxelsize)
    
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
        
        crop_box = QCheckBox('Crop data', self)
        crop_box.setCheckState(Qt.Unchecked)
        crop_box.stateChanged.connect(self.cropBox)
        
        btn_process = QPushButton("Continue", self)
        btn_process.clicked.connect(self.finished)
        
        hr2 = QFrame()
        hr2.setFrameShape(QFrame.HLine)
        
        self.ui_gridLayout.addWidget(hr, rstart + 0, 0, 1, 3)
        self.ui_gridLayout.addWidget(info, rstart + 1, 0, 1, 3)
        self.ui_gridLayout.addWidget(btn_dcmdir, rstart + 2, 0)
        self.ui_gridLayout.addWidget(btn_datafile, rstart + 2, 1)
        self.ui_gridLayout.addWidget(btn_dataclear, rstart + 2, 2)
        self.ui_gridLayout.addWidget(self.text_dcm_dir, rstart + 3, 0, 1, 3)
        self.ui_gridLayout.addWidget(self.text_dcm_data, rstart + 4, 0, 1, 3)
        self.ui_gridLayout.addWidget(crop_box, rstart + 5, 0)
        self.ui_gridLayout.addWidget(hr2, rstart + 6, 0, 1, 3)
        self.ui_gridLayout.addWidget(btn_process, rstart + 7, 1)
        rstart +=8
        
        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0,)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1
        
        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()
    
    def finished(self,event):
        if (self.data3d is not None) and (self.metadata is not None):
            self.mainWindow.processDataGUI(self.data3d, self.metadata, self.crgui)
        
    def cropBox(self, state):
        if state == QtCore.Qt.Checked:
            self.crgui = True
        else:
            self.crgui = False
    
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
    
    def loadDataClear(self,event): # TODO - Generate data here => display real data shape
        self.inputfile=None
        self.importDataWithGui()
        self.mainWindow.setStatusBarText('Ready')
        
    def __get_datafile(self, app=False, directory=''):
        """
        Draw a dialog for directory selection.
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
        
    def importDataWithGui(self,voxelsize=None):
        if self.inputfile is None:
            ### Generating data if no input file
            logger.info('Generating sample data...')
            self.mainWindow.setStatusBarText('Generating sample data...')
            self.metadata = {'voxelsize_mm': [1, 1, 1]}
            self.data3d = HA.generate_sample_data(1)
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
        
        if voxelsize is not None:
            self.metadata['voxelsize_mm'] = voxelsize
            
        voxelsize = self.metadata['voxelsize_mm']
        shape = self.data3d.shape
        self.text_dcm_data.setText('Data info: '+str(shape[0])+'x'+str(shape[1])+'x'+str(shape[2])+', '+str(voxelsize))
            
        self.mainWindow.setStatusBarText('Ready')
        
if __name__ == "__main__":
    HA.main()
