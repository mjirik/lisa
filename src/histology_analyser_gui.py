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

from PyQt4 import QtCore, Qt
from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
    QGridLayout, QLabel, QPushButton, QFrame, \
    QFont, QPixmap, QDialog, QVBoxLayout
from PyQt4.Qt import QString

import numpy as np

import datareader
from seed_editor_qt import QTSeedEditor
import py3DSeedEditor

import histology_analyser as HA

class HistologyAnalyserWindow(QMainWindow): 
    HEIGHT = 600
    WIDTH = 800
    
    def __init__(self,inputfile=None,threshold=None,skeleton=False,crop=None,crgui=False):
        self.args_inputfile=inputfile
        self.args_threshold=threshold
        self.args_skeleton=skeleton
        self.args_crop=crop
        self.args_crgui=crgui
        
        QMainWindow.__init__(self)   
        self.initUI()
        
        self.loadData()
        
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
        self.ui_embeddedAppWindow = MessageDialog('Default window')  
        self.ui_embeddedAppWindow_pos = rstart + 1
        self.ui_gridLayout.addWidget(self.ui_embeddedAppWindow, rstart + 1, 1, 1, 2)
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
    
    def processDataGUI(self):
        """
        GUI version of histology analysation algorithm
        """
        
        ### when input is just skeleton
        if self.args_skeleton:  #!!!!! NOT TESTED!!!!
            logger.info("input is skeleton")
            struct = misc.obj_from_file(filename='tmp0.pkl', filetype='pickle')
            self.data3d_skel = struct['skel']
            self.data3d_thr = struct['thr']
            self.data3d = struct['data3d']
            self.metadata = struct['metadata']
            self.ha = HA.HistologyAnalyser(self.data3d, self.metadata, self.args_threshold, nogui=False)
            logger.info("end of is skeleton")
            self.fixWindow() # just to be sure
        else:
            ### Reading/Generating data
            if self.args_inputfile is None: ## Using generated sample data
                logger.info('Generating sample data...')
                self.setStatusBarText('Generating sample data...')
                self.metadata = {'voxelsize_mm': [1, 1, 1]}
                self.data3d = HA.generate_sample_data(2)
            else: ## Normal runtime
                dr = datareader.DataReader()
                self.data3d, self.metadata = dr.Get3DData(self.args_inputfile)
                
            ### Crop data
            self.setStatusBarText('Crop Data')
            if (self.args_crop is None) and (self.args_crgui is True):
                self.data3d = self.cropData(self.data3d)
            elif self.args_crop is not None:    
                crop = self.args_crop
                logger.debug('Croping data: %s', str(crop))
                self.data3d = self.data3d[crop[0]:crop[1], crop[2]:crop[3], crop[4]:crop[5]]
            
            ### Init HistologyAnalyser object
            logger.debug('Init HistologyAnalyser object')
            self.ha = HA.HistologyAnalyser(self.data3d, self.metadata, self.args_threshold, nogui=False)
            
            ### Remove Area
            logger.debug('Remove area')
            self.setStatusBarText('Remove area')
            self.removeArea(self.ha.data3d)

            ### Segmentation
            logger.debug('Segmentation')
            self.setStatusBarText('Segmentation')
            self.showMessage('Segmentation\n1. Select segmentation Area\n2. Select finer segmentation settings\n3. Wait until segmentation is finished')
            
            self.data3d_thr, self.data3d_skel = self.ha.data_to_skeleton()
            self.fixWindow()
        
        ### Show Segmented Data
        logger.debug('Preview of segmented data')
        self.showMessage('Preview of segmented data')
        self.setStatusBarText('Ready')
        self.ha.showSegmentedData(self.data3d_thr, self.data3d_skel)
        self.fixWindow()
        
        ### Computing statistics
        logger.info("######### statistics")
        self.setStatusBarText('Computing Statistics')
        self.showMessage('Computing Statistics\nPlease wait... (it can take very long)')
        
        self.ha.skeleton_to_statistics(self.data3d_thr, self.data3d_skel)
        self.fixWindow()
        
        ### Saving files
        logger.info("##### write to file")
        self.setStatusBarText('Statistics - write file')
        self.showMessage('Writing files\nPlease wait...') ### TO DO!! - file save dialog
        
        self.ha.writeStatsToCSV()
        self.ha.writeStatsToYAML()
        self.ha.writeSkeletonToPickle('skel.pkl')
        #struct = {'skel': self.data3d_skel, 'thr': self.data3d_thr, 'data3d': self.data3d, 'metadata':self.metadata}
        #misc.obj_to_file(struct, filename='tmp0.pkl', filetype='pickle')
        
        ### End
        self.showMessage('Finished')
        self.setStatusBarText('Finished')
        
    def setStatusBarText(self,text=""):
        """
        Changes status bar text
        """
        self.statusBar().showMessage(text)
        QtCore.QCoreApplication.processEvents()
        
    def embedWidget(self, widget=None):     
        """
        Replaces widget embedded that is in gui
        """
        # removes old widget
        self.ui_gridLayout.removeWidget(self.ui_embeddedAppWindow)
        self.ui_embeddedAppWindow.close()
        
        # init new widget
        if widget is None:
            self.ui_embeddedAppWindow = MessageDialog()
        else:
            self.ui_embeddedAppWindow = widget
        
        # add new widget to layout and update
        self.ui_gridLayout.addWidget(self.ui_embeddedAppWindow, self.ui_embeddedAppWindow_pos, 1, 1, 2)
        self.ui_gridLayout.update()
        
        self.fixWindow()
    
    def fixWindow(self):
        """
        Resets Main window size, and makes sure all events (gui changes) were processed
        """
        self.resize(self.WIDTH, self.HEIGHT)
        QtCore.QCoreApplication.processEvents() # this is very important
    
    def showMessage(self, text="Default"):
        newapp = MessageDialog(text)
        self.embedWidget(newapp)
        
    def removeArea(self, data3d=None):
        if data3d is None:
            data3d=self.ha.data3d
            
        newapp = QTSeedEditor(data3d, mode='mask')
        self.embedWidget(newapp)
        self.ui_embeddedAppWindow.status_bar.hide()
        
        newapp.exec_()
        
        self.fixWindow()
        
    def cropData(self,data3d=None):
        if data3d is None:
            data3d=self.data3d
            
        newapp = QTSeedEditor(data3d, mode='crop')
        self.embedWidget(newapp)
        self.ui_embeddedAppWindow.status_bar.hide()
        
        newapp.exec_()
        
        self.fixWindow()
        
        return newapp.img
        
    def loadData(self):
        newapp = LoadDialog(self)
        self.embedWidget(newapp)
        
        newapp.exec_()
        
class MessageDialog(QDialog):
    def __init__(self,text=None):
        self.text = text
        
        QDialog.__init__(self)
        self.initUI()
    
    def initUI(self):
        vbox_app = QVBoxLayout()
        
        font_info = QFont()
        font_info.setBold(True)
        font_info.setPixelSize(20)
        info = QLabel(str(self.text))
        info.setFont(font_info)
        
        vbox_app.addWidget(info)
        vbox_app.addStretch(1) # misto ktery se muze natahovat
        #####vbox_app.addWidget(...) nejakej dalsi objekt
        
        self.setLayout(vbox_app)
        self.show()
        
class LoadDialog(QDialog):
    def __init__(self,mainWindow=None):
        self.mainWindow = mainWindow
        
        QDialog.__init__(self)
        self.initUI()
    
    def initUI(self):
        vbox_app = QVBoxLayout()
        
        font_info = QFont()
        font_info.setBold(True)
        font_info.setPixelSize(20)
        info = QLabel('Load Data window is not yet implemented\nSorry...')
        info.setFont(font_info)
        btn_process = QPushButton("OK", self)
        btn_process.clicked.connect(self.mainWindow.processDataGUI)
       
        
        vbox_app.addWidget(info)
        vbox_app.addStretch(1) # misto ktery se muze natahovat
        vbox_app.addWidget(btn_process)
        
        #vbox_app.setAlignment(Qt.AlignCenter)
        self.setLayout(vbox_app)
        self.show()
        
if __name__ == "__main__":
    HA.main()
