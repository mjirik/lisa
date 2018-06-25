#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script,
                             "../extern/sed3/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)


from PyQt4.QtCore import Qt
from PyQt4.QtGui import QApplication, QMainWindow, QWidget,\
        QGridLayout, QLabel, QPushButton, QFrame, QFileDialog,\
        QFont, QInputDialog, QComboBox, QRadioButton, QButtonGroup

#import apdb
#  apdb.set_trace();
#import scipy.io
import numpy as np
import scipy
#from scipy import sparse
import traceback

# ----------------- my scripts --------
import sed3
#
try:
    import dcmreaddata as dcmr
except:
    from imcut import dcmreaddata as dcmr
try:
    from imcut import pycut
except:
    logger.warning("Deprecated of pyseg_base as submodule")
    import pycut
import argparse
#import sed3

import segmentation
import qmisc
import misc
import organ_segmentation
try:
    from imcut import seed_editor_qt
except:
    logger.warning("Deprecated of pyseg_base as submodule")
    import seed_editor_qt



class MainWindow(QMainWindow):
    def __init__(self):
        self.inps = {'os_inps':{}}
        self.pars = {'os_pars':{}}

        pass

    def run(self, qt_app=None):
        self.qt_app = qt_app
        QMainWindow.__init__(self)


        self.initUI()


    def initUI(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        grid = QGridLayout()
        grid.setSpacing(15)

        # status bar
        self.statusBar().showMessage('Ready')

        font_label = QFont()
        font_label.setBold(True)
        ################ dicom reader
        rstart = 0
        text_dcm = QLabel('DICOM reader')
        text_dcm.setFont(font_label)
        self.text_dcm_dir = QLabel('DICOM dir:')
        self.text_dcm_data = QLabel('DICOM data:')
        self.text_dcm_out = QLabel('output file:')
        grid.addWidget(text_dcm, rstart + 0, 1, 1, 4)
        grid.addWidget(self.text_dcm_dir, rstart + 1, 1, 1, 4)
        grid.addWidget(self.text_dcm_data, rstart + 2, 1, 1, 4)
        grid.addWidget(self.text_dcm_out, rstart + 3, 1, 1, 4)
        btn_dcmdir = QPushButton("Load DICOM", self)
        btn_dcmdir.clicked.connect(self.loadDcmDir)
        btn_dcmred = QPushButton("Organ Segmentation", self)
        btn_dcmred.clicked.connect(self.organSegmentation)
        btn_dcmcrop = QPushButton("Crop", self)
        btn_dcmcrop.clicked.connect(self.cropDcm)
        btn_dcmsave = QPushButton("Save DCM", self)
        btn_dcmsave.clicked.connect(self.saveDcm)
        grid.addWidget(btn_dcmdir, rstart + 4, 1)
        grid.addWidget(btn_dcmred, rstart + 4, 2)
        grid.addWidget(btn_dcmcrop, rstart + 4, 3)
        grid.addWidget(btn_dcmsave, rstart + 4, 4)

        hr = QFrame()
        hr.setFrameShape(QFrame.HLine)
        grid.addWidget(hr, rstart + 5, 0, 1, 6)

        # quit
        btn_quit = QPushButton("Quit", self)
        btn_quit.clicked.connect(self.quit)
        grid.addWidget(btn_quit, 24, 2, 1, 2)

        cw.setLayout(grid)
        self.setWindowTitle('liver-surgery')
        self.show()

    def getPars(self):
        pass

    def saveDcm(self, event=None, filename=None):
        if self.dcm_3Ddata is not None:
            self.statusBar().showMessage('Saving DICOM data...')
            QApplication.processEvents()

            if filename is None:
                filename = str(QFileDialog.getSaveFileName(self, 'Save DCM file',
                                                           filter='Files (*.dcm)'))
            if len(filename) > 0:
                savemat(filename, {'data': self.dcm_3Ddata,
                                   'voxelsize_mm': self.voxel_size_mm,
                                   'offsetmm': self.dcm_offsetmm},
                                   appendmat=False)

                self.setLabelText(self.text_dcm_out, filename)
                self.statusBar().showMessage('Ready')

            else:
                self.statusBar().showMessage('No output file specified!')

        else:
            self.statusBar().showMessage('No DICOM data!')

    def loadDcmDir(self, event=None, filename=None):
        self.statusBar().showMessage('Loading DICOM data...')
        QApplication.processEvents()


        # TODO uninteractive Serie selection



        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(datadir)
        datadir = dcmr.get_dcmdir_qt(self.qt_app)

        # @TODO dialog v qt
        reader = dcmr.DicomReader(datadir)#, qt_app=self.qt_app)
        self.data3d = reader.get_3Ddata()
        self.metadata = reader.get_metaData()
        self.inps['series_number'] = reader.series_number
        self.inps['datadir'] = datadir

        self.statusBar().showMessage('DICOM data loaded')

        #QApplication.processEvents()
#        if len(filename) > 0:
#
#            data = loadmat(filename,
#                           variable_names=['data', 'voxelsize_mm', 'offsetmm'],
#                           appendmat=False)
#
#            self.dcm_3Ddata = data['data']
#            self.voxel_size_mm = data['voxelsize_mm']
#            self.dcm_offsetmm = data['offsetmm']
#            self.setVoxelVolume(self.voxel_size_mm.reshape((3,)))
#            self.setLabelText(self.text_seg_in, filename)
#            self.statusBar().showMessage('Ready')
#
#        else:
#            self.statusBar().showMessage('No input file specified!')
#    def loadDcmDir(self):
#        pass
    def reduceDcm(self):
        pass
    def cropDcm(self):
        pyed = seed_editor_qt.QTSeedEditor(self.data3d, mode='crop')
        pyed.exec_()
        pass
    def organSegmentation(self):
        self.inps['os_inps'].update({
            'data3d':self.data3d,
            'metadata':self.metadata
            })
        os_pars_all = self.pars['os_pars'].copy()
        os_pars_all.update(self.inps['os_inps'])
        os_pars_all.update({'qt_app':self.qt_app})
        print(os_pars_all)
        oseg = organ_segmentation.OrganSegmentation(**os_pars_all)
      #  args.dcmdir,
      #          working_voxelsize_mm=args.voxelsize_mm,
      #          manualroi=args.manualroi,
      #          texture_analysis=args.textureanalysis,
      #          edit_data=args.editdata,
      #          smoothing=args.segmentation_smoothing,
      #          iparams=args.iparams
      #          )

        oseg.interactivity()

        print(
                "Volume " +
                str(oseg.get_segmented_volume_size_mm3() / 1000000.0) + ' [l]')



    def quit(self,event):
        self.close()





def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
            description='Segment vessels from liver \n\
                    \npython organ_segmentation.py\n\
                    \npython organ_segmentation.py -mroi -vs 0.6')
    parser.add_argument('-dd', '--dcmdir',
            default=None,
            help='path to data dir')
    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-vs', '--voxelsize_mm', default='3', type=str,
            help='Insert working voxelsize. It can be number or \
            array of three numbers. \n \
            -vs 3 \n \
            -vs [3,3,5]')
    parser.add_argument('-mroi', '--manualroi', action='store_true',
            help='manual crop before data processing')
    parser.add_argument('-iparams', '--iparams',
            default=None,
            help='filename of ipars file with stored interactivity')
    parser.add_argument('-t', '--tests', action='store_true',
            help='run unittest')
    parser.add_argument('-tx', '--textureanalysis', action='store_true',
            help='run with texture analysis')
    parser.add_argument('-exd', '--exampledata', action='store_true',
            help='run unittest')
    parser.add_argument('-ed', '--editdata', action='store_true',
            help='Run data editor')
    parser.add_argument('-so', '--show_output', action='store_true',
            help='Show output data in viewer')
    parser.add_argument(
            '-ss',
            '--segmentation_smoothing',
            action='store_true',
            help='Smoothing of output segmentation',
            default=False
            )
    args = parser.parse_args()

    # voxelsize_mm can be number or array
    args.voxelsize_mm = np.array(eval(args.voxelsize_mm))

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.tests:
        # hack for use argparse and unittest in one module
        sys.argv[1:] = []
        unittest.main()
        sys.exit()

    if args.exampledata:

        args.dcmdir = '../sample_data/\
                matlab/examples/sample_data/DICOM/digest_article/'

    if args.iparams is not None:
        args.iparams = misc.obj_from_file(args.iparams)
    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
        #data3d, metadata = dcmreaddata.dcm_read_from_dir()

    oseg = OrganSegmentation(args.dcmdir,
                             working_voxelsize_mm=args.voxelsize_mm,
                             manualroi=args.manualroi,
                             texture_analysis=args.textureanalysis,
                             edit_data=args.editdata,
                             smoothing=args.action_segmentation_smoothing,
                             iparams=args.iparams
                             )

    oseg.interactivity()

    #igc = pycut.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()

    #igc.make_gc()
    #igc.show_segmentation()

    # volume
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)

    print(
            "Volume " +
            str(oseg.get_segmented_volume_size_mm3() / 1000000.0) + ' [l]')
    #pyed = sed3.sed3(oseg.data3d, contour =
    # oseg.segmentation)
    #pyed.show()

    if args.show_output:
        oseg.show_output()

    savestring = raw_input('Save output data? (y/n): ')
    #sn = int(snstring)
    if savestring in ['Y', 'y']:

        data = oseg.export()

        misc.obj_to_file(data, "organ.pkl", filetype='pickle')
        misc.obj_to_file(oseg.get_ipars(), 'ipars.pkl', filetype='pickle')
    #output = segmentation.vesselSegmentation(oseg.data3d,
    # oseg.orig_segmentation)


    def checkSegData(self):
        if self.segmentation_data is None:
            self.statusBar().showMessage('No SEG data!')
            return
        nzs = self.segmentation_data.nonzero()
        nn = nzs[0].shape[0]
        if nn > 0:
            aux = ' voxels = %d, volume = %.2e mm3' % (nn, nn * self.voxel_volume)
            self.setLabelText(self.text_seg_data, aux)
            self.setLabelText(self.text_mesh_in, 'segmentation data')
            self.statusBar().showMessage('Ready')

        else:
            self.statusBar().showMessage('Zero SEG data!')

    def autoSeg(self):
        if self.dcm_3Ddata is None:
            self.statusBar().showMessage('No DICOM data!')
            return
        igc = pycut.ImageGraphCut(self.dcm_3Ddata,
                voxelsize=self.voxel_size_mm)

        pyed = QTSeedEditor(self.dcm_3Ddata,
                seeds=self.segmentation_seeds,
                modeFun=igc.interactivity_loop,
                voxelVolume=self.voxel_volume)
        pyed.exec_()

        self.segmentation_data = pyed.getContours()
        self.segmentation_seeds = pyed.getSeeds()
        self.checkSegData()



def main2():
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.run(qt_app=app)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main2()
