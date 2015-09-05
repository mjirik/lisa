#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Generator of histology report

"""
import logging
logger = logging.getLogger(__name__)

import os
import sys
import argparse
import numpy as np

import misc
import csv

import pandas as pd
import datetime

# used by gui dialog
# TODO - remove unneded imports
from PyQt4 import QtCore
from PyQt4.QtCore import pyqtSignal, QObject, QRunnable, QThreadPool, Qt
from PyQt4.QtGui import QMainWindow, QWidget, QDialog, QLabel, QFont,\
    QGridLayout, QFrame, QPushButton, QSizePolicy, QProgressBar, QSpacerItem,\
    QCheckBox, QLineEdit, QApplication, QHBoxLayout, QFileDialog, QMessageBox
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import io3d

class HistologyReport:

    def __init__(self):
        self.data = None
        self.stats = None

    def importFromYaml(self, filename):
        data = misc.obj_from_file(filename=filename, filetype='yaml')
        self.data = self.fixData(data)

    def fixData(self, data):
        """
        fix older versions
        """
        if 'General' in data.keys():
            data['general'] = data.pop('General')
        if 'Graph' in data.keys():
            gr =  data.pop('Graph')
            data['graph'] = {'microstructure': gr}

        try:
            data['general']['used_volume_mm3']
            data['general']['used_volume_px']
        except:
            data['general']['used_volume_mm3'] = data['general']['volume_mm3']
            data['general']['used_volume_px'] = data['general']['volume_px']
        try:
            data['general']['surface_density']
        except:
            data['general']['surface_density'] = None
        return data

    def writeReportToYAML(self, filename='hist_report.yaml'):
        logger.debug('write report to yaml')
        misc.obj_to_file(self.stats, filename=filename, filetype='yaml')

    def writeReportToCSV(self, filename='hist_report.csv'):
        logger.debug('write report to csv')
        data = self.stats['Report']

        with open(filename, 'wb') as csvfile:
            writer = csv.writer(
                csvfile,
                delimiter=';',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL
            )
            # Main
            writer.writerow([data['Main']['Vessel volume fraction (Vv)']])
            writer.writerow([data['Main']['Surface density (Sv)']])
            writer.writerow([data['Main']['Length density (Lv)']])
            writer.writerow([data['Main']['Tortuosity']])
            writer.writerow([data['Main']['Nv']])
            writer.writerow([data['Main']['Vref']])
            # Other
            writer.writerow([data['Other']['Avg length mm']])
            writer.writerow([data['Other']['Total length mm']])
            writer.writerow([data['Other']['Avg radius mm']])
            writer.writerow(data['Other']['Radius histogram'][0])
            writer.writerow(data['Other']['Radius histogram'][1])
            writer.writerow(data['Other']['Length histogram'][0])
            writer.writerow(data['Other']['Length histogram'][1])
            
    def addResultsRecord(self, label='_LABEL_', datapath="_GENERATED_DATA_", recordfilename='statsRecords'):
        logger.debug("Adding Results record to file: "+recordfilename+".*")
        cols = ['label', 'Vv', 'Sv', 'Lv', 'Tort', 'Nv', 'Vref', 'shape', 'voxelsize', 'datetime', 'path']
        
        data_r_m = self.stats['Report']['Main']
        data_g = self.data['general']
        newrow = [[label, 
                    data_r_m['Vessel volume fraction (Vv)'], 
                    data_r_m['Surface density (Sv)'], 
                    data_r_m['Length density (Lv)'], 
                    data_r_m['Tortuosity'], 
                    data_r_m['Nv'], 
                    data_r_m['Vref'], 
                    "-".join(map(str, data_g['shape_px'])), 
                    "-".join(map(str, data_g['voxel_size_mm'])), 
                    str(datetime.datetime.now()),
                    datapath]]
        
        df = pd.DataFrame(newrow, columns=cols)
        
        filename = recordfilename+'.csv'
        append = os.path.isfile(filename)
        with open(filename, 'a') as f:
            if append:
                df.to_csv(f, header=False)
            else:
                df.to_csv(f)

    def generateStats(self, binNum=40):
        # TODO - upravit dokumentaci
        """
        Funkce na vygenerování statistik.

        | Avg length mm: průměrná délka jednotlivých segmentů
        | Avg radius mm: průměrný poloměr jednotlivých segmentů
        | Total length mm: celková délka cév ve vzorku
        | Radius histogram: pole čísel, kde budou data typu:
        |    v poloměru od 1 do 5 je ve vzorku 12 cév, od 5 do 10 je jich 36,
        |    nad 10 je jich 12.
        |    Využijte třeba funkci np.hist()
        | Length histogram: obdoba předchozího pro délky
        """
        stats = {
            'Main': {
                'Vessel volume fraction (Vv)': '-',
                'Surface density (Sv)': '-',
                'Length density (Lv)': '-',
                'Tortuosity': '-',
                'Nv': '-',
                'Vref': '-'
            },
            'Other': {
                'Avg length mm': '-',
                'Total length mm': '-',
                'Avg radius mm': '-',
                'Radius histogram': None,
                'Length histogram': None
            }
        }

        # Get other stats
        radius_array = []
        length_array = []
        for tree_part in self.data['graph']:
            for key in self.data['graph'][tree_part]:
                # from PyQt4.QtCore import pyqtRemoveInputHook; pyqtRemoveInputHook(); import ipdb; ipdb.set_trace()  # noqa BREAKPOINT 
                length_array.append(self.data['graph'][tree_part][key]['lengthEstimation'])
                radius_array.append(self.data['graph'][tree_part][key]['radius_mm'])

        num_of_entries = len(length_array)
        stats['Other']['Total length mm'] = sum(length_array)
        stats['Other']['Avg length mm'] = stats[
            'Other']['Total length mm'] / float(num_of_entries)
        stats['Other']['Avg radius mm'] = sum(
            radius_array) / float(num_of_entries)

        radiusHistogram = np.histogram(radius_array, bins=binNum)
        stats['Other']['Radius histogram'] = [
            radiusHistogram[0].tolist(), radiusHistogram[1].tolist()]
        lengthHistogram = np.histogram(length_array, bins=binNum)
        stats['Other']['Length histogram'] = [
            lengthHistogram[0].tolist(), lengthHistogram[1].tolist()]

        # get main stats
        stats['Main']['Vref'] = float(self.data['general']['used_volume_mm3'])
        tortuosity_array = []
        for key in self.data['graph']['microstructure']:
            tortuosity_array.append(self.data['graph']['microstructure'][key]['tortuosity'])
        num_of_entries = len(tortuosity_array)
        stats['Main']['Tortuosity'] = sum(
            tortuosity_array) / float(num_of_entries)
        stats['Main']['Length density (Lv)'] = float(
            stats['Other']['Total length mm']) / float(
                self.data['general']['used_volume_mm3'])
        stats['Main']['Vessel volume fraction (Vv)'] = self.data[
            'general']['vessel_volume_fraction']
        stats['Main']['Surface density (Sv)'] = self.data[
            'general']['surface_density']
        stats['Main']['Nv'] = self.data['general']['Nv'] 

        # save stats
        self.stats = {'Report': stats}
        logger.debug('Main stats: ' + str(stats['Main']))


class HistologyReportDialog(QDialog):
    def __init__(self, mainWindow=None, histologyAnalyser=None, stats=None):
        """
        histologyAnalyser or stats parameter is required
        """
        self.mainWindow = mainWindow
        self.ha = histologyAnalyser
        self.recordAdded = False

        self.hr = HistologyReport()
        if stats is None:
            self.hr.data = self.ha.stats
        else:
            self.hr.data = stats
        self.hr.generateStats()

        QDialog.__init__(self)
        self.initUI()

        if self.mainWindow is None:
            self.mainWindow = self
            self.resize(800, 600)

        self.mainWindow.setStatusBarText('Finished')

    def setStatusBarText(self,text=""):
        logger.info(text)

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
        report_m = report['Main']
        report_o = report['Other']

        report_label_main = QLabel('Vessel volume fraction (Vv): \t'+str(report_m['Vessel volume fraction (Vv)'])+' [-]\n'
                                +'Surface density (Sv): \t'+str(report_m['Surface density (Sv)'])+' [mm-1]\n'
                                +'Length density (Lv): \t'+str(report_m['Length density (Lv)'])+' [mm-2]\n'
                                +'Tortuosity: \t\t'+str(report_m['Tortuosity'])+' [length/distance]\n'
                                +'Nv: \t\t'+str(report_m['Nv'])+' [mm-3]'
                                )
        report_label_main.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        report_label_other = QLabel('Total vessel length: \t'+str(report_o['Total length mm'])+' [mm]\n'
                                +'Average vessel length: \t'+str(report_o['Avg length mm'])+' [mm]\n'
                                +'Average vessel radius: \t'+str(report_o['Avg radius mm'])+' [mm]\n'
                                +'Vref: \t\t'+str(report_m['Vref'])+' [mm3]'
                                )
        report_label_other.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # mili -> mikro (becouse mili has to much 0)
        histogram_radius = self.HistogramMplCanvas(report_o['Radius histogram'][0],
                                        (np.array(report_o['Radius histogram'][1])*1000).tolist(),
                                        title='Radius histogram',
                                        xlabel="Blood-vessel radius ["+r'$\mu$'+"m]",
                                        ylabel="Count"
                                        )
        histogram_length = self.HistogramMplCanvas(report_o['Length histogram'][0],
                                        (np.array(report_o['Length histogram'][1])*1000).tolist(),
                                        title='Length histogram',
                                        xlabel="Blood-vessel length ["+r'$\mu$'+"m]",
                                        ylabel="Count"
                                        )

        self.ui_gridLayout.addWidget(report_label_main, rstart + 0, 0, 1, 2)
        self.ui_gridLayout.addWidget(report_label_other, rstart + 0, 2, 1, 2)
        self.ui_gridLayout.addWidget(histogram_radius, rstart + 1, 0, 1, 4)
        self.ui_gridLayout.addWidget(histogram_length, rstart + 2, 0, 1, 4)
        rstart +=3

        ### buttons
        btn_yaml = QPushButton("Write statistics to YAML", self)
        btn_yaml.clicked.connect(self.writeYAML)
        btn_csv = QPushButton("Write statistics to CSV", self)
        btn_csv.clicked.connect(self.writeCSV)
        btn_data3d = QPushButton("Save labeled image", self)
        btn_data3d.clicked.connect(self.btnWriteData3d)
        btn_rep_yaml = QPushButton("Write report to YAML", self)
        btn_rep_yaml.clicked.connect(self.writeReportYAML)
        btn_rep_csv = QPushButton("Write report to CSV", self)
        btn_rep_csv.clicked.connect(self.writeReportCSV)

        self.ui_gridLayout.addWidget(btn_yaml, rstart + 0, 0)
        self.ui_gridLayout.addWidget(btn_csv, rstart + 0, 1)
        self.ui_gridLayout.addWidget(btn_data3d, rstart + 0, 2)
        self.ui_gridLayout.addWidget(btn_rep_yaml, rstart + 0, 3)
        self.ui_gridLayout.addWidget(btn_rep_csv, rstart + 0, 4)
        rstart +=1

        ### Stretcher
        self.ui_gridLayout.addItem(QSpacerItem(0,0), rstart + 0, 0,)
        self.ui_gridLayout.setRowStretch(rstart + 0, 1)
        rstart +=1

        ### Setup layout
        self.setLayout(self.ui_gridLayout)
        self.show()
        
    def getSavePath(self, ofilename="stats", extension="yaml"):
        logger.debug("GetSavePathDialog")
        
        filename = str(QFileDialog.getSaveFileName( self,
        "Save file",
        "./"+ofilename+"."+extension,
        filter="*."+extension))
        
        return filename

    def btnWriteData3d(self):
        logger.info("Writing skeleton")
        filename = self.getSavePath("sklabel", "dcm")
        if filename is None or filename == "":
            logger.debug("File save cenceled")
            return
        
        self.mainWindow.setStatusBarText('Saving labeled skeleton image')
        io3d.datawriter.write(self.ha.sklabel, filename)
        # self.ha.writeStatsToYAML(filename)
        self.mainWindow.setStatusBarText('Ready')
        
        if not self.recordAdded:
            self.addResultsRecord()
    def writeYAML(self):
        logger.info("Writing statistics YAML file")
        filename = self.getSavePath("hist_stats", "yaml")
        if filename is None or filename == "":
            logger.debug("File save cenceled")
            return
        
        self.mainWindow.setStatusBarText('Statistics - writing YAML file')
        self.ha.writeStatsToYAML(filename)
        self.mainWindow.setStatusBarText('Ready')
        
        if not self.recordAdded:
            self.addResultsRecord()

    def writeCSV(self):
        logger.info("Writing statistics CSV file")
        filename = self.getSavePath("hist_stats", "csv")
        if filename is None or filename == "":
            logger.debug("File save cenceled")
            return
        
        self.mainWindow.setStatusBarText('Statistics - writing CSV file')
        self.ha.writeStatsToCSV(filename)
        self.mainWindow.setStatusBarText('Ready')
        
        if not self.recordAdded:
            self.addResultsRecord()

    def writeReportYAML(self):
        logger.info("Writing report YAML file")
        filename = self.getSavePath("hist_report", "yaml")
        if filename is None or filename == "":
            logger.debug("File save cenceled")
            return
        
        self.mainWindow.setStatusBarText('Report - writing YAML file')
        self.hr.writeReportToYAML(filename)
        self.mainWindow.setStatusBarText('Ready')
        
        if not self.recordAdded:
            self.addResultsRecord()

    def writeReportCSV(self):
        logger.info("Writing report CSV file")
        filename = self.getSavePath("hist_report", "csv")
        if filename is None or filename == "":
            logger.debug("File save cenceled")
            return
        
        self.mainWindow.setStatusBarText('Report - writing CSV file')
        self.hr.writeReportToCSV(filename)
        self.mainWindow.setStatusBarText('Ready')
        
        if not self.recordAdded:
            self.addResultsRecord()
        
    def addResultsRecord(self):
        # Add results Record
        label = "GUI mode"
        if self.mainWindow.inputfile is None or self.mainWindow.inputfile == "":
            self.hr.addResultsRecord(label=label)
        else:
            self.hr.addResultsRecord(label=label, datapath=self.mainWindow.inputfile)
        self.recordAdded = True

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
            start = xaxis_min + spacing
            end = xaxis_max + (spacing/10.0)

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


if __name__ == "__main__":
    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser reporter'
    )
    parser.add_argument(
        '-i', '--inputfile',
        default="hist_stats.yaml",
        help='input file, yaml file'
    )
    parser.add_argument(
        '-o', '--outputfile',
        default='hist_report',
        help='output file, yaml,csv file (without file extension)'
    )
    parser.add_argument(
        '--gui', action='store_true',
        help='Run in GUI mode')
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()
    
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    if args.debug:
        logger.setLevel(logging.DEBUG)

    # get report
    hr = HistologyReport()
    hr.importFromYaml(args.inputfile)
    
    if args.gui:
        app = QApplication(sys.argv)
        HistologyReportDialog(stats = hr.data)
        sys.exit(app.exec_())
    else:
        hr.generateStats()
        # save report to files
        hr.writeReportToYAML(args.outputfile)
        hr.writeReportToCSV(args.outputfile)
