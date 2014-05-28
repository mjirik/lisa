#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Generator of histology report

"""
import logging
logger = logging.getLogger(__name__)

import argparse
import numpy as np

import misc

class HistologyReport:
    def __init__(self):
        self.data = None
        self.stats = None
        pass

    def importFromYaml(self, filename):
        data = misc.obj_from_file(filename=filename, filetype='yaml')
        self.data = data
        
    def writeReportToYAML(self, filename='hist_report.yaml'):
        logger.debug('write report to yaml')
        misc.obj_to_file(self.stats, filename=filename, filetype='yaml')

    def generateStats(self):
        """
        Funkce na vygenerování statistik.

        Avg length mm: průměrná délka jednotlivých segmentů
        Avg radius mm: průměrný poloměr jednotlivých segmentů
        Total length mm: celková délka cév ve vzorku
        Radius histogram: pole čísel, kde budou data typu: v poloměru od 1 do 5
            je ve vzorku 12 cév, od 5 do 10 je jich 36, nad 10 je jich 12.
            Využijte třeba funkci np.hist()
        Length histogram: obdoba předchozího pro délky

        """
        stats = {
            'Avg length mm': 0,
            'Total length mm': 0,
            'Avg radius mm': 0,
            'Radius histogram': None,
            'Length histogram': None,

        }

        radius_array = []
        length_array = []
        for key in self.data['Graph']:
            length_array.append(self.data['Graph'][key]['lengthEstimation'])
            radius_array.append(self.data['Graph'][key]['radius_mm'])
        num_of_entries = len(length_array)
        stats['Total length mm'] = sum(length_array)
        stats['Avg length mm'] = stats['Total length mm']/float(num_of_entries)
        stats['Avg radius mm'] = sum(radius_array)/float(num_of_entries)
        radiusHistogram = np.histogram(radius_array)
        stats['Radius histogram'] = [radiusHistogram[0].tolist(),radiusHistogram[1].tolist()]
        lengthHistogram = np.histogram(length_array)
        stats['Length histogram'] = [lengthHistogram[0].tolist(),lengthHistogram[1].tolist()]

        self.stats = {'Report':stats}
        logger.debug(stats)


if __name__ == "__main__":
    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser reporter'
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        help='input file, yaml file'
    )
    parser.add_argument(
        '-o', '--outputfile',
        default='hist_report.yaml',
        help='output file, yaml file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)
    #ch = logging.StreamHandler()
    #logger.addHandler(ch)

    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # get report
    hr = HistologyReport()
    hr.importFromYaml(args.inputfile)
    hr.generateStats()
    
    # save report to file
    hr.writeReportToYAML(args.outputfile)
