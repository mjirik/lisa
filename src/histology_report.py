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
        stats['Radius histogram'] = np.histogram(radius_array)
        stats['Length histogram'] = np.histogram(length_array)

        logger.debug(stats)
        self.stats = stats


if __name__ == "__main__":
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    #ch = logging.StreamHandler()
    #logger.addHandler(ch)

    #logger.debug('input params')

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
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    #logger.error("pokus")
    hr = HistologyReport()
    hr.importFromYaml(args.inputfile)
    hr.generateStats()

    #data = misc.obj_from_file(args.inputfile, filetype = 'pickle')

    #print args.input_is_skeleton
