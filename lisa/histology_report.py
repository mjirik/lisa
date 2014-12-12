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
import csv

class HistologyReport:
    def __init__(self):
        self.data = None
        self.stats = None

    def importFromYaml(self, filename):
        data = misc.obj_from_file(filename=filename, filetype='yaml')
        self.data = data
        
    def writeReportToYAML(self, filename='hist_report'):
        logger.debug('write report to yaml')
        filename = filename+'.yaml'
        misc.obj_to_file(self.stats, filename=filename, filetype='yaml')
        
    def writeReportToCSV(self, filename='hist_report'):
        logger.debug('write report to csv')
        filename = filename+'.csv'
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
            # Other
            writer.writerow([data['Other']['Avg length mm']])
            writer.writerow([data['Other']['Total length mm']])
            writer.writerow([data['Other']['Avg radius mm']])
            writer.writerow(data['Other']['Radius histogram'][0])
            writer.writerow(data['Other']['Radius histogram'][1])
            writer.writerow(data['Other']['Length histogram'][0])
            writer.writerow(data['Other']['Length histogram'][1])

    def generateStats(self,binNum=40):
        # TODO - upravit dokumentaci
        """
        Funkce na vygenerování statistik.

        | Avg length mm: průměrná délka jednotlivých segmentů
        | Avg radius mm: průměrný poloměr jednotlivých segmentů
        | Total length mm: celková délka cév ve vzorku
        | Radius histogram: pole čísel, kde budou data typu: 
        |    v poloměru od 1 do 5 je ve vzorku 12 cév, od 5 do 10 je jich 36, nad 10 je jich 12.
        |    Využijte třeba funkci np.hist()
        | Length histogram: obdoba předchozího pro délky
        """
        stats = {
            'Main':{
                'Vessel volume fraction (Vv)': '-',
                'Surface density (Sv)': '-',
                'Length density (Lv)': '-',
                'Tortuosity': '-',
                'Nv': '-',
                },
            'Other':{
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
        for key in self.data['Graph']:
            length_array.append(self.data['Graph'][key]['lengthEstimation'])
            radius_array.append(self.data['Graph'][key]['radius_mm'])
            
        num_of_entries = len(length_array)
        stats['Other']['Total length mm'] = sum(length_array)
        stats['Other']['Avg length mm'] = stats['Other']['Total length mm']/float(num_of_entries)
        stats['Other']['Avg radius mm'] = sum(radius_array)/float(num_of_entries)
        
        radiusHistogram = np.histogram(radius_array,bins=binNum)
        stats['Other']['Radius histogram'] = [radiusHistogram[0].tolist(),radiusHistogram[1].tolist()]
        lengthHistogram = np.histogram(length_array,bins=binNum)
        stats['Other']['Length histogram'] = [lengthHistogram[0].tolist(),lengthHistogram[1].tolist()]
        
        # get main stats
        tortuosity_array = []
        for key in self.data['Graph']:
            tortuosity_array.append(self.data['Graph'][key]['tortuosity'])
        num_of_entries = len(tortuosity_array)
        stats['Main']['Tortuosity'] = sum(tortuosity_array)/float(num_of_entries)
        stats['Main']['Length density (Lv)'] = float(stats['Other']['Total length mm'])/float(self.data['General']['used_volume_mm3'])
        stats['Main']['Vessel volume fraction (Vv)'] = self.data['General']['vessel_volume_fraction']
        #stats['Main']['Surface density (Sv)'] =
        stats['Main']['Nv'] = float(self.getNv())
        
        # save stats
        self.stats = {'Report':stats}
        logger.debug('Main stats: '+str(stats['Main']))
        
    def getNv(self):
        logger.debug('Computing Nv...')
        nodes = []
        for key in self.data['Graph']:
            edge = self.data['Graph'][key]
            try:
                nodeIdA = edge['nodeIdA']
                nodeIdB = edge['nodeIdB']
                
                if not nodeIdA in nodes:
                    if len(edge['connectedEdgesA']) > 0:
                        nodes.append(nodeIdA)
                        
                if not nodeIdB in nodes:
                    if len(edge['connectedEdgesB']) > 0:
                        nodes.append(nodeIdB)

            except Exception, e:
                logger.warning('getNv(): bad key '+str(e))
                
        logger.debug('Got '+str(len(nodes))+' connected nodes')
        Nv = len(nodes) / float(self.data['General']['used_volume_mm3'])
        
        logger.debug('Nv is '+str(Nv))
        return Nv


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
        default='hist_report',
        help='output file, yaml,csv file (without file extension)'
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
    
    # save report to files
    hr.writeReportToYAML(args.outputfile)
    hr.writeReportToCSV(args.outputfile)
