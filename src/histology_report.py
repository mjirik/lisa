#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
python src/histology_analyser.py -i ~/data/medical/data_orig/jatra_mikro_data/Nejlepsi_rozliseni_nevycistene -t 6800 -cr 0 -1 100 300 100 300

"""
import logging
logger = logging.getLogger(__name__)


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

        self.stats = stats


if __name__ == "__main__":
    import misc
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(
        description='Histology analyser reporter'
    )
    parser.add_argument('-i', '--inputfile',
            default=None,
            help='input file, yaml file')
#    parser.add_argument('-o', '--outputfile',
#            default='histout.pkl',
#            help='output file')
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.debug)

    hr = HistologyReport()
    hr.importFromYaml(args.inputfile)
    hr.generateStats()

    #data = misc.obj_from_file(args.inputfile, filetype = 'pickle')

    print args.input_is_skeleton
