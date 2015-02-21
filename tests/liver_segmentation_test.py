#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 mjirik <mjirik@mjirik-HP-Compaq-Elite-8300-MT>
#
# Distributed under terms of the MIT license.

"""

"""
import unittest
from nose.plugins.attrib import attr

from lisa import liver_segmentation
import os
path_to_script = os.path.dirname(os.path.abspath(__file__))
import logging
logger = logging.getLogger(__name__)


class LiverSegmentationTest(unittest.TestCase):

    # test se pouští ze složky lisa
    # nosetests tests/liver_segmentation_test.py -a actual

    @attr('interactive')
    @attr('actual')
    def test_automatic(self):
        pass

    @attr('interactive')
    def test_liver_segmentation(self):
        import numpy as np
        # import sed3
        img3d = np.random.rand(32, 64, 64) * 4
        img3d[4:24, 12:32, 5:25] = img3d[4:24, 12:32, 5:25] + 25

# seeds
        seeds = np.zeros([32, 64, 64], np.int8)
        seeds[9:12, 13:29, 18:24] = 1
        seeds[9:12, 4:9, 3:32] = 2
# [mm]  10 x 10 x 10        # voxelsize_mm = [1, 4, 3]
        voxelsize_mm = [5, 5, 5]

        ls = liver_segmentation.LiverSegmentation(
            data3d=img3d,
            voxelsize=voxelsize_mm,
            # seeds=seeds
        )
        ls.run()
        volume = np.sum(ls.segmentation == 1) * np.prod(voxelsize_mm)

        # ed = sed3.sed3(img3d, contour=ls.segmentation, seeds=seeds)
        # ed.show()

        # import pdb; pdb.set_trace()

        # mel by to být litr. tedy milion mm3
        self.assertGreater(volume, 900000)
        self.assertLess(volume, 1100000)

    def test_liver_segmenation_just_run(self):
        """
        Tests only if it run. No strong assert.
        """
        import numpy as np
        img3d = np.random.rand(32, 64, 64) * 4
        img3d[4:24, 12:32, 5:25] = img3d[4:24, 12:32, 5:25] + 25

# seeds
        seeds = np.zeros([32, 64, 64], np.int8)
        seeds[9:12, 13:29, 18:24] = 1
        seeds[9:12, 4:9, 3:32] = 2
# [mm]  10 x 10 x 10        # voxelsize_mm = [1, 4, 3]
        voxelsize_mm = [5, 5, 5]

        ls = liver_segmentation.LiverSegmentation(
            data3d=img3d,
            voxelsize=voxelsize_mm,
            # seeds=seeds
        )
        ls.run()

        # ed = sed3.sed3(img3d, contour=ls.segmentation, seeds=seeds)
        # ed.show()

    @attr('incomplete')
    def test_automatickyTest(self):
        ''' nacte prvni dva soubory koncici .mhd z adresare sample_data
        prvni povazuje za originalni a provede na nem segmentaci defaultni
        metodou z liver_segmentation. Pote nacte druhy a povazuje jej za
        rucni segmentaci, na vysledku a rucni provede srovnani a podle
        vysledku vypise verdikt na konzoli'''

        import nearpy
        import io3d
        
        logger.setLevel(logging.DEBUG) #ZDE UPRAVIT POKUD NECHCETE VSECHNY VYPISY
        
        

        path_to_script = os.path.dirname(os.path.abspath(__file__))
        # print path_to_script
        b = path_to_script[0:-5]
        b = b + 'sample_data'
        cesta = b
        
        logger.info('probiha nacitani souboru z adresare sample_data')
        seznamSouboru = liver_segmentation.vyhledejSoubory(cesta)
        reader = io3d.DataReader()
        vektorOriginal = liver_segmentation.nactiSoubor(
            cesta, seznamSouboru, 0, reader)
        originalPole = vektorOriginal[0]
        originalVelikost = vektorOriginal[1]
        logger.info( '***zahajeni segmentace***')
        vytvoreny = liver_segmentation.LiverSegmentation(
            originalPole, originalVelikost)
        vytvoreny.setCisloMetody(2)
        vytvoreny.run()
        segmentovany = vytvoreny.segmentation
        segmentovanyVelikost = vytvoreny.voxelSize
        engine = nearpy.Engine(dim=3)
        logger.info('segmentace dokoncena, nacitani rucni segmentace z adresare sample_data')
        vektorOriginal = liver_segmentation.nactiSoubor(
            cesta, seznamSouboru, 1, reader)
        rucniPole = vektorOriginal[0]
        rucniVelikost = vektorOriginal[1]
        logger.info('zahajeni vyhodnoceni segmentace')
        vysledky = liver_segmentation.vyhodnoceniSnimku(
            rucniPole, rucniVelikost, segmentovany, segmentovanyVelikost, engine)
        logger.info(str(vysledky))
        skore = vysledky[1]
        pravda = True
        if(skore > 75):
            logger.info('metoda funguje uspokojive')
            pravda = False
        if(pravda and (skore > 50)):
            logger.info('metoda funguje relativne dobre')
            pravda = False
        if(pravda):
            logger.info('metoda funguje spatne')
            pravda = False
        self.assertGreater(skore, 5)


        return

if __name__ == "__main__":
    unittest.main()
