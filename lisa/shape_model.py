#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""

"""
import numpy as np

import logging
logger = logging.getLogger(__name__)
import argparse

import qmisc


class ShapeModel():
    """
    Cílem je dát dohromady vstupní data s různou velikostí a různou polohou
    objektu. Výstup je pak zapotřebí opět přizpůsobit libovolné velikosti a
    poloze objektu v obraze.

    Model je tvořen polem s velikostí definovanou v konstruktoru (self.shape).
    U modelu je potřeba brát v potaz polohu objektu. Ta je udávána pomocí
    crinfo. To je skupina polí s minimální a maximální hodnotou pro každou osu.

    Trénování je prováděno opakovaným voláním funkce train_one().

    :param model_margin: stanovuje velikost okraje v modelu. Objekt bude ve
    výchozím nastavení vzdálen 0 px od každého okraje.

    """

    def __init__(self, shape=[10, 10, 10]):
        """TODO: to be defined1. """
        self.model = np.ones(shape)
        self.data_number = 0
        self.model_margin = [0, 0, 0]
        pass

    def get_model(self, crinfo, image_shape):
        """
        :param image_shape: Size of output image
        :param crinfo: Array with min and max index of object for each axis.
        [[minx, maxx], [miny, maxy], [minz, maxz]]

        """
        # Průměrování
        mdl = self.model / self.data_number
        print mdl.shape
        print crinfo
        # mdl_res = qmisc.resize_to_shape(mdl, crinfo[0][]
        uncr = qmisc.uncrop(mdl, crinfo, image_shape, resize=True)
        return uncr

    def train_one(self, data):
        """
        """
        crinfo = qmisc.crinfo_from_specific_data(data, margin=self.model_margin)
        datacr = qmisc.crop(data, crinfo=crinfo)
        datacrres = qmisc.resize_to_shape(datacr, self.model.shape)

        self.model += datacrres

        self.data_number += 1

# Tady bude super kód pro trénování

    def train(self, data_arr):
        for data in data_arr:
            self.train_one(data)


def main():
    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    # create file handler which logs even debug messages
    # fh = logging.FileHandler('log.txt')
    # fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    # logger.debug('start')

    # input parser
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        '-i', '--inputfile',
        default=None,
        required=True,
        help='input file'
    )
    parser.add_argument(
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)


if __name__ == "__main__":
    main()
