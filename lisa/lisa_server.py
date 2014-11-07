#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2014 mjirik <mjirik@mjirik-Latitude-E6520>
#
# Distributed under terms of the MIT license.

"""

"""

import logging
logger = logging.getLogger(__name__)
import argparse

import web


class LisaServer:
    def __init__(self):
        """TODO: Docstring for __import__.

        :arg1: TODO
        :returns: TODO

        """
        self.urls = ('/(.*)', 'hello')
        self.app = web.application(self.urls, globals())

    def run(self):
        self.app.run()


class hello:
    def __init__(self):
        # self.oseg = osg.OrganSegmentation()
        pass

    def process_msg(self, data):
        if not data:
            data = 'abs(50)'
        web.header('Access-Control-Allow-Origin',      '*')
        web.header('Access-Control-Allow-Credentials', 'true')
        retval = \
            "<!DOCTYPE html><html><body><div>" + \
            data +\
            "</div></body></html>"
        # "<b>" + str(eval(data)) + "</b>" +
        str(eval(data))
        return str(retval)

    def GET(self, data):
        return self.process_msg(data)

    def POST(self, data):
        return self.process_msg(data)


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
        '-d', '--debug', action='store_true',
        help='Debug mode')
    args = parser.parse_args()

    if args.debug:
        ch.setLevel(logging.DEBUG)

    server = LisaServer()
    server.run()


if __name__ == "__main__":
    main()
