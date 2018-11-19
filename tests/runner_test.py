#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
import unittest
import os.path as op
from nose.plugins.attrib import attr
from lisa import runner


class MyClass:
    def __init__(self):
        self.retval = None
        pass

    def return_string(self):
        self.retval = "string"
        return "string"

    def return_int(self):
        self.retval = 1
        return 1

    def return_third_param(self, first, second, third):
        self.retval = third


class RunnerTest(unittest.TestCase):

    def test_runner(self):
        mcls = MyClass()
        rnr = runner.Runner(mcls)
        rnr.extend(["return_string", "return_int"])
        rnr.run()

        self.assertEqual(mcls.retval, 1)


    def test_runner_by_function(self):
        mcls = MyClass()
        rnr = runner.Runner(mcls)
        rnr.extend([mcls.return_string])
        rnr.run()

        self.assertEqual(mcls.retval, "string")

    def test_runner_complex(self):

        mcls = MyClass()
        rnr = runner.Runner(mcls)
        rnr.extend([mcls.return_string, ["return_third_param",[1, 2, 3], {}]])
        rnr.run()

        self.assertEqual(mcls.retval, 3)
