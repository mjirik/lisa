#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 mjirik <mjirik@hp-mjirik>
#
# Distributed under terms of the MIT license.

"""

"""
import unittest
# from nose.plugins.attrib import attr


class TemplateTest(unittest.TestCase):

    def test_update(self):
        import lisa.update_stable as upd
        upd.update(dry_run=True)

if __name__ == "__main__":
    unittest.main()
