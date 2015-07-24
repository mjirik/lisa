#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
from setuptools import setup



#from cx_Freeze import setup, Executable

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

#setup(
#        name = " ",
#        version = "0.1",
#        description = " ",
#        options = {"build_exe": {"packages": ["numpy.lib.format"]}},
#        executables = [Executable("./src/cxpokus.py")]
#        #executables = [Executable("./src/organ_segmentation.py")]
#        )
#        #executables = [Executable("./src/organ_segmentation.py")]


setup(
    name = "liversurgery",
    version = "0.0.4",
    author = "Miroslav Jirik",
    author_email = "miroslav.jirik@gmail.com",
    description = ("Comuter aided liver surgery"
        ""),
    license = "BSD",
    keywords = "liver surgery computer vision",
    url = "http://github.com/mjirik",
    packages=['tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: BSD License",
    ],
)
