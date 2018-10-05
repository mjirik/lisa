[![Build Status](https://travis-ci.org/mjirik/lisa.svg)](https://travis-ci.org/mjirik/lisa)
[![Coverage Status](https://coveralls.io/repos/mjirik/lisa/badge.svg)](https://coveralls.io/r/mjirik/lisa)
[![Documentation Status](https://readthedocs.org/projects/liver-surgery-analyser/badge/?version=latest)](https://readthedocs.org/projects/liver-surgery-analyser/?badge=latest)
  
LISA 
=============

LIver Surgery Analyser.

![lisa logo](https://raw.githubusercontent.com/mjirik/lisa/master/applications/LISA256.png)




Install
-------

See our [install notes](https://github.com/mjirik/lisa/blob/master/INSTALL.md) for Linux, Mac OS and Windows 

or use [experimental windows installer](http://home.zcu.cz/~mjirik/lisa/install/setup_lisa.exe)


Install stable branche on Linux or Mac OS with:

    wget https://raw.githubusercontent.com/mjirik/lisa/master/installer.sh
    source installer.sh stable



or use [Lisa in Ubuntu for VirtualBox (deprecated)](http://147.228.240.61/queetech/install/lisa_ubuntu14.04.vdi)



Get sample data
---------------

    python -m lisa --get_sample_data



Run
---

Object (usualy liver) extraction is started by organ_segmentation script

    lisa

or

    ./lisa.sh

or

    python lisa

or

    python -m lisa

Segmentation use two types of seeds wich are setted by left and right mouse
button. For better volume measurement control use additional parameters
for example "-mroi" and "-vs 0.8". If "-mroi" parameter is used you can
select region of interest.


Vessel localization uses saved data (organ.pkl) from organ_segmentation:

    python src/vessels_segmentation.py -bg -i organ.pkl

Now user interactivity is used to set threshold parametr.

Virtual liver resection is based on data stored in previous step 
(vessels.pkl).

    python src/vessel_cut.py -oe -i vessels.pkl

In this script is selected cut on vessel by user interactivity. Resected and
remaining volume is then calculated.

Documentation
-------------

Generated documentation can be found [here](http://147.228.240.61/queetech/Lisa-docs/html/)

Manual generation:

    cd docs
    make latexpdf


Tests
-----

    nosetests
    
    
Video
-----

[![Video webcam tracking](https://img.youtube.com/vi/O408OKV5LhQ/0.jpg)](https://www.youtube.com/watch?v=O408OKV5LhQ)

