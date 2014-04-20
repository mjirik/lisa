LISA 
=============

LIver Surgery Analyser.

![lisa logo](https://raw.githubusercontent.com/mjirik/lisa/master/applications/LISA256.png)



Install
-------

See our [install notes](https://github.com/mjirik/lisa/blob/master/INSTALL.md) for Linux, Mac OS and Windows 


Get stable branche

    git clone --recursive -b stable https://github.com/mjirik/lisa.git

or for current developement

    git clone --recursive git@github.com:mjirik/lisa.git

or

    git clone git@github.com:mjirik/lisa.git
    git submodule update --init --recursive



Get sample data
---------------

    python mysetup.py -d



Run
---

Object (usualy liver) extraction is started by organ_segmentation script

    ./lisa.sh

or

    python lisa.py

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


Tests
-----

    nosetests
    
    
