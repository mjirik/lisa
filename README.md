liver-surgery
=============

Coputer-assisted liver surgery. 

Requirements
------------

    sudo apt-get install python git 

Fallowing packages need to be installed

    sudo apt-get install python-numpy python-scipy python-matplotlib python-sklearn python-dicom cython python-yaml sox python-insighttoolkit3 

SimpleITK is not in ubuntu packages. You can use easy_install

    sudo easy_install -U SimpleITK

Install
-------

    git clone --recursive git@github.com:mjirik/liver-surgery.git

    python ./mysetup.py

or

    git clone git@github.com:mjirik/liver-surgery.git

    git submodule update --init --recursive

    python ./mysetup.py

You can find more install notes in file 'notes.txt'


Run
---

Object (usualy liver) extraction is started by organ_segmentation script

    python python/organ_segmentation.py 

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
