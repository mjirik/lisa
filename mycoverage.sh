#/bin/bash
rm -rf cover
nosetests --with-coverage --cover-inclusive --cover-html --cover-package=audiosupport,cxpokus,datareader,dcmreaddata1,experiments,inspector,lesions,liver_surgery,misc,organ_segmentation,qmisc,resection,segmentation,show_segmentation,show3,simple_segmentation,support_structure_segmentation,texture_analysis,uiThreshold,vessel_cut,vessels_segmentation,viewer3
