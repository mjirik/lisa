#! /usr/bin/python
# -*- coding: utf-8 -*-

# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(path_to_script, "../extern/pycat/"))
#sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))

import itk
import SimpleITK as sitk

imraw = sitk.ReadImage(os.path.join(path_to_script,"../sample_data/jatra_5mm/IM-0001-0005.dcm"))

pimsh = sitk.Show(imraw)

import pdb; pdb.set_trace()


