#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module is used for converting data from pkl format to raw.
"""
import argparse

import misc
import qmisc

def main():

    parser = argparse.ArgumentParser(description=__doc__) # 'Simple VTK Viewer')

    parser.add_argument('-i','--inputfile', default=None,
                      help='File as .pkl')
    parser.add_argument('-o','--outputfile', default=None,
                      help='File as raw')
    args = parser.parse_args()
    data = misc.obj_from_file(args.inputfile, filetype = 'pickle')
    data3d_uncrop = qmisc.uncrop(data['data3d'], data['crinfo'], data['orig_shape'])
    import ipdb; ipdb.set_trace() # BREAKPOINT
    import SimpleITK as sitk
    sitk_img = sitk.GetImageFromArray(data3d_uncrop, isVector=True)
    sitk.WriteImage(sitk_img, args.outputfile)
if __name__ == "__main__":
    main()
