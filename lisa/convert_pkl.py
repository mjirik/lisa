#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module is used for converting data from pkl format to raw.
"""
import argparse
import numpy as np

import misc
import qmisc

def main():

    parser = argparse.ArgumentParser(description=__doc__) # 'Simple VTK Viewer')

    parser.add_argument('-i','--inputfile', default=None,
                      help='File as .pkl')
    parser.add_argument('-o','--outputfile', default=None,
                      help='Output file. Filetype is given by extension.')

    parser.add_argument('-k','--key', default='data3d',
                      help='Which key should be writen to output file. \
                        Default is "data3d". You can use "segmentation"')
    args = parser.parse_args()
    data = misc.obj_from_file(args.inputfile, filetype = 'pickle')
    data3d_uncrop = qmisc.uncrop(data[args.key], data['crinfo'], data['orig_shape'])
    #import ipdb; ipdb.set_trace() # BREAKPOINT
    import SimpleITK as sitk
    sitk_img = sitk.GetImageFromArray(data3d_uncrop.astype(np.uint16), isVector=True)
    sitk.WriteImage(sitk_img, args.outputfile)
    print("Warning: .mhd and .raw format has corupted metadta. You can edit it manually.")
if __name__ == "__main__":
    main()
