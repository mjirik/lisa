import SimpleITK as sitk
import numpy as np

image = sitk.ReadImage('/home/mjirik/data/medical/data_orig/sliver07/01/liver-orig001.mhd')
sz = image.GetSize()

imagetrans = sitk.Image(sz[0],sz[1],sz[2], sitk.sitkInt16)

for i in range(0,sz[0]):
    print(i)
    for j in range(0,sz[1]):
        for k in range(0,sz[2]):
            imagetrans[i,j,k]=image[i,j,k]

sitk.Show(imagetrans)
