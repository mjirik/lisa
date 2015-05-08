#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os.path

path_to_script = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(path_to_script, "../extern/pyseg_base/src"))
sys.path.append(os.path.join(path_to_script, "../extern/sed3/"))
#sys.path.append(os.path.join(path_to_script, "../extern/"))
#import featurevector
import unittest

import logging
logger = logging.getLogger(__name__)


#import apdb
#  apdb.set_trace();\
#import scipy.io
import numpy as np
import scipy

# ----------------- my scripts --------
import sed3
try:
    import dcmreaddata as dcmr
except:
    from pysegbase import dcmreaddata as dcmr

try:
    from pysegbase import pycut
except:
    logger.warning("Deprecated of pyseg_base as submodule")
    import pycut
import argparse
import sed3
import collections
import segmentation
import qmisc
import io3d
import ipdb
import scipy.ndimage.filters as filters
import scipy.signal
import multipolyfit as mpf
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from io3d import datareader
from scipy.ndimage.measurements import label
from scipy.ndimage import morphology





class SupportStructureSegmentation():
    def __init__(self,
            data3d = None,
            voxelsize_mm = None,
            autocrop = True,
            autocrop_margin_mm = [10,10,10],
            modality = 'CT',
            slab = {'none':0,'llung':4, 'bone':8,'rlung':9,'heart':10,'diaphragm':5},
            maximal_lung_diff = 0.2,
	    rad_diaphragm=4,
	    smer=None,
	    ):

        """
        Segmentaton of support structures for liver segmentatio based on
        locati0on prior.
        """



        self.data3d = data3d
        self.voxelsize_mm = voxelsize_mm
        self.autocrop = autocrop
        self.autocrop_margin_mm = np.array(autocrop_margin_mm)
        self.autocrop_margin = self.autocrop_margin_mm/self.voxelsize_mm
        self.crinfo = [[0,-1],[0,-1],[0,-1]]
        self.segmentation = np.zeros(data3d.shape)
        self.slab = slab
	self.maximal_lung_diff = maximal_lung_diff
	self.rad_diaphragm=rad_diaphragm
	self.smer=smer



        #import pdb; pdb.set_trace()


    def run(self):
	self.bone_segmentation()
	self.spine_segmentation()
	self.lungs_segmentation()
	self.heart_segmentation()


    def import_data(self, data):
        self.data = data
        self.data3d = data['data3d']
        self.voxelsize_mm = data['voxelsize_mm']

    def import_dir(self, datadir):
        reader = dcmr.DicomReader(datadir)
        self.data3d = reader.get_3Ddata()
        self.metadata = reader.get_metaData()
        #self.data3d, self.metadata = dcmr.dcm_read_from_dir(datadir)
        self.voxelsize_mm = np.array(self.metadata['voxelsize_mm'])


    def convolve_structure_heart( self , size=9 ):
	a = np.zeros(( size , size , size ))
	c = int (np.floor( size / 2 ))
	a[c,c,c]=1
	structure = filters.gaussian_filter( a , self.voxelsize_mm )
	structure[:c,:,:] *= -1
	structure[c,:,:] = 0
	return structure

    

    def bone_segmentation(self, bone_threshold = 330):
	#ipdb.set_trace()
        return np.array(self.data3d > bone_threshold)#.astype(np.int8)*self.slab['bone']

    def convolve_structure_spine(self, velikost = [300 , 2, 2]):
	structure = np.ones((int(velikost[2]/self.voxelsize_mm[0]),int(velikost[1]/self.voxelsize_mm[1]),int(velikost[2]/self.voxelsize_mm[2])))
	return structure

    

    def spine_segmentation(self):
	seg_prub = filters.gaussian_filter(self.data3d, 5.0/np.asarray(self.voxelsize_mm))
	seg_prub = filters.convolve(seg_prub, self.convolve_structure_spine())
	#seg_prub = scipy.signal.fftconvolve(seg_prub, self.convolve_structure_spine())
	maximum = np.amax(seg_prub)
	seg_prub = np.array(seg_prub > 0.4*maximum)
	#import sed3
	#ed = sed3.sed3(seg_prub)
	#ed.show()
	self.segmentation = seg_prub
        pass


    def __spl(self, x,y):
        return interpolate.bisplev(x,y, tck)

    def __above_diaphragm_calculation(self, seg_prub, internal_resize_shape=[50, 50, 50]):
        # seg_prub_tmp = misc.resize_to_shape(seg_prub, internal_resize_shape)
        seg_prub_tmp = seg_prub
        z, x, y= np.nonzero(seg_prub_tmp)
        s = np.array([x , y]).T
        h = np.array(z)
        model = mpf.multipolyfit(s, h, self.rad_diaphragm, model_out = True)
        # tck = interpolate.bisplrep(x,y,z, s=10)
        ran = self.segmentation.shape
        x = np.arange( ran[1] )
        y = np.arange( ran[2] )
        x, y = np.meshgrid( x, y)
        # sh = [len(x), len(y)]
##x, y = np.meshgrid(x, y)
        # z = np.asarray(map(self.__spl(x,y), x.reshape(-1), y.reshape(-1))).reshape(sh)


        z = np.floor(np.asarray(map(model, x.reshape(-1), y.reshape(-1)))).astype(int)
        x = x.reshape(z.shape)
        y = y.reshape(z.shape)
        cc = np.zeros(ran)
        for a in range(z.shape[0]):
            if (z[a] < ran[0]) and (z[a] >= 0):
                self.segmentation[z[a], x[a], y[a]] = self.slab['diaphragm']
                for b in range(z[a]+1, ran[0]):
                    cc[b, x[a], y[a]] = 1

        # cc = misc.resize_to_shape(cc, seg_prub.shape)
        return cc

    def heart_segmentation(self, heart_threshold = 0):
        a=self.convolve_structure_heart()
        seg_prub = np.array(self.segmentation == self.slab['rlung'])+np.array(self.segmentation == self.slab['llung'])
        seg_prub = filters.convolve( (seg_prub-0.5) , a )
# import sed3
# ed = sed3.sed3(seg_prub)
# ed.show()
        seg_prub = np.array(seg_prub<=-0.3)
        cc = self.__above_diaphragm_calculation(seg_prub)
#ipdb.set_trace()
        plice1=np.array(self.segmentation==self.slab['llung'])
        z, x, y = np.nonzero(plice1)
        xmin1=np.min(x)
        xmax1=np.max(x)
        ymin1=np.min(y)
        ymax1=np.max(y)
        z1=np.max(z)
        z2=np.min(z)
        plice2=np.array(self.segmentation==self.slab['rlung'])
        z, x, y = np.nonzero(plice2)
        xmin2=np.min(x)	
        xmax2=np.max(x)
        ymin2=np.min(y)
        ymax2=np.max(y)	
        mp=np.zeros(self.segmentation.shape)

        zd=(int) (z2+(np.floor((z1-z2)/3)))
        mp[zd:, ymin2:ymax1, xmin2:xmax1]=1
        bones = np.array(self.data3d>=200)
        aaa = np.array(self.data3d >= heart_threshold)
        aaa = aaa - bones
        aaa=morphology.binary_opening(aaa , iterations=self.iteration()+2).astype(self.segmentation.dtype)
        aaa = morphology.binary_erosion(aaa, iterations=self.iteration())	
        aaa=cc * aaa * mp
        lab , num = label(aaa)
        counts= [0]*(num+1)
        for x in range(1, num+1):
            a = np.sum(np.array(lab == x))
            counts[x] = a
        index= np.argmax(counts)
        aaa = np.array(lab==index)
        aaa = morphology.binary_dilation(aaa, iterations=self.iteration())
        self.segmentation= aaa
	#self.segmentation = self.segmentation + aaa
        pass

    def iteration(self, sirka = 5):
	prumer= np.mean(self.voxelsize_mm)
	a= int (sirka/prumer)
	return a

    def orientation(self):
	if(self.segmentation.shape[0]%2==0):
	    split1, split2 = np.split(self.segmentation, 2, 0)
	    if(np.sum(split1)>np.sum(split2)):
		self.smer=1

	    else:
		self.smer=0

	
	else:
	    split1, split2 = np.split(self.segmentation[1:,:,:], 2, 0)
	    if(np.sum(split1)>np.sum(split2)):
		self.smer=1

	    else:
		self.smer=0



	pass

    def lungs_segmentation(self, lungs_threshold = -360):
	seg_prub = np.array(self.data3d <= lungs_threshold)
	seg_prub = morphology.binary_closing(seg_prub , iterations=self.iteration()).astype(self.segmentation.dtype)
	seg_prub = morphology.binary_opening(seg_prub , iterations = 5)
	labeled_seg , num_seg = label(seg_prub)
	counts= [0]*(num_seg+1)
	for x in range(1, num_seg+1):
	    a = np.sum(np.array(labeled_seg == x))
	    counts[x] = a

	#self.segmentation = seg_prub
	#for x in np.nditer(labeled_seg, op_flags=['readwrite']):
	#    if x[...]!=0:
	#    	counts[x[...]]=counts[x[...]]+1
	z, x, y = labeled_seg.shape
	index=labeled_seg[self.iteration()+5,self.iteration()+5,self.iteration()+5]
	counts[index]=0
	index=labeled_seg[self.iteration()+5,self.iteration()+5,y-self.iteration()-5]
	counts[index]=0
	index=labeled_seg[self.iteration()+5,x-self.iteration()-5,self.iteration()+5]
	counts[index]=0
	index=labeled_seg[self.iteration()+5,x-self.iteration()-5,y-self.iteration()-5]
	counts[index]=0
	#index=np.argmax(counts) #pozadí
	#counts[index]=0
	index=np.argmax(counts) #jedna nebo obě plíce
	velikost1=counts[index]
	counts[index]=0
	index2=np.argmax(counts)# druhá plíce nebo nečo jiného
	velikost2=counts[index2]
	if ((velikost2 / velikost1) > (1-self.maximal_lung_diff)) | ((velikost2/velikost1) < (1 + self.maximal_lung_diff)):
	    print("plice separované")
	else:
	    pocet=0
	    while ((velikost2 / velikost1) < (1 - self.maximal_lung_diff)) | ((velikost2/velikost1)> (1 + self.maximal_lung_diff)):
		morphology.binary_erosion(self.segmentation,iterations=1).astype(self.segmentation.dtype)
		labeled_seg , num_seg = label(seg_prub)
		counts= [0] * (num_seg+1)
		for x in range(1, num_seg+1):
	    	    a = np.sum(np.array(labeled_seg == x))
	    	    counts[x] = a

	    	index=np.argmax(counts) #pozadí
	    	counts[index]=0
	    	index=np.argmax(counts) #jedna nebo obě plíce
	    	velikost1=counts[index]
	    	counts[index]=0
	    	index2=np.argmax(counts)# druhá plíce nebo nečo jiného
	    	velikost2=counts[index2]
		pocet=pocet+1

	    morphology.binary_dilation(self.segmentation,iterations=pocet).astype(self.segmentation.dtype)
	#self.segmentation = self.segmentation + np.array(labeled_seg==index).astype(np.int8)*self.slab['lungs']
	#self.segmentation = self.segmentation + np.array(labeled_seg==index2).astype(np.int8)*self.slab['lungs']
	plice1 = np.array(labeled_seg==index)
	z,x,y = np.nonzero(plice1)
	m1 = np.max(y)
	if m1<(self.segmentation.shape[1]/2):
	    self.segmentation = self.segmentation + np.array(labeled_seg==index).astype(np.int8)*self.slab['llung']
	    self.segmentation = self.segmentation + np.array(labeled_seg==index2).astype(np.int8)*self.slab['rlung']
	else:
	    self.segmentation = self.segmentation + np.array(labeled_seg==index).astype(np.int8)*self.slab['rlung']
	    self.segmentation = self.segmentation + np.array(labeled_seg==index2).astype(np.int8)*self.slab['llung']
	self.orientation()
	if self.smer==1:#netestovano
	    self.segmentation[self.segmentation==self.slab['llung']]=3
	    self.segmentation[self.segmentation==self.slab['rlung']]=self.slab['llung']
	    self.segmentation[self.segmentation==3]=self.slab['rlung']

        pass
	


    def _crop(self, data, crinfo):
        """
        Crop data with crinfo
        """
        data = data[crinfo[0][0]:crinfo[0][1], crinfo[1][0]:crinfo[1][1], crinfo[2][0]:crinfo[2][1]]
        return data


    def _crinfo_from_specific_data (self, data, margin):
# hledáme automatický ořez, nonzero dá indexy
        nzi = np.nonzero(data)

        x1 = np.min(nzi[0]) - margin[0]
        x2 = np.max(nzi[0]) + margin[0] + 1
        y1 = np.min(nzi[1]) - margin[0]
        y2 = np.max(nzi[1]) + margin[0] + 1
        z1 = np.min(nzi[2]) - margin[0]
        z2 = np.max(nzi[2]) + margin[0] + 1

# ošetření mezí polí
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if z1 < 0:
            z1 = 0

        if x2 > data.shape[0]:
            x2 = data.shape[0]-1
        if y2 > data.shape[1]:
            y2 = data.shape[1]-1
        if z2 > data.shape[2]:
            z2 = data.shape[2]-1

# ořez
        crinfo = [[x1, x2],[y1,y2],[z1,z2]]
        #dataout = self._crop(data,crinfo)
        #dataout = data[x1:x2, y1:y2, z1:z2]
        return crinfo


    def im_crop(self, im,  roi_start, roi_stop):
        im_out = im[ \
                roi_start[0]:roi_stop[0],\
                roi_start[1]:roi_stop[1],\
                roi_start[2]:roi_stop[2],\
                ]
        return  im_out

    def export(self):
        slab={}
        slab['none'] = 0
        slab['liver'] = 1
        slab['lesions'] = 6
	slab['lungs'] = 9
        data = {}
        data['version'] = (1,0,0)
        data['data3d'] = self.data3d
        data['crinfo'] = self.crinfo
        data['segmentation'] = self.segmentation
        data['slab'] = slab
        data['voxelsize_mm'] = self.voxelsize_mm
        #import pdb; pdb.set_trace()
        return data




    def visualization(self):
        """
        Run viewer with output data3d and segmentation
        """

        try:
            from pysegbase.seed_editor_qt import QTSeedEditor
        except:
            logger.warning("Deprecated of pyseg_base as submodule")
            from seed_editor_qt import QTSeedEditor
        from PyQt4.QtGui import QApplication
        import numpy as np
#, QMainWindow
        app = QApplication(sys.argv)
        #pyed = QTSeedEditor(self.data3d, contours=(self.segmentation>0))
        pyed = QTSeedEditor(self.segmentation)
        pyed.exec_()


        #import pdb; pdb.set_trace()


        #pyed = QTSeedEditor(deletemask, mode='draw')
        #pyed.exec_()

        app.exit()






def main():

    #logger = logging.getLogger(__name__)
    logger = logging.getLogger()

    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    # input parser
    parser = argparse.ArgumentParser(description=
            'Segmentation of bones, lungs and heart.')
    parser.add_argument('-i','--datadir',
            default=None,
            help='path to data dir')
    parser.add_argument('-o','--output',
            default=None,
            help='output file')

    parser.add_argument('-d', '--debug', action='store_true',
            help='run in debug mode')
    parser.add_argument('-ss', '--segspine', action='store_true',
            help='run spine segmentaiton')
    parser.add_argument('-sl', '--seglungs', action='store_true',
            help='run lungs segmentation')
    parser.add_argument('-sh', '--segheart', action='store_true',
            help='run heart segmentation')
    parser.add_argument('-sb', '--segbones', action='store_true',
            help='run bones segmentation')
    parser.add_argument('-exd', '--exampledata', action='store_true',
            help='run unittest')
    parser.add_argument('-so', '--show_output', action='store_true',
            help='Show output data in viewer')
    args = parser.parse_args()



    if args.debug:
        logger.setLevel(logging.DEBUG)


    if args.exampledata:

        args.dcmdir = '../sample_data/liver-orig001.raw'

#    if dcmdir == None:

    #else:
    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
    #data3d, metadata = dcmr.dcm_read_from_dir(args.dcmdir)

    data3d , metadata = io3d.datareader.read(args.datadir)

    sseg = SupportStructureSegmentation(data3d = data3d,
            voxelsize_mm = metadata['voxelsize_mm'],
            )


    #sseg.orientation()
    if args.segbones:
	sseg.bone_segmentation()
    if args.segspine:
        sseg.spine_segmentation()
    if args.seglungs or args.segheart:
    	sseg.lungs_segmentation()
    if args.segheart:
    	sseg.heart_segmentation()

    #print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )

    #igc = pycut.ImageGraphCut(data3d, zoom = 0.5)
    #igc.interactivity()


    #igc.make_gc()
    #igc.show_segmentation()

    # volume
    #volume_mm3 = np.sum(oseg.segmentation > 0) * np.prod(oseg.voxelsize_mm)


    #pyed = sed3.sed3(oseg.data3d, contour = oseg.segmentation)
    #pyed.show()

    if args.show_output:
        sseg.visualization()

    # savestring = raw_input ('Save output data? (y/n): ')
    #sn = int(snstring)
    if args.output is not None: # savestring in ['Y','y']:
        import misc

        data = sseg.export()

        misc.obj_to_file(data, args.output, filetype = 'pickle')
    #output = segmentation.vesselSegmentation(oseg.data3d, oseg.orig_segmentation)


if __name__ == "__main__":
    main()
