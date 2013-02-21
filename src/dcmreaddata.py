#! /usr/bin/python
# -*- coding: utf-8 -*-



# import funkcí z jiného adresáře
import sys
import os
#sys.path.append("../src/")
#import featurevector

import logging
logger = logging.getLogger(__name__)


import pdb; 
#  pdb.set_trace();
#import scipy.io
#mat = scipy.io.loadmat('step0.mat')

#print mat
import Tkinter, tkFileDialog

import dicom

import numpy as np


def dcm_read_from_dir(dirpath=None, initialdir = os.path.expanduser("~")):
    
    if dirpath == None:
        root = Tkinter.Tk()
        dirpath = tkFileDialog.askdirectory(parent=root, initialdir = initialdir, title = "Please select a directory")
        root.destroy()
    pass
    
    dcmdir = getdicomdir(dirpath)
    #counts, bins = dcmdirstats(dcmdir)
    series_info = dcmdirstats(dcmdir)
    strcounts = ""
    if len (series_info) > 1:
        for serie_number in series_info.keys():
            strl = str(serie_number) + " (" + str(series_info[serie_number]['Count']) 
            try:
                strl = strl + ", " + str( series_info[serie_number]['Modality'])
                strl = strl + ", " + str( series_info[serie_number]['ImageComment'])
            except:
                logger.debug('Tag Modlity or ImageComment not found in dcminfo')
                pass

            strl = strl + ')'
            print strl
            #strcounts = strcounts + str(bins[i]) + "(" + str(counts[i]) + ") "
        #print counts
        #print bins
        #logger.info(counts)
        #logger.info(bins)
        #print strcounts

        snstring = raw_input ('Select Serie: ')
        sn = int(snstring)
    else:
        sn = series_info.keys()[0]

    #dcmdir = sort_dcmdir(dcmdir, SeriesNumber = sn)
# Now we need list of files with specific SeriesNumer
# dcmdir has just filenamse (no full path), 
    dcmlist = dcmsortedlist(dirpath, dcmdir = dcmdir, SeriesNumber = sn)

    data3d, metadata = dcmlist_to_3D_data(dcmlist)
    return data3d, metadata
    

def dcmlist_to_3D_data(dcmlist):
    """
    Function make 3D data from dicom file slices
    """
    data3d = []
    #for onefile in dcmlist:
    for i  in range(len(dcmlist)):
        onefile = dcmlist[i]
        #filelist.append(os.path.join(startpath, dirpath,onefile['filename']))
        #head, tail = os.path.split(onefile['filename'])
        logger.info(onefile)#['filepath']
        data = dicom.read_file(onefile)#['filepath'])
        data2d = data.pixel_array
        #pdb.set_trace();
        if len(data3d) == 0:
            shp2 = data2d.shape
            data3d = np.zeros([shp2[0],shp2[1], len(dcmlist)], dtype=np.int16)
            #data3d = np.zeros([shp2[0],shp2[1], len(dcmlist)])
            # data3d = data2d[:,:,np.newaxis]
            metadata = get_metadata(data)
        else:
            #data3d = np.concatenate((data3d,data2d[...,np.newaxis]), axis = 2)
            data3d [:,:,i] = data2d

    logger.debug("Data size: " + str(data3d.nbytes) + ', shape: ' + str(shp2) +'x'+ str(len(dcmlist)) )
    #logger.debug(data3d.nbytes)

    #pdb.set_trace();
    return data3d, metadata



def get_metadata(data):
    #import pdb; pdb.set_trace()
    pixelsizemm = data.PixelSpacing
    voxelsizemm = [pixelsizemm[0], pixelsizemm[1], data.SliceThickness]

    metadata = {'voxelsizemm':voxelsizemm, 'Modality':data.Modality}
    return metadata

def obj_from_file(filename = 'annotation.yaml', filetype = 'yaml'):
    ''' Read object from file '''
# TODO solution for file extensions
    f = open(filename, 'r')
    if filetype == 'yaml':
        import yaml
        obj = yaml.load(f)
    elif filetype == 'pickle':
        import pickle
        obj = pickle.load(f)
    else:
        logger.error('Unknown filetype')
    f.close()
    return obj


def obj_to_file(obj, filename = 'annotation.yaml', filetype = 'yaml'):
    '''Writes annotation in file
    '''
    #import json
    #with open(filename, mode='w') as f:
    #    json.dump(annotation,f)

    # write to yaml

    f = open(filename, 'w')
    if filetype == 'yaml':
        import yaml
        yaml.dump(obj,f)
    elif filetype == 'pickle':
        import pickle
        pickle.dump(obj,f)
    else:
        logger.error('Unknown filetype')
    f.close

def filesindir(dirpath, wildcard="*", startpath=None):
    """ Function generates list of files from specific dir

    filesindir(dirpath, wildcard="*.*", startpath=None)

    dirpath: required directory
    wilcard: mask for files
    startpath: start for relative path

    Example
    filesindir('medical/jatra-kiv','*.dcm', '/home/mjirik/data/')


    """
    import os
    import glob


    filelist = []
    #print dirpath

    if startpath != None:
        completedirpath = os.path.join( startpath, dirpath)
    else:
        completedirpath = dirpath

    if os.path.exists(completedirpath):
        logger.info('completedirpath = '  + completedirpath )
        #print completedirpath
    else:
        logger.error('Wrong path: '  + completedirpath )
        raise Exception('Wrong path : ' + completedirpath )

    #print 'copmpletedirpath = ', completedirpath

    for infile in glob.glob( os.path.join(completedirpath, wildcard) ):
        filelist.append(infile)
        #print "current file is: " + infile


    if len(filelist) == 0:
        logger.error('No required files in  path: '  + completedirpath )
        raise Exception ('No required file in path: ' + completedirpath )
    return filelist

def getdicomdir(dirpath, writedicomdirfile = True, forcecreate = False):
    ''' Function check if exists dicomdir file and load it or cerate it

    dcmdir = getdicomdir(dirpath)

    dcmdir: dcmdir = {'version':version_number, 'filesinfo',filesinfo}
    filesinfo: list with filenames, SeriesNumber, InstanceNumber and 

    AcquisitionNumber
    '''
    
    createdcmdir = True

    # if exist dicomdir file and is in correct version, use it
    dcmdiryamlpath = os.path.join( dirpath, 'dicomdir.yaml')
    if os.path.exists(dcmdiryamlpath):
        dcmdir = obj_from_file(dcmdiryamlpath)
        try:
            if dcmdir ['version'] == [1,0]:
                createdcmdir = False
        except:
            logger.debug('Found dicomdir.yaml with wrong version')
            pass


    if createdcmdir:
        dcmdir = createdicomdir(dirpath)
        if (writedicomdirfile):
            obj_to_file(dcmdir, dcmdiryamlpath )

    dcmdir = dcmdir['filesinfo']
    return dcmdir

def createdicomdir(dirpath):
    """Function crates list of all files in dicom dir with all IDs
    """
    import dicom

    # TODO přidat informaci o vezri a při kolizi, generovat znova
    filelist = filesindir(dirpath)
    files=[]

    ## doplneni o cestu k datovemu adresari
    #if startpath != None:
    #    completedirpath = os.path.join( startpath, dirpath)
    #else:
    #    completedirpath = dirpath

    # pruchod soubory
    for filepath in filelist:
        fullfilepath = filepath
        head, teil = os.path.split(fullfilepath)
        try:
            dcmdata=dicom.read_file(fullfilepath)
            metadataline = {'filename' : teil, 
                #copy.copy(dcmdata.FrameofReferenceUID), 
                #copy.copy(dcmdata.StudyInstanceUID),
                #copy.copy(dcmdata.SeriesInstanceUID) 
                'InstanceNumber' : dcmdata.InstanceNumber,
                'SeriesNumber' : dcmdata.SeriesNumber,
                'AcquisitionNumber' : dcmdata.AcquisitionNumber
                }
            try:
                metadataline ['ImageComment'] = dcmdata.ImageComments
                metadataline ['Modality'] = dcmdata.Modality
            except:
                print 'Problem with ImageComments and Modality tags'
                #import pdb; pdb.set_trace()
                pass
            

            files.append(metadataline)
            #logger.debug( \
            #    'FrameUID : ' + str(dcmdata.InstanceNumber) + \
            #    ' ' + str(dcmdata.SeriesNumber) + \
            #    ' ' + str(dcmdata.AcquisitionNumber)\
            #    )
            #import pdb; pdb.set_trace()
        except Exception as e:
            print 'Dicom read problem with file ' + fullfilepath
            print e

        # dcmdata.InstanceNumber
        #logger.info('Modality: ' + dcmdata.Modality)
        #logger.info('PatientsName: ' + dcmdata.PatientsName)
        #logger.info('BodyPartExamined: '+ dcmdata.BodyPartExamined)
        #logger.info('SliceThickness: '+ str(dcmdata.SliceThickness))
        #logger.info('PixelSpacing: '+ str(dcmdata.PixelSpacing))
        # get data
        #data = dcmdata.pixel_array

    # a řadíme podle frame 

    files.sort(key=lambda x: x['InstanceNumber'])
    files.sort(key=lambda x: x['SeriesNumber'])
    files.sort(key=lambda x: x['AcquisitionNumber'])


    dicomdir = {'version':[1,0],'filesinfo':files}
    return dicomdir 

    #files.sort(key=operator.itemgetter(1))
    #files.sort(key=operator.itemgetter(2))
    #files.sort(key=operator.itemgetter(3))




def dcmdirstats(dcmdir):
    """ input is dcmdir, not dirpath """
    import numpy as np
    # get series number
# vytvoření slovníku, kde je klíčem číslo série a hodnotou jsou všechny 
# informace z dicomdir
    series_info = {line['SeriesNumber']:line for line in dcmdir}

# počítání velikosti série
    try:
        dcmdirseries = [line['SeriesNumber'] for line in dcmdir ]
    except:
        logger.debug('Dicom tag SeriesNumber not found')
        series_info = {0:{'Count':0}}
        return series_info
        #return [0],[0]

    bins = np.unique(dcmdirseries)
    binslist = bins.tolist()
#  kvůli správným intervalům mezi biny je nutno jeden přidat na konce
    mxb = np.max(bins)+1
    binslist.append(mxb)
    #binslist.insert(0,-1)
    counts, binsvyhodit = np.histogram(dcmdirseries, bins = binslist)

    #pdb.set_trace();

    # sestavení informace o velikosti série a slovníku 

    for i in range(0,len(bins)):
        series_info[bins[i]]['Count']=counts[i]

    return series_info

    #return counts, bins

def dcmsortedlist(dirpath=None, wildcard='*', startpath="", 
        dcmdir=None, writedicomdirfile = True , SeriesNumber = None):
    """ Function returns sorted list of dicom files. File paths are organized by
    SeriesUID, StudyUID and FrameUID

    Example:
    dcmsortedlist ('/home/mjirik/data/medical/jatra_5mm','*.dcm')
    dcmsortedlist ('medical/jatra_5mm','*.dcm','/home/mjirik/data/')

    or you can give dcmdir for sorting

    dcmsortedlist(dcmdir)

    """
    #import dicom
    #import operator
    #import copy
    #pdb.set_trace();
    if dcmdir == None:
        if dirpath != None:
# TODO doplnit prevod dcmdir na filelist

            #pdb.set_trace();
            dcmdir = getdicomdir(os.path.join(startpath,dirpath), 
                    writedicomdirfile)

            #filelist = filesindir(dirpath, wildcard, startpath)
        else:
            logger.error('Wrong input params')
    #else:
    #    logger.error('Deprecated call with filellist')

#    files=[]
#
#    ## doplneni o cestu k datovemu adresari
#    #if startpath != None:
#    #    completedirpath = os.path.join( startpath, dirpath)
#    #else:
#    #    completedirpath = dirpath
#
#    # pruchod soubory
#    for filepath in filelist:
#        fullfilepath = os.path.join(startpath, filepath)
#
#        try:
#            dcmdata=dicom.read_file(fullfilepath)
#            files.append([fullfilepath, 
#                #copy.copy(dcmdata.FrameofReferenceUID), 
#                #copy.copy(dcmdata.StudyInstanceUID),
#                #copy.copy(dcmdata.SeriesInstanceUID) 
#                dcmdata.InstanceNumber,
#                dcmdata.SeriesNumber,
#                dcmdata.AcquisitionNumber
#                ])
#            logger.debug( \
#                'FrameUID : ' + str(dcmdata.InstanceNumber) + \
#                ' ' + str(dcmdata.SeriesNumber) + \
#                ' ' + str(dcmdata.AcquisitionNumber)\
#                )
#        except Exception as e:
#            print 'Dicom read problem with file ' + fullfilepath
#            print e
#
#        # dcmdata.InstanceNumber
#        #logger.info('Modality: ' + dcmdata.Modality)
#        #logger.info('PatientsName: ' + dcmdata.PatientsName)
#        #logger.info('BodyPartExamined: '+ dcmdata.BodyPartExamined)
#        #logger.info('SliceThickness: '+ str(dcmdata.SliceThickness))
#        #logger.info('PixelSpacing: '+ str(dcmdata.PixelSpacing))
#        # get data
#        #data = dcmdata.pixel_array
#
#    # a řadíme podle frame 
#    files.sort(key=operator.itemgetter(1))
#    files.sort(key=operator.itemgetter(2))
#    files.sort(key=operator.itemgetter(3))
#
#    # TODO dopsat řazení
#    #filelist.sort(lambda:files)
#    dcmdirfile = []

    dcmdir = sort_dcmdir(dcmdir, SeriesNumber)

    logger.debug('SeriesNumber: ' +str(SeriesNumber))

    filelist = []
    #pdb.set_trace();
    for onefile in dcmdir:
        filelist.append(os.path.join(startpath, dirpath,onefile['filename']))
        head, tail = os.path.split(onefile['filename'])
#
#        dcmdirfile.append({'filename': tail,'InstanceNumber': onefile[1],
#            'SeriesNumber':onefile[2], 'AcquisitionNumber':onefile[3] })



    #pdb.set_trace();
    return filelist


def sort_dcmdir(dcmdir, SeriesNumber = None):
    """ 
    Returns sorted dcmdir. You can extract specific serie.
    """
    # sort (again) just for sure
    dcmdir.sort(key=lambda x: x['InstanceNumber'])
    dcmdir.sort(key=lambda x: x['SeriesNumber'])
    dcmdir.sort(key=lambda x: x['AcquisitionNumber'])

    # select sublist with SeriesNumber
    #SeriesNumber = 5
    if SeriesNumber != None:
        dcmdir = [line for line in dcmdir if line['SeriesNumber']==SeriesNumber]

    return dcmdir


if __name__ == "__main__":

    #logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    #logger.debug('input params')

    #dcm_read_from_dir('/home/mjirik/data/medical/data_orig/46328096/')
    data3d = dcm_read_from_dir()

    print ("Data size: " + str(data3d.nbytes) + ', shape: ' + str(data3d.shape) )
   # for arg in sys.argv:
   #     logger.debug(''+arg)

   # databasedir = '/home/mjirik/data'
   # if len(sys.argv) < 1:
   #     datatraindir = '/home/mjirik/data/jatra_06mm_jenjatra'
   # else:
   #     datatraindir = sys.argv[1]

   # logger.debug('Adresar ' + datatraindir)
    #dm = Dialogmenu()
    #print dm.retval

