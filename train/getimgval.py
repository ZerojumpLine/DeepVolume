import scipy.io as sio
import numpy as np

def getimgval(val_3Dpath, val_axialpath, val_sagittalpath, kv):

    val_3Dfile = open(val_3Dpath)
    val_3Dread = val_3Dfile.read().splitlines()
    val_axialfile = open(val_axialpath)
    val_axialread = val_axialfile.read().splitlines()
    val_sagittalfile = open(val_sagittalpath)
    val_sagittalread = val_sagittalfile.read().splitlines()

    val_3Dname = val_3Dread[kv]
    val_3Dimgread = sio.loadmat(val_3Dname)
    val_3Dimg = val_3Dimgread['T3_image']

    val_axialname = val_axialread[kv]
    val_axialimgread = sio.loadmat(val_axialname)
    val_axialimg = val_axialimgread['T1_image']

    val_sagittalname = val_sagittalread[kv]
    val_sagittalimgread = sio.loadmat(val_sagittalname)
    val_sagittalimg = val_sagittalimgread['T2_image']


    LRimg = np.zeros((1, val_3Dimg.shape[0], val_3Dimg.shape[1], val_3Dimg.shape[2], 2))
    HRimg = np.zeros((1, val_3Dimg.shape[0], val_3Dimg.shape[1], val_3Dimg.shape[2], 1))

    LRimg[0, :, :, :, 0] = val_axialimg
    LRimg[0, :, :, :, 1] = val_sagittalimg
    HRimg[0, :, :, :, 0] = val_3Dimg


    return LRimg, HRimg