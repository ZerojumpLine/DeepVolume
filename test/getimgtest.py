import scipy.io as sio
import numpy as np

def getimgtest(val_axialpath, val_sagittalpath, kv):

    val_axialfile = open(val_axialpath)
    val_axialread = val_axialfile.read().splitlines()
    val_sagittalfile = open(val_sagittalpath)
    val_sagittalread = val_sagittalfile.read().splitlines()

    val_axialname = val_axialread[kv]
    val_axialimgread = sio.loadmat(val_axialname)
    val_axialimg = val_axialimgread['T1_image']

    val_sagittalname = val_sagittalread[kv]
    val_sagittalimgread = sio.loadmat(val_sagittalname)
    val_sagittalimg = val_sagittalimgread['T2_image']


    LRimg = np.zeros((1, val_axialimg.shape[0], val_axialimg.shape[1], val_axialimg.shape[2], 2))

    LRimg[0, :, :, :, 0] = val_axialimg
    LRimg[0, :, :, :, 1] = val_sagittalimg


    return LRimg
