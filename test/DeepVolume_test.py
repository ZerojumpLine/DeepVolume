import argparse
import os
import scipy.io as sio
import numpy as np
import nibabel as nib
import time
import tensorflow as tf
from BrainStructureAwareNetwork_arch import BrainStructureAwareNetwork
from SpatialConnectionAwareNetwork_arch import SpatialConnectionAwareNetwork
from getimgtest import getimgtest
import math

batch_size = 1
# to recover the origin distribution
Intensity_max = 255
Intensity_mean = 0.1616
Intensity_std = 0.2197

parser = argparse.ArgumentParser(description='Tensorflow DeepVolume Test')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--datafile', default='../datafile/', type=str, help='path to datafile folder')
parser.add_argument('--savepath', default='../output/', type=str, help='path to output folder')
parser.add_argument('--modelpath', default='../models/', type=str, help='path to model save folder')
parser.add_argument('-s', '--stage', type=int, default=1, help='load the network one by one...')
args = parser.parse_args()

def test_BrainStructureNetwork():
    filespath = args.datafile
    axialThickpath = filespath + 'axialThick-test.txt'
    sagittalThickpath = filespath + 'sagittalThick-test.txt'
    savepath = args.savepath + 'test'
    modelpath = args.modelpath + 'BrainStructureAwareNetwork/Model100.ckpt'

    val_axialfile = open(axialThickpath)
    val_axialread = val_axialfile.read().splitlines()
    ntest = len(val_axialread)

    testsavepath = savepath + str(1)
    if (os.path.isdir(testsavepath) == False) :
        for k in range(0, ntest):
            os.mkdir(savepath + str(k + 1))
        print("Folder created")


    with tf.name_scope('input'):
        LR = tf.placeholder(tf.float32, shape=[batch_size, 200, 200, 200, 2])
        keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')

    probs, logits = BrainStructureAwareNetwork(LR, keep_prob)

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(args.gpu)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, modelpath)
    print("Model loaded")

    def feed_dict(xstart, ystart, zstart, LRimg):
        xs = np.zeros((1, 200, 200, 200, 2))
        xs[:, :, :, :, :] = LRimg[:,xstart:xstart + 200, ystart:ystart + 200, zstart:zstart + 200, :]
        return {LR: xs, keep_prob: 1}

    # sample patches with tumor
    for kv in range(0, ntest):
        time_start = time.time()
        print('Loading from test case ' + str(kv + 1) + ' for test for stage 1')

        LRimg = getimgtest(axialThickpath, sagittalThickpath, kv)

        x_range = LRimg.shape[1]
        y_range = LRimg.shape[2]
        z_range = LRimg.shape[3]

        if z_range < 200:
            LRimgpad = np.zeros((1, x_range, y_range, 200, 2))
            LRimgpad[:,:,:,0:z_range,:] = LRimg
            LRimg = LRimgpad

        # The receptive field of 3D U-net is 68
        # We should retrive the center 40^3 pixel of 200^3 to reconstruct

        if z_range < 200:
            hp = np.zeros((x_range, y_range, 200, 1))
        else:
            hp = np.zeros((x_range, y_range, z_range, 1))
        x_sample = np.floor((x_range -160) / 40) + 1
        x_sample = x_sample.astype(np.int16)
        y_sample = np.floor((y_range -160) / 40) + 1
        y_sample = y_sample.astype(np.int16)
        z_sample = np.maximum(np.floor((z_range -160) / 40) + 1, 1)
        z_sample = z_sample.astype(np.int16)

        for jx in range(0, x_sample):
            for jy in range(0, y_sample):
                for jz in range(0, z_sample):

                    # deal with the boundaries
                    if jx < x_sample - 1:  # not the last
                        xstart = jx * 40
                    else:
                        xstart = x_range - 200

                    if jy < y_sample - 1:  # not the last
                        ystart = jy * 40
                    else:
                        ystart = y_range - 200

                    if jz < z_sample - 1:  # not the last
                        zstart = jz * 40
                    else:
                        zstart = LRimg.shape[3] - 200

                    ht = sess.run(probs, feed_dict=feed_dict(xstart, ystart, zstart, LRimg))
                    # setting the middle content
                    hp[xstart + 80:xstart + 120, ystart + 80:ystart + 120, zstart + 80:zstart + 120, :] = ht[0, 80:120, 80:120, 80:120, :]
                    # care about the boundies! the patch near the boundies should have half-full padding
                    if jx == 0:
                        hp[xstart:xstart + 120, ystart + 80:ystart + 120, zstart + 80:zstart + 120, :] = ht[0, 0:120, 80:120, 80:120, :]
                    if jx == x_sample - 1:
                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 120, zstart + 80:zstart + 120, :] = ht[0, 80:200, 80:120, 80:120, :]
                    if jy == 0:
                        hp[xstart + 80:xstart + 120, ystart:ystart + 120, zstart + 80:zstart + 120, :] = ht[0, 80:120, 0:120, 80:120, :]
                    if jy == y_sample - 1:
                        hp[xstart + 80:xstart + 120, ystart + 80:ystart + 200, zstart + 80:zstart + 120, :] = ht[0, 80:120, 80:200, 80:120, :]
                    if jz == 0:
                        hp[xstart + 80:xstart + 120, ystart + 80:ystart + 120, zstart:zstart + 120, :] = ht[0, 80:120, 80:120, 0:120, :]
                    if jz == z_sample - 1:
                        hp[xstart + 80:xstart + 120, ystart + 80:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 80:120, 80:120, 80:200, :]
                    # then the 4 corner...xy
                    if jx == 0 and jy == 0:
                        hp[xstart:xstart + 120, ystart:ystart + 120, zstart + 80:zstart + 120, :] = ht[0, 0:120, 0:120, 80:120, :]
                    if jx == 0 and jy == y_sample - 1:
                        hp[xstart:xstart + 120, ystart + 80:ystart + 200, zstart + 80:zstart + 120, :] = ht[0, 0:120, 80:200, 80:120, :]
                    if jx == x_sample - 1 and jy == 0:
                        hp[xstart + 80:xstart + 200, ystart:ystart + 120, zstart + 80:zstart + 120, :] = ht[0, 80:200, 0:120, 80:120, :]
                    if jx == x_sample - 1 and jy == y_sample - 1:
                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 200, zstart + 80:zstart + 120, :] = ht[0, 80:200, 80:200, 80:120, :]
                    # then the 4 corner...xz
                    if jx == 0 and jz == 0:
                        hp[xstart:xstart + 120, ystart+80:ystart + 120, zstart:zstart + 120, :] = ht[0, 0:120, 80:120, 0:120, :]
                    if jx == 0 and jz == z_sample - 1:
                        hp[xstart:xstart + 120, ystart + 80:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 0:120, 80:120, 80:200, :]
                    if jx == x_sample - 1 and jz == 0:
                        hp[xstart + 80:xstart + 200, ystart+80:ystart + 120, zstart:zstart + 120, :] = ht[0, 80:200, 80:120, 0:120, :]
                    if jx == x_sample - 1 and jz == z_sample - 1:
                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 80:200, 80:120, 80:200, :]
                    # then the 4 corner...yz
                    if jy == 0 and jz == 0:
                        hp[xstart+80:xstart + 120, ystart:ystart + 120, zstart:zstart + 120, :] = ht[0, 80:120, 0:120, 0:120, :]
                    if jy == 0 and jz == z_sample - 1:
                        hp[xstart+80:xstart + 120, ystart:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 80:120, 0:120, 80:200, :]
                    if jy == y_sample - 1 and jz == 0:
                        hp[xstart + 80:xstart + 120, ystart+80:ystart + 200, zstart:zstart + 120, :] = ht[0, 80:120, 80:200, 0:120, :]
                    if jy == y_sample - 1 and jz == z_sample - 1:
                        hp[xstart + 80:xstart + 120, ystart + 80:ystart + 200, zstart + 80:zstart + 200, :] = ht[0, 80:120, 80:200, 80:200, :]
                    # the last 8 small corners..
                    if jx == 0 and jy == 0 and jz == 0:
                        hp[xstart:xstart + 120, ystart:ystart + 120, zstart:zstart + 120, :] = ht[0, 0:120, 0:120, 0:120, :]
                    if jx == 0 and jy == 0 and jz == z_sample - 1:
                        hp[xstart:xstart + 120, ystart:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 0:120, 0:120, 80:200, :]
                    if jx == 0 and jy == y_sample - 1 and jz == 0:
                        hp[xstart:xstart + 120, ystart + 80:ystart + 200, zstart:zstart + 120, :] = ht[0, 0:120, 80:200, 0:120, :]
                    if jx == 0 and jy == y_sample - 1 and jz == z_sample - 1:
                        hp[xstart:xstart + 120, ystart + 80:ystart + 200, zstart + 80:zstart + 200, :] = ht[0, 0:120, 80:200, 80:200, :]
                    if jx == x_sample - 1 and jy == 0 and jz == 0:
                        hp[xstart + 80:xstart + 200, ystart:ystart + 120, zstart:zstart + 120, :] = ht[0, 80:200, 0:120, 0:120, :]
                    if jx == x_sample - 1 and jy == 0 and jz == z_sample - 1:
                        hp[xstart + 80:xstart + 200, ystart:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 80:200, 0:120, 80:200, :]
                    if jx == x_sample - 1 and jy == y_sample - 1 and jz == 0:
                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 200, zstart:zstart + 120, :] = ht[0, 80:200, 80:200, 0:120, :]
                    if jx == x_sample - 1 and jy == y_sample - 1 and jz == z_sample - 1:
                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 200, zstart + 80:zstart + 200, :] = ht[0, 80:200, 80:200, 80:200, :]

            print('processing Brain Structure Aware Model.. ' + str(jx/x_sample*100) + '%')

        if z_range < 200:
            hp = hp[:,:,0:z_range]

        print('processing Brain Structure Aware Model.. ' + '100%')

        time_end = time.time()
        print('Time cost of test at case ' + str(kv + 1) + '  for stage 1 has been ' + str(time_end - time_start) + ' s')
        savename = '%s%s%s' % (savepath, str(kv + 1), '/Reconstruction_BrainStructureAwareModel.mat')
        sio.savemat(savename, {'Reconstruction': hp})
        print('################ case ' + str(kv + 1) + ' has been done for Brain Structure Aware Model ################')
    sess.close()


def test_SpatialConnectionAwareNetwork():
    seq_length = 2
    network_template = tf.make_template('network', SpatialConnectionAwareNetwork)

    filespath = args.datafile
    axialThickpath = filespath + 'axialThick-test.txt'
    sagittalThickspath = filespath + 'sagittalThicks-test.txt'
    savepath = args.savepath + 'test'
    modelpath = args.modelpath + 'SpatialConnectionAwareNetwork/Model40.ckpt'

    val_axialfile = open(axialThickpath)
    val_axialread = val_axialfile.read().splitlines()
    val_sagittalfile = open(sagittalThickspath)
    val_sagittalread = val_sagittalfile.read().splitlines()
    ntest = len(val_axialread)

    axialThinpath = filespath + 'axialThin-test.txt'
    val_GTfile = open(axialThinpath)
    val_GTread = val_GTfile.read().splitlines()

    with tf.name_scope('input'):
        LR = tf.placeholder(tf.float32, shape=[batch_size, seq_length, 360, 432, 3], name='Lowresolute_image')

    # conv network
    hidden = None
    x_1, hidden = network_template(LR[:, 0, :, :, :], hidden)
    x_2, hidden = network_template(LR[:, 1, :, :, :], hidden)

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(args.gpu)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, modelpath)
    print("Model loaded")

    for kv in range(0, ntest):

        time_start = time.time()
        print('Loading from test case ' + str(kv + 1) + ' for stage 2 for test')

        matinput0 = val_axialread[kv]
        load_data_input0 = sio.loadmat(matinput0)
        datainput0 = load_data_input0['T1_image']
        Input_full0cut = datainput0[int(datainput0.shape[0] / 2 - 180):int(datainput0.shape[0] / 2 + 180), int(datainput0.shape[1] / 2 - 216):int(datainput0.shape[1] / 2 + 216), :]
        datainput0 = np.transpose(Input_full0cut, [2, 0, 1])

        matinput1 = savepath + str(kv + 1) + '/Reconstruction_BrainStructureAwareModel.mat'
        load_data_input1 = sio.loadmat(matinput1)
        datainput1 = load_data_input1['Reconstruction']
        datainput1m = datainput1[:, :, :, 0]
        Input_fullrcut = datainput1m[int(datainput1m.shape[0] / 2 - 180):int(datainput1m.shape[0] / 2 + 180), int(datainput1m.shape[1] / 2 - 216):int(datainput1m.shape[1] / 2 + 216), :]
        datainput1 = np.transpose(Input_fullrcut, [2, 0, 1])

        test_sagittalname = val_sagittalread[kv]
        load_data_input2 = sio.loadmat(test_sagittalname)
        datainput2m = load_data_input2['T2s2_image']
        Input_full2cut = datainput2m[int(datainput1m.shape[0] / 2 - 180):int(datainput1m.shape[0] / 2 + 180), int(datainput1m.shape[1] / 2 - 216):int(datainput1m.shape[1] / 2 + 216), :]
        if Input_full2cut.shape[2] < 2 * datainput1m.shape[2]:
            Input_full2cut = np.dstack((Input_full2cut, Input_full2cut[:, :, Input_full2cut.shape[2] - 1]))

        datainputsag = np.transpose(Input_full2cut, [2, 0, 1])

        totalnum = datainput1.shape[0]

        def feed_dict(j):
            pointer = j
            xs = np.zeros((batch_size, seq_length, 360, 432, 3))
            xs[:, 0, :, :, 0] = datainput1[pointer, 0:360, 0:432]
            xs[:, 0, :, :, 1] = datainput0[pointer, 0:360, 0:432]
            xs[:, 0, :, :, 2] = datainputsag[2 * pointer, 0:360, 0:432]
            xs[:, 1, :, :, 0] = datainput1[pointer, 0:360, 0:432]
            xs[:, 1, :, :, 1] = datainput0[pointer, 0:360, 0:432]
            xs[:, 1, :, :, 2] = datainputsag[2 * pointer + 1, 0:360, 0:432]
            return {LR: xs}

        hp = datainput1m
        for j in range(0, np.int16(totalnum)):
            ht = sess.run(x_2, feed_dict=feed_dict(j))
            hp[int(datainput1m.shape[0] / 2 - 180):int(datainput1m.shape[0] / 2 + 180), int(datainput1m.shape[1] / 2 - 216):int(datainput1m.shape[1] / 2 + 216), j] = ht[0, :, :, 0]

        time_end = time.time()
        print('Time cost of test at case ' + str(kv + 1) + ' for stage 2 has been ' + str(time_end - time_start) + ' s')
        savename = '%s%s%s' % (savepath, str(kv + 1), '//Reconstruction_DeepVolume.mat')
        sio.savemat(savename, {'Reconstruction': hp})

        # load the brain mask, which was generated based on the axial thin MRI
        c1map = val_GTread[kv][0:-4] + 'c1.nii'
        c1load = nib.load(c1map)
        c1im = c1load.get_fdata()

        c2map = val_GTread[kv][0:-4] + 'c2.nii'
        c2load = nib.load(c2map)
        c2im = c2load.get_fdata()

        c3map = val_GTread[kv][0:-4] + 'c3.nii'
        c3load = nib.load(c3map)
        c3im = c3load.get_fdata()

        cim = c1im + c2im + c3im
        RecIntensity = np.abs((hp * Intensity_std + Intensity_mean) * Intensity_max)
        imgToSave = np.int16(RecIntensity * cim)

        npDtype = np.dtype(np.int16)
        proxy_origin = nib.load(c1map)
        affine_origin = proxy_origin.affine
        proxy_origin.uncache()

        newImg = nib.Nifti1Image(imgToSave, affine_origin)
        newImg.set_data_dtype(npDtype)

        nib.save(newImg, savepath + str(kv + 1) + '//pred.nii.gz')

        print('################ case ' + str(kv + 1) + ' has been done for Spatial Connection Aware Model ################')


    sess.close()

def evaluation():
    filespath = args.datafile
    savepath = args.savepath + 'test'
    axialThinpath = filespath + 'axialThin-test.txt'
    val_GTfile = open(axialThinpath)
    val_GTread = val_GTfile.read().splitlines()

    ntest = len(val_GTread)

    PSNRall = []

    print('################################ Doing evaluation ################################')

    for kv in range(0, ntest):

        predmap = savepath + str(kv + 1) + '//pred.nii.gz'
        predload = nib.load(predmap)
        predim = np.uint8(predload.get_fdata())

        matGT = val_GTread[kv]
        load_data_input0 = sio.loadmat(matGT)
        dataGT = load_data_input0['T3_image']

        GTIntensity = (dataGT * Intensity_std + Intensity_mean) * Intensity_max

        c1map = val_GTread[kv][0:-4] + 'c1.nii'
        c1load = nib.load(c1map)
        c1im = c1load.get_fdata()

        c2map = val_GTread[kv][0:-4] + 'c2.nii'
        c2load = nib.load(c2map)
        c2im = c2load.get_fdata()

        c3map = val_GTread[kv][0:-4] + 'c3.nii'
        c3load = nib.load(c3map)
        c3im = c3load.get_fdata()

        cim = c1im + c2im + c3im

        GTim = np.uint8(GTIntensity * cim)

        Resultpsnr = psnr(predim, GTim)
        PSNRall.append(Resultpsnr)
        print('PSNR of case ' + str(kv+1) + ' is ' + str(Resultpsnr))

    print('average PSNR is ' + str(np.mean(PSNRall)))


def psnr(img1, img2):
    mse = np.mean((np.double(img1) - np.double(img2)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
    if args.stage == 1:
        test_BrainStructureNetwork()
    else:
        test_SpatialConnectionAwareNetwork()
        evaluation()
