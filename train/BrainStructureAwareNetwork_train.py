import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
import time
import logging
from BrainStructureAwareNetwork_arch import BrainStructureAwareNetwork
import h5py
from getimgval import getimgval
from getimgtest import getimgtest
import argparse

parser = argparse.ArgumentParser(description='Tensorflow DeepVolume Test')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--datafile', default='../datafile/', type=str, help='path to datafile folder')
parser.add_argument('--savepath', default='../output/', type=str, help='path to output folder')
parser.add_argument('--modelpath', default='../models/', type=str, help='path to model save folder')
parser.add_argument('--samplingpath', default='../preprocessing/SamplingForBrainStructureAwareNetwork/', type=str, help='path to model save folder')
parser.add_argument('-b', '--batch_size', type=int, default=10, help='The batch size of the training')
parser.add_argument('--start_epoch', type=int, default=0, help='Set the starting epoch of the training')
parser.add_argument('--epoch', type=int, default=100, help='Number of training epoch')
parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training", action="store_true")
parser.add_argument("-val", "--do_validation", help="use this if you want to do validation during training", action="store_true")
parser.add_argument("-train", "--do_training", help="use this if you want to do training", action="store_true")

args = parser.parse_args()

def train_BrainStructureNetwork():

    batch_size = args.batch_size
    start_epoch = args.start_epoch
    epoch = args.epoch
    Samplingpath = args.samplingpath
    filespath = args.datafile

    logname = 'TrainBrainStructureAwareNetwork_log.log'

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=logname,
                        filemode='a')

    val_axialThinpath = filespath + 'axialThin-val.txt'
    val_axialThickpath = filespath + 'axialThick-val.txt'
    val_sagittalThickpath = filespath + 'sagittalThick-val.txt'

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(args.gpu)
    sess = tf.InteractiveSession(config=config)

    with tf.name_scope('input'):
        LR = tf.placeholder(tf.float32, shape=[None, None, None, None, 2], name='Lowresolute_image')
        HR = tf.placeholder(tf.float32, shape=[None, None, None, None, 1], name='Highresolute_image')
        Labels = tf.placeholder(tf.float32, shape=[None, None, None, None, 3], name='Segmentation_label')
        keep_prob = tf.placeholder(tf.float32, name='dropout_ratio')

    probs, logits = BrainStructureAwareNetwork(LR, keep_prob)

    with tf.name_scope('loss_function'):
        loss_recon = tf.losses.mean_squared_error(HR, probs)
        loss_seg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Labels))
        loss = loss_recon + loss_seg

    with tf.name_scope('global_step'):
        global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('trainer'):
        starter_learning_rate = 1e-3
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96)
        trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    sess.run(tf.global_variables_initializer())


    if args.continue_training == True:
        saver = tf.train.Saver()
        savepath = args.modelpath + 'BrainStructureAwareNetwork/model'
        saver.restore(sess, savepath + str(start_epoch) + '.ckpt')
        print("Model loaded")

    def feed_dict(datainput1, datainput2, dataoutput, dataoutput2, k):
        start = batch_size * k
        xs = np.zeros((batch_size, 64, 64, 64, 2))
        ys = np.zeros((batch_size, 64, 64, 64, 1))
        seg = np.zeros((batch_size, 64, 64, 64, 3))
        xs[0:batch_size, :, :, :, 0] = datainput1[start:start + batch_size, :, :, :]
        xs[0:batch_size, :, :, :, 1] = datainput2[start:start + batch_size, :, :, :]
        ys[0:batch_size, :, :, :, 0] = dataoutput[start:start + batch_size, :, :, :]
        seg[0:batch_size, :, :, :, :] = dataoutput2[start:start + batch_size, :, :, :]
        return {LR: xs, HR: ys, Labels: seg, keep_prob: 1}

    def feed_dict_val(xstart, ystart, zstart, LRimg):
        xs = np.zeros((1, 200, 200, 200, 2))
        xs[:, :, :, :, :] = LRimg[:, xstart:xstart + 200, ystart:ystart + 200, zstart:zstart + 200, :]
        return {LR: xs, keep_prob: 1}

    # all sampling is done at first
    # too large to load at a time

    if args.do_training :

        for i in range(start_epoch, epoch):  # epoch

            if args.do_validation :

                ########################################################  validation starts every 5 epoch ##############################################################

                if i % 5 == 0:  # validation
                    logging.info('Start validation at epoch ' + str(i + 1))
                    print('Start validation at epoch ' + str(i + 1))

                    val_axialfile = open(val_axialThickpath)
                    val_axialread = val_axialfile.read().splitlines()
                    nval = len(val_axialread)

                    valMSE = np.zeros(nval)

                    # sample patches with tumor
                    for kv in range(0, nval):

                        logging.info('Loading from val case ' + str(kv + 1) + ' for validation')
                        print('Loading from val case ' + str(kv + 1) + ' for validation')

                        LRimg, HRimg = getimgval(val_axialThinpath, val_axialThickpath, val_sagittalThickpath, kv)

                        x_range = LRimg.shape[1]
                        y_range = LRimg.shape[2]
                        z_range = LRimg.shape[3]

                        if z_range < 200:
                            LRimgpad = np.zeros((1, x_range, y_range, 200, 2))
                            LRimgpad[:, :, :, 0:z_range, :] = LRimg
                            LRimg = LRimgpad

                        # The receptive field of 3D U-net is 68
                        # We should retrive the center 40^3 pixel of 200^3 to reconstruct

                        if z_range < 200:
                            hp = np.zeros((x_range, y_range, 200, 1))
                        else:
                            hp = np.zeros((x_range, y_range, z_range, 1))
                        x_sample = np.floor((x_range - 160) / 40) + 1
                        x_sample = x_sample.astype(np.int16)
                        y_sample = np.floor((y_range - 160) / 40) + 1
                        y_sample = y_sample.astype(np.int16)
                        z_sample = np.maximum(np.floor((z_range - 160) / 40) + 1, 1)
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

                                    ht = sess.run(probs, feed_dict=feed_dict_val(xstart, ystart, zstart, LRimg))
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
                                        hp[xstart:xstart + 120, ystart + 80:ystart + 120, zstart:zstart + 120, :] = ht[0, 0:120, 80:120, 0:120, :]
                                    if jx == 0 and jz == z_sample - 1:
                                        hp[xstart:xstart + 120, ystart + 80:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 0:120, 80:120, 80:200, :]
                                    if jx == x_sample - 1 and jz == 0:
                                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 120, zstart:zstart + 120, :] = ht[0, 80:200, 80:120, 0:120, :]
                                    if jx == x_sample - 1 and jz == z_sample - 1:
                                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 80:200, 80:120, 80:200, :]
                                    # then the 4 corner...yz
                                    if jy == 0 and jz == 0:
                                        hp[xstart + 80:xstart + 120, ystart:ystart + 120, zstart:zstart + 120, :] = ht[0, 80:120, 0:120, 0:120, :]
                                    if jy == 0 and jz == z_sample - 1:
                                        hp[xstart + 80:xstart + 120, ystart:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 80:120, 0:120, 80:200, :]
                                    if jy == y_sample - 1 and jz == 0:
                                        hp[xstart + 80:xstart + 120, ystart + 80:ystart + 200, zstart:zstart + 120, :] = ht[0, 80:120, 80:200, 0:120, :]
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

                            print('processing validation.. ' + str(jx / x_sample * 100) + '%')

                        if z_range < 200:
                            hp = hp[:, :, 0:z_range]

                        valMSE[kv] = (np.square(HRimg - hp)).mean(axis=None)
                        print('val MSE of case ' + str(kv + 1) + ' is ' + str(valMSE[kv]))
                        logging.info('val MSE of case ' + str(kv + 1) + ' is ' + str(valMSE[kv]))

                    print('$$$$$$$$$$$$$$$$$$$$$$$$$ epoch ' + str(i + 1) + ' val MSE mean is ' + str(np.mean(valMSE)) + ' $$$$$$$$$$$$$$$$$$$$$$$$$')
                    logging.info('$$$$$$$$$$$$$$$$$$$$$$$$$ epoch ' + str(i + 1) + ' val MSE mean is ' + str(np.mean(valMSE)) + ' $$$$$$$$$$$$$$$$$$$$$$$$$')
                    print('$$$$$$$$$$$$$$$$$$$$$$$$$ epoch ' + str(i + 1) + ' val MSE std is ' + str(np.std(valMSE)) + ' $$$$$$$$$$$$$$$$$$$$$$$$$')
                    logging.info('$$$$$$$$$$$$$$$$$$$$$$$$$ epoch ' + str(i + 1) + ' val MSE std is ' + str(np.std(valMSE)) + ' $$$$$$$$$$$$$$$$$$$$$$$$$')

            ####################################################  validation ends ##################################################################################

            ######################################################  training starts ################################################################################

            time_start = time.time()

            logging.info('++++++++++++++++++++++++++++++++++++ Start training at epoch ' + str(i + 1) + ' ++++++++++++++++++++++++++++++++++++')
            print('++++++++++++++++++++++++++++++++++++ Start training at epoch ' + str(i + 1) + ' ++++++++++++++++++++++++++++++++++++')

            train_axialThickpath = filespath + 'axialThick-train.txt'
            train_axialfile = open(train_axialThickpath)
            train_axialread = train_axialfile.read().splitlines()
            ntrain = len(train_axialread)

            lossallr = np.zeros(ntrain * 100)
            lossalls = np.zeros(ntrain * 100)
            ncounti = 0

            matinput1 = Samplingpath + 'train-1-T1r.mat'
            file1 = h5py.File(matinput1)
            datainput1 = file1['T1r']

            matinput2 = Samplingpath + 'train-1-T2r.mat'
            file2 = h5py.File(matinput2)
            datainput2 = file2['T2r']

            matoutput = Samplingpath + 'train-1-T3r.mat'
            file3 = h5py.File(matoutput)
            dataoutput = file3['T3r']

            matoutput2 = Samplingpath + 'train-1-T4r.mat'
            file4 = h5py.File(matoutput2)
            dataoutput2 = file4['T4r']

            logging.info('Data loaded')
            print('Data loaded')

            nbatch = np.floor(datainput1.shape[0] / batch_size)
            nbatch = nbatch.astype(np.int16)

            lossrec = np.zeros(nbatch)
            lossseg = np.zeros(nbatch)

            for k in range(0, nbatch):  # batch
                _, lossr, losss = sess.run([trainer, loss_recon, loss_seg], feed_dict=feed_dict(datainput1, datainput2, dataoutput, dataoutput2, k))
                lossallr[ncounti] = lossr
                lossalls[ncounti] = losss
                ncounti = ncounti + 1
                lossrec[k] = lossr
                lossseg[k] = losss

                if (k + 1) % 100 == 0:
                    logging.info('Processing epoch ' + str(i + 1) + ' subepoch ' + str(j + 1) + ' batch ' + str(k + 1))
                    print('Processing epoch ' + str(i + 1) + ' subepoch ' + str(j + 1) + ' batch ' + str(k + 1))

            time_end = time.time()
            logging.info('Reconstruction loss mean of epoch ' + str(i + 1) + ' is ' + str(np.mean(lossrec)))
            print('Reconstruction loss mean of epoch ' + str(i + 1) + ' is ' + str(np.mean(lossrec)))
            logging.info('Reconstruction loss std of epoch ' + str(i + 1) + ' is ' + str(np.std(lossrec)))
            print('Reconstruction loss std of epoch ' + str(i + 1) + ' is ' + str(np.std(lossrec)))
            logging.info('Segmentation loss mean of epoch ' + str(i + 1) + ' is ' + str(np.mean(lossseg)))
            print('Segmentation loss mean of epoch ' + str(i + 1) + ' is ' + str(np.mean(lossseg)))
            logging.info('Segmentation loss std of epoch ' + str(i + 1) + ' is ' + str(np.std(lossseg)))
            print('Segmentation loss std of epoch ' + str(i + 1) + ' is ' + str(np.std(lossseg)))
            logging.info('Learning rate of epoch ' + str(i + 1) + ' is ' + str(sess.run(learning_rate)))
            print('Learning rate of epoch ' + str(i + 1) + ' is ' + str(sess.run(learning_rate)))
            logging.info('Time cost of training at epoch ' + str(i + 1) + ' has been ' + str(time_end - time_start) + ' s')
            print('Time cost of training at epoch ' + str(i + 1) + ' has been ' + str(time_end - time_start) + ' s')

            ######################################################  training ends ##################################################################################

            # Memory release
            del datainput1
            del datainput2
            del dataoutput
            del dataoutput2

            saver = tf.train.Saver()
            savepath = args.modelpath + 'BrainStructureAwareNetwork/Model'
            model_path = '%s%s%s' % (savepath, i + 1, '.ckpt')
            saver.save(sess, model_path)
            logging.info('Models saved for epoch ' + str(i + 1))
            print('Models saved for epoch ' + str(i + 1))

    ############################################  do evaluation on the trianing data, which prepares for stage 2 ############################################

    axialThickpath = filespath + 'axialThick-train.txt'
    sagittalThickpath = filespath + 'sagittalThick-train.txt'
    savepath = args.savepath + 'train'
    modelpath = args.modelpath + 'BrainStructureAwareNetwork/Model' + str(args.epoch) + '.ckpt'

    val_axialfile = open(axialThickpath)
    val_axialread = val_axialfile.read().splitlines()
    ntest = len(val_axialread)

    testsavepath = savepath + str(1)
    if (os.path.isdir(testsavepath) == False):
        for k in range(0, ntest):
            os.mkdir(savepath + str(k + 1))
        print("Folder created")

    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(args.gpu)
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, modelpath)
    print("Model loaded")

    # sample patches with tumor
    for kv in range(0, ntest):
        time_start = time.time()
        print('Loading from training case ' + str(kv + 1) + ' for test for stage 1')

        LRimg = getimgtest(axialThickpath, sagittalThickpath, kv)

        x_range = LRimg.shape[1]
        y_range = LRimg.shape[2]
        z_range = LRimg.shape[3]

        if z_range < 200:
            LRimgpad = np.zeros((1, x_range, y_range, 200, 2))
            LRimgpad[:, :, :, 0:z_range, :] = LRimg
            LRimg = LRimgpad

        # The receptive field of 3D U-net is 68
        # We should retrive the center 40^3 pixel of 200^3 to reconstruct

        if z_range < 200:
            hp = np.zeros((x_range, y_range, 200, 1))
        else:
            hp = np.zeros((x_range, y_range, z_range, 1))
        x_sample = np.floor((x_range - 160) / 40) + 1
        x_sample = x_sample.astype(np.int16)
        y_sample = np.floor((y_range - 160) / 40) + 1
        y_sample = y_sample.astype(np.int16)
        z_sample = np.maximum(np.floor((z_range - 160) / 40) + 1, 1)
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

                    ht = sess.run(probs, feed_dict=feed_dict_val(xstart, ystart, zstart, LRimg))
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
                        hp[xstart:xstart + 120, ystart + 80:ystart + 120, zstart:zstart + 120, :] = ht[0, 0:120, 80:120, 0:120, :]
                    if jx == 0 and jz == z_sample - 1:
                        hp[xstart:xstart + 120, ystart + 80:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 0:120, 80:120, 80:200, :]
                    if jx == x_sample - 1 and jz == 0:
                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 120, zstart:zstart + 120, :] = ht[0, 80:200, 80:120, 0:120, :]
                    if jx == x_sample - 1 and jz == z_sample - 1:
                        hp[xstart + 80:xstart + 200, ystart + 80:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 80:200, 80:120, 80:200, :]
                    # then the 4 corner...yz
                    if jy == 0 and jz == 0:
                        hp[xstart + 80:xstart + 120, ystart:ystart + 120, zstart:zstart + 120, :] = ht[0, 80:120, 0:120, 0:120, :]
                    if jy == 0 and jz == z_sample - 1:
                        hp[xstart + 80:xstart + 120, ystart:ystart + 120, zstart + 80:zstart + 200, :] = ht[0, 80:120, 0:120, 80:200, :]
                    if jy == y_sample - 1 and jz == 0:
                        hp[xstart + 80:xstart + 120, ystart + 80:ystart + 200, zstart:zstart + 120, :] = ht[0, 80:120, 80:200, 0:120, :]
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

            print('processing Brain Structure Aware Model.. ' + str(jx / x_sample * 100) + '%')

        if z_range < 200:
            hp = hp[:, :, 0:z_range]

        print('processing Brain Structure Aware Model.. ' + '100%')

        time_end = time.time()
        print('Time cost of test at case ' + str(kv + 1) + '  for stage 1 has been ' + str(time_end - time_start) + ' s')
        savename = '%s%s%s' % (savepath, str(kv + 1), '/Reconstruction_BrainStructureAwareModel.mat')
        sio.savemat(savename, {'Reconstruction': hp})
        print('################ case ' + str(kv + 1) + ' has been done for Brain Structure Aware Model ################')
    sess.close()


if __name__ == '__main__':
    train_BrainStructureNetwork()
