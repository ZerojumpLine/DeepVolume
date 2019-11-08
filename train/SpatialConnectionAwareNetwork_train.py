import numpy as np
import tensorflow as tf
from SpatialConnectionAwareNetwork_arch import SpatialConnectionAwareNetwork
import h5py
import time
import logging
import argparse

parser = argparse.ArgumentParser(description='Tensorflow DeepVolume Test')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--datafile', default='../datafile/', type=str, help='path to datafile folder')
parser.add_argument('--savepath', default='../output/', type=str, help='path to output folder')
parser.add_argument('--modelpath', default='../models/', type=str, help='path to model save folder')
parser.add_argument('--samplingpath', default='../preprocessing/SamplingForSpatialConnectionAwareNetwork/', type=str, help='path to model save folder')
parser.add_argument('-b', '--batch_size', type=int, default=2, help='The batch size of the training')
parser.add_argument('--start_epoch', type=int, default=0, help='Set the starting epoch of the training')
parser.add_argument('--epoch', type=int, default=30, help='Number of training epoch')
parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training", action="store_true")

args = parser.parse_args()


def train():

  batch_size = args.batch_size
  Samplingpath = args.samplingpath
  start_epoch = args.start_epoch
  epoch = args.epoch
  filespath = args.datafile

  # make a template for reuse
  network_template = tf.make_template('network', SpatialConnectionAwareNetwork)

  logname = 'TrainSpatialConnectionAwareNetwork_log.log'

  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                      datefmt='%a, %d %b %Y %H:%M:%S',
                      filename=logname,
                      filemode='a')

  config = tf.ConfigProto()
  config.gpu_options.visible_device_list = str(args.gpu)
  sess = tf.InteractiveSession(config=config)

  with tf.name_scope('input'):
    LR = tf.placeholder(tf.float32, shape=[batch_size, 2, 360, 432, 3], name='Lowresolute_image')
    HR = tf.placeholder(tf.float32, shape=[batch_size, 360, 432, 1], name='Highresolute_image')

  # conv network
  hidden = None
  x_1, hidden = network_template(LR[:, 0, :, :, :], hidden, batch_size)
  x_2, hidden = network_template(LR[:, 1, :, :, :], hidden, batch_size)
  
  with tf.name_scope('loss_function'):
    loss1 = tf.losses.mean_squared_error(HR, x_1)
    loss2 = tf.losses.mean_squared_error(HR, x_2)
    loss = loss1 + loss2

  with tf.name_scope('trainer'):
    trainer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

  sess.run(tf.global_variables_initializer())

  if args.continue_training == True:
    saver = tf.train.Saver()
    savepath = args.modelpath + 'SpatialConnectionAwareNetwork/Model'
    saver.restore(sess, savepath + str(start_epoch) + '.ckpt')
    print("Model loaded")

  def feed_dict(j, k):
    pointer = j
    xs = np.zeros((batch_size, 2, 360, 432, 3))
    ys = np.zeros((batch_size, 360, 432, 1))
    xs[:, 0, :, :, 0] = datainput1[batch_size * k:batch_size * (k + 1), pointer, 0:360, 0:432]
    xs[:, 0, :, :, 1] = datainput0[batch_size * k:batch_size * (k + 1), pointer, 0:360, 0:432]
    xs[:, 0, :, :, 2] = datainputsag[batch_size * k:batch_size * (k + 1), 2 * pointer, 0:360, 0:432]
    xs[:, 1, :, :, 0] = datainput1[batch_size * k:batch_size * (k + 1), pointer, 0:360, 0:432]
    xs[:, 1, :, :, 1] = datainput0[batch_size * k:batch_size * (k + 1), pointer, 0:360, 0:432]
    xs[:, 1, :, :, 2] = datainputsag[batch_size * k:batch_size * (k + 1), 2 * pointer + 1, 0:360, 0:432]
    ys[:, :, :, 0] = dataoutput[batch_size * k:batch_size * (k + 1), pointer, 0:360, 0:432]
    return {LR: xs, HR: ys}

  for i in range(start_epoch, epoch):
    time_start = time.time()

    logging.info('++++++++++++++++++++++++++++++++++++ Start training at epoch ' + str(i + 1) + ' ++++++++++++++++++++++++++++++++++++')
    print('++++++++++++++++++++++++++++++++++++ Start training at epoch ' + str(i + 1) + ' ++++++++++++++++++++++++++++++++++++')

    train_axialThickpath = filespath + 'axialThick-train.txt'
    train_axialfile = open(train_axialThickpath)
    train_axialread = train_axialfile.read().splitlines()
    ntrain = len(train_axialread)

    lossall = np.zeros(ntrain * 140)
    ncounti = 0

    matinput0 = Samplingpath + 'train-1-total_Axial2.mat'
    file0 = h5py.File(matinput0)
    datainput0 = file0['total_Axial2'][()]
    logging.info('Data Axial2 loaded')
    print('Data Axial2 loaded')

    matinput1 = Samplingpath + 'train-1-total_Reconstruction.mat'
    file1 = h5py.File(matinput1)
    datainput1 = file1['total_Reconstruction'][()]
    logging.info('Data Reconstruction loaded')
    print('Data Reconstruction loaded')

    matinput_sag = Samplingpath + 'train-1-totalSag.mat'
    file2 = h5py.File(matinput_sag)
    datainputsag = file2['total_Sag'][()]
    logging.info('Data Sagittal loaded')
    print('Data Sagittal loaded')

    matoutput = Samplingpath + 'train-1-total_Axial3.mat'
    file3 = h5py.File(matoutput)
    dataoutput = file3['total_Axial3'][()]
    logging.info('Data GT loaded')
    print('Data GT loaded')

    totalnum = datainput1.shape[1]
    nbatch = np.ceil(datainput1.shape[0] / batch_size)
    nbatch = nbatch.astype(np.int16)

    losss = np.zeros(nbatch*totalnum)

    for k in range(0,nbatch):
      for j in range(0, np.int16(totalnum)):
        _, lossp = sess.run([trainer, loss], feed_dict=feed_dict(j, k))
        lossall[ncounti] = lossp
        ncounti = ncounti + 1
        losss[k * totalnum + j] = lossp

    time_end = time.time()
    logging.info('Loss mean of epoch ' + str(i + 1) + ' is ' + str(np.mean(losss)))
    print('Loss mean of epoch ' + str(i + 1) + ' is ' + str(np.mean(losss)))
    logging.info('Loss std of epoch ' + str(i + 1) + ' is ' + str(np.std(losss)))
    print('Loss std of epoch ' + str(i + 1) + ' is ' + str(np.std(losss)))
    logging.info('Time cost of training at epoch ' + str(i + 1) + ' has been ' + str(time_end - time_start) + ' s')
    print('Time cost of training at epoch ' + str(i + 1) + ' has been ' + str(time_end - time_start) + ' s')


    saver = tf.train.Saver()
    savepath = args.modelpath + 'SpatialConnectionAwareNetwork/Model'
    model_path = '%s%s%s' % (savepath, i + 1, '.ckpt')
    saver.save(sess, model_path)
    logging.info('Models saved for epoch ' + str(i + 1))
    print('Models saved for epoch ' + str(i + 1))
    



if __name__ == '__main__':
  train()
